import contextvars
import logging
import os
import pickle
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
from typing import Any, Callable

import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException

from .checkpoint_base import CheckpointWriteInfo
from .checkpoint_writer import CheckpointWriterArgs
from .logging_utils import checkpoint_logging_context, EventLogger, EventType
from .types import RankInfo
from .utils import set_thread_name_safe

logger = logging.getLogger(__name__)

# Payload key constants to avoid string literal bugs
PAYLOAD_KEY_PATH = "path"
PAYLOAD_KEY_CHECKPOINT_INFO = "checkpoint_info"
PAYLOAD_KEY_LOGGING_CONTEXT = "logging_context"


@dataclass
class CheckpointProcessConfig:
    """
    Configuration options for the CheckpointProcess.

    This class provides configuration options for the checkpoint process,
    including initialization functions, timeouts, and writer configuration.

    Attributes:
        subprocess_init_timeout_secs: Maximum time in seconds to wait for subprocess initialization.
        subprocess_shutdown_timeout_secs: Maximum time in seconds to wait for subprocess shutdown.
    """

    subprocess_init_timeout_secs: int = 30
    subprocess_shutdown_timeout_secs: int = 60
    thread_name_prefix: str = "ckpt"


class RequestType(Enum):
    PING = "ping"
    WRITE_CHECKPOINT = "write_checkpoint"
    TERMINATE_PROCESS = "exit"


@dataclass
class WorkerRequest:
    """
    A dataclass for storing the command to be sent to the worker process.
    Note: This relies on pickling to send the command to the worker process. Handle
    backward compatibility accordingly.
    """

    request_type: RequestType
    payload: dict[str, Any]


@dataclass
class WorkerResponse:
    request_type: RequestType
    success: bool
    error_msg: str | None = None
    payload: dict[str, Any] | None = None


class CheckpointProcess:
    """
    A checkpoint writer that writes checkpoints to a remote process.
    """

    def __init__(
        self,
        rank_info: RankInfo,
        config: CheckpointProcessConfig,
        subprocess_init_fn: Callable[[Any], None],
        subprocess_init_args: tuple[Any, ...],
        checkpoint_writer_args: CheckpointWriterArgs,
    ):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._rank_info = rank_info
        self._config = config
        self._subprocess_init_fn = subprocess_init_fn
        self._subprocess_init_args = subprocess_init_args
        self._checkpoint_writer_args = checkpoint_writer_args
        self._metric_prefix = checkpoint_writer_args.metric_prefix
        self.process: mp.ProcessContext | None = None
        self._parent_end: Connection | None = None
        self._child_end: Connection | None = None
        self._closed: bool = False
        self._pickle_protocol: int = pickle.DEFAULT_PROTOCOL
        if self._pickle_protocol < 5 and pickle.HIGHEST_PROTOCOL >= 5:
            # Use more efficient protocol 5 if available
            self._pickle_protocol = 5

        self._creation_future: Future[None] | None = self._executor.submit(
            self._create_subprocess,
            config,
        )

    def wait_for_init(self) -> None:
        if self._creation_future:
            self._creation_future.result()

    def check_ok(self) -> None:
        if self._creation_future and self._creation_future.done():
            self._creation_future.result()

        if (
            not self._closed
            and self.process
            and not self.process.processes[0].is_alive()
        ):
            raise RuntimeError(
                f"Child checkpointer process pid={self.process.processes[0].pid} died unexpectedly with exit code={self.process.processes[0].exitcode}"
            )

    def _create_subprocess(
        self,
        config: CheckpointProcessConfig,
    ) -> None:
        logger.info(f"Creating checkpoint subprocess for rank={self._rank_info}")
        set_thread_name_safe(
            f"{self._config.thread_name_prefix}-comm-{self._rank_info.global_rank}"
        )

        spawn_context = mp.get_context("spawn")
        self._parent_end, child_end = spawn_context.Pipe()

        # Known workaround for https://github.com/pytorch/pytorch/issues/37377
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "GNU"

        self.process = mp.spawn(
            fn=CheckpointProcess._subprocess,
            args=(
                self._rank_info,
                child_end,
                self._subprocess_init_fn,
                self._subprocess_init_args,
                self._checkpoint_writer_args,
                self._config.thread_name_prefix,
                self._metric_prefix,
            ),
            nprocs=1,
            join=False,
            daemon=True,
        )

        # close the child end of the pipe so recv on it will fail
        # fast when the child process is terminated unexpectedly.
        child_end.close()
        self._send(
            request_type=RequestType.PING,
            payload={},
        )

        logger.info(
            f"Waiting for checkpoint subprocess to initialize (timeout: {config.subprocess_init_timeout_secs}s)"
        )

        # wait for the timeout or a response from subprocess
        assert self._parent_end is not None, "Parent end of pipe should be initialized"
        if not self._parent_end.poll(timeout=config.subprocess_init_timeout_secs):
            msg = f"Timed out after {config.subprocess_init_timeout_secs}s waiting for checkpoint subprocess to initialize"
            logger.error(msg)
            raise TimeoutError(msg)

        self._recv()
        logger.info("Checkpoint subprocess initialized successfully")

    @staticmethod
    def _subprocess(
        sub_rank: int,
        rank_info: RankInfo,
        parent_pipe: Connection,
        subprocess_init_fn: Callable[[Any], None],
        subprocess_init_args: tuple[Any, ...],
        checkpoint_writer_args: CheckpointWriterArgs,
        thread_name_prefix: str = "ckpt",
        metric_prefix: str = "train.checkpoint_write",
    ) -> None:
        logger.info(
            f"Checkpoint subprocess started for rank {rank_info.global_rank}/{rank_info.global_world_size} (PID: {os.getpid()})"
        )
        set_thread_name_safe(f"{thread_name_prefix}-proc-{rank_info.global_rank}")

        assert sub_rank == 0, "We need only one checkpointer per parent training"
        request = WorkerRequest(request_type=RequestType.PING, payload={})

        # Cache for serialized metadata - once received, reuse for subsequent writes
        cached_serialized_metadata: bytes | None = None

        try:
            event_logger = EventLogger()
            # Calling initialize callback, so we can perform app-specific initialization of the subprocess.
            subprocess_init_fn(*subprocess_init_args)
            logger.info(
                f"Starting: initializing sub-process for checkpointing for sub rank {sub_rank}",
                extra=event_logger(EventType.CHECKPOINT_SUBPROCESS_INIT_START),
            )

            # Initialize checkpoint writer via builder pattern.
            checkpoint_writer = checkpoint_writer_args.build()

            logger.info(
                "Checkpointing subprocess initialized",
                extra=event_logger(
                    EventType.CHECKPOINT_SUBPROCESS_INIT_END,
                    metric_name=f"{metric_prefix}.execute.subprocess_execute.initialize_e2e.latency_ms",
                    end_to_end=True,
                ),
            )

            while True:
                logger.info(
                    f"Waiting for a checkpoint request for sub rank {sub_rank}",
                    extra=event_logger(EventType.CHECKPOINT_WAIT_FOR_REQUEST),
                )
                # Safe to use recv() here because the parent sends a the byes of a
                # pickled object
                try:
                    request = parent_pipe.recv()
                except EOFError:
                    # Die quietly rather than loudly to avoid log spam
                    logger.info(
                        "Checkpoint subprocess: Parent has closed or exited, therefore I should exit."
                    )
                    break

                log_ctx = request.payload.get(PAYLOAD_KEY_LOGGING_CONTEXT)
                if log_ctx is not None:
                    checkpoint_logging_context.import_context(log_ctx)

                if request.request_type == RequestType.PING:
                    parent_pipe.send(
                        WorkerResponse(request_type=RequestType.PING, success=True)
                    )
                elif request.request_type == RequestType.WRITE_CHECKPOINT:
                    path = request.payload[PAYLOAD_KEY_PATH]
                    step = checkpoint_logging_context.get("step")
                    logger.info(
                        f"(step {step}) Writing checkpoint to {path}",
                        extra=event_logger(EventType.CHECKPOINT_WRITE_START),
                    )

                    checkpoint_info: CheckpointWriteInfo = request.payload[
                        PAYLOAD_KEY_CHECKPOINT_INFO
                    ]

                    # Cache serialized metadata on first write that has it
                    if (
                        cached_serialized_metadata is None
                        and checkpoint_info.serialized_distributed_metadata is not None
                    ):
                        cached_serialized_metadata = (
                            checkpoint_info.serialized_distributed_metadata
                        )
                        logger.debug(
                            "Cached serialized_distributed_metadata for subsequent writes"
                        )
                    checkpoint_info = checkpoint_info.for_writes(
                        cached_serialized_metadata
                    )

                    checkpoint_writer.write(
                        path=path,
                        checkpoint_info=checkpoint_info,
                    )

                    logger.info(
                        f"(step {step}) Checkpoint written successfully to {path}",
                        extra=event_logger(EventType.CHECKPOINT_WRITE_END),
                    )
                    parent_pipe.send(
                        WorkerResponse(RequestType.WRITE_CHECKPOINT, success=True)
                    )
                elif request.request_type == RequestType.TERMINATE_PROCESS:
                    logger.debug("Received termination request.")
                    parent_pipe.send(
                        WorkerResponse(RequestType.TERMINATE_PROCESS, success=True)
                    )
                    logger.info("Subprocess terminated gracefully")
                    break
                else:
                    error_msg = f"Unknown request type: {request.request_type}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        except Exception as e:
            error_text = traceback.format_exc()
            logger.error(f"Exception in subprocess  ({type(e).__name__}): {error_text}")

            # Communicating exception via the queue to the main process
            parent_pipe.send(
                WorkerResponse(
                    request_type=request.request_type,
                    success=False,
                    error_msg=error_text,
                )
            )
            parent_pipe.close()
            logger.error(f"Subprocess terminated due to exception: {e}")

    def _send(self, request_type: RequestType, payload: dict[str, Any]) -> None:
        # Attach checkpoint logging context to every request so the subprocess
        # always has up-to-date context regardless of request type.
        payload[PAYLOAD_KEY_LOGGING_CONTEXT] = (
            checkpoint_logging_context.export_context()
        )

        # Only log pickle metric for WRITE_CHECKPOINT requests
        event_logger = (
            EventLogger() if request_type == RequestType.WRITE_CHECKPOINT else None
        )
        pickled_obj = ForkingPickler.dumps(
            WorkerRequest(request_type=request_type, payload=payload),
            protocol=self._pickle_protocol,
        )
        if event_logger:
            logger.info(
                f"Pickled {request_type.value} request ({len(pickled_obj)} bytes)",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"{self._metric_prefix}.execute.subprocess_comm.pickle.latency_ms",
                ),
            )
        try:
            assert self._parent_end is not None, (
                "Parent end of pipe should be initialized"
            )
            self._parent_end.send_bytes(pickled_obj)
        except OSError as e:
            error_msg = "Child process terminated unexpectedly"
            logger.error(
                f"Communication failed during {request_type.value} request: {e}"
            )
            raise RuntimeError(error_msg) from e

    def _recv(self) -> dict[str, Any] | None:
        try:
            assert self._parent_end is not None, (
                "Parent end of pipe should be initialized"
            )
            response = self._parent_end.recv()
            if response.success is False:
                error_msg = (
                    f"Unexpected response from worker process: {response.error_msg}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            return response.payload
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            error_msg = f"Child process terminated unexpectedly: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def write(
        self,
        checkpoint_info: CheckpointWriteInfo | Future[CheckpointWriteInfo],
        path: str,
    ) -> Future[None]:
        event_logger = EventLogger()
        logger.debug("Waiting for subprocess initialization to complete")

        # wait until the process is started
        if self._creation_future:
            self._creation_future.result()
            self._creation_future = None
        # Snapshot the current contextvars (including checkpoint_logging_context
        # step) so the comm thread sees the step that was set at submission
        # time, not whatever the trainer updates it to later.
        ctx = contextvars.copy_context()
        write_future = self._executor.submit(
            ctx.run,
            self._write,
            checkpoint_info,
            path,
        )
        logger.info(
            "Submitted executor",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.submit.executor_submit.latency_ms",
            ),
        )
        return write_future

    def _write(
        self,
        checkpoint_info: CheckpointWriteInfo | Future[CheckpointWriteInfo],
        path: str,
    ) -> None:
        event_logger = EventLogger()
        step = checkpoint_logging_context.get("step")
        logger.info(
            f"(step {step}) Checkpointing communication thread started write to {path}",
            extra=event_logger(EventType.CHECKPOINT_THREAD_START),
        )
        set_thread_name_safe(
            f"{self._config.thread_name_prefix}-comm-{self._rank_info.global_rank}"
        )

        # Wait for checkpoint_info Future to be available
        if isinstance(checkpoint_info, Future):
            logger.debug("Waiting for checkpoint_info Future to resolve")
            ckpt_info = checkpoint_info.result()
        else:
            ckpt_info = checkpoint_info

        logger.info(
            "Future wait latency",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.execute.wait_future.latency_ms",
            ),
        )

        # Log state_dict info only if debug logging is enabled
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"CheckpointInfo contains {len(ckpt_info.keys)} keys")

        start_ts = time.perf_counter()
        self._send(
            request_type=RequestType.WRITE_CHECKPOINT,
            payload={
                PAYLOAD_KEY_CHECKPOINT_INFO: ckpt_info,
                PAYLOAD_KEY_PATH: path,
            },
        )
        end_ts = time.perf_counter()
        diff_ts = end_ts - start_ts
        logger.info(
            "Finished state dict put into subprocess",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.execute.subprocess_comm.put.latency_ms",
                value=diff_ts * 1000,
            ),
        )

        logger.debug("Waiting for write completion response")
        self._recv()
        logger.info(
            f"(step {step}) Checkpointing communication thread finished write to {path}",
            extra=event_logger(
                EventType.CHECKPOINT_THREAD_END,
                end_to_end=True,
                metric_name=f"{self._metric_prefix}.execute.subprocess_comm.e2e.latency_ms",
            ),
        )

    def close(self) -> None:
        logger.info(
            f"Shutting down checkpoint process for rank {self._rank_info.global_rank}"
        )
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

        if self.process is None:
            logger.warning("Checkpoint process did not start, skipping termination!")
            return

        subprocess_pid = self.process.processes[0].pid
        # send graceful termination to sub process
        try:
            if self._parent_end is not None:
                self._parent_end.send(
                    WorkerRequest(
                        request_type=RequestType.TERMINATE_PROCESS,
                        payload={},
                    )
                )
            logger.info(
                f"Sent termination request to subprocess (PID: {subprocess_pid})."
            )
        except BrokenPipeError:
            logger.warning(
                f"BrokenPipeError when sending termination request - subprocess (PID: {subprocess_pid}) may have already terminated"
            )

        try:
            if not self.process.join(
                timeout=self._config.subprocess_shutdown_timeout_secs
            ):
                # graceful shutdown failed, kill the process.
                logger.warning(
                    f"Subprocess (PID: {subprocess_pid}) did not terminate gracefully within {self._config.subprocess_shutdown_timeout_secs}s, killing it"
                )
                self.process.processes[0].kill()
                logger.warning("Subprocess killed forcefully")
        except ProcessExitedException as e:
            # Get additional process information for debugging the root cause
            exit_code = self.process.processes[0].exitcode

            # Use exit code to determine if this was successful termination or an error
            if exit_code != 0:
                message = (
                    f"Checkpoint subprocess (PID: {subprocess_pid}) terminated with exit_code: {exit_code}. ProcessExitedException details: {e}"
                    if exit_code > 0
                    else f"Checkpoint subprocess (PID: {subprocess_pid}) terminated - killed by signal {exit_code} (exit_code: {exit_code}). ProcessExitedException details: {e}"
                )
                logger.warning(message, exc_info=True)
        logger.info(f"Terminated checkpointing process (PID: {subprocess_pid}).")
