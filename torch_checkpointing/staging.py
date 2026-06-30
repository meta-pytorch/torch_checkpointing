"""
Experimental staging module for PyTorch Distributed Checkpointing.

This module provides advanced staging capabilities for checkpoints including:
- Asynchronous staging using ThreadPoolExecutor
- Pinned memory allocation for faster CPU-GPU transfers
- Shared memory support for multi-process scenarios
- Non-blocking CUDA operations with stream synchronization
- Caching of frequently used storages for efficient memory management
- Automatic resource cleanup and memory management

Classes:
    CheckpointStager: Abstract base class defining the staging interface
    StagingOptions: Configuration dataclass for staging behavior
    DefaultStager: Default implementation with comprehensive staging features
"""

import abc
import contextvars
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TypeVar

import torch

from ._state_dict_stager import StateDictStager
from .logging_utils import EventLogger, EventType
from .types import STATE_DICT
from .utils import set_thread_name_safe

T = TypeVar("T")

logger = logging.getLogger(__name__)


class CheckpointStager(abc.ABC):
    """
    Abstract base class for checkpoint staging implementations.

    CheckpointStager defines the interface that all staging implementations
    must follow. Staging is the process of offloading state dictionaries
    for async checkpointing.
    """

    @abc.abstractmethod
    def stage(
        self,
        state_dict: STATE_DICT,
    ) -> STATE_DICT | Future[STATE_DICT]:
        """
        Stage a state dictionary for checkpointing.

        Args:
            state_dict: The state dictionary to stage

        Returns:
            Either a staged state dictionary (synchronous) or a Future
            that will resolve to the staged state dictionary (asynchronous)
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Clean up all resources used by the stager.
        """


@dataclass
class CheckpointStagerConfig:
    """
    Configuration options for checkpoint staging behavior.

    Attributes:
        use_pinned_memory (bool): Enable pinned memory allocation for faster
            CPU-GPU transfers. Requires CUDA to be available. Default: True
        use_shared_memory (bool): Enable shared memory for multi-process
            scenarios. Useful when multiple processes need access to the
            same staged data. Default: True
        use_async_staging (bool): Enable asynchronous staging using a
            background thread pool. Allows overlapping computation with
            staging operations. Requires CUDA. Default: True
        use_non_blocking_copy (bool): Use non-blocking device memory
            copies with stream synchronization. Improves performance by
            allowing CPU work to continue during GPU transfers. Default: True

    Note:
        CUDA-dependent features will raise exception if CUDA is not available.
    """

    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = True
    thread_name: str = "ckpt-staging"
    metric_prefix: str = "train.checkpoint_write"


class DefaultStager(CheckpointStager):
    """
    DefaultStager provides a full-featured staging implementation that combines
    multiple optimization techniques for efficient checkpoint preparation.

    The staging process works as follows:
    1. State dictionary is submitted for staging (sync or async)
    2. Tensors are copied from GPU to optimized CPU storage
    3. CUDA operations are synchronized if non-blocking copies are used
    4. Staged state dictionary is returned or made available via Future

    NOTE: state_dict should be deep-copyable object as staging will create a
    copy of it.

    Usage Patterns:
        # Synchronous staging
        stager = DefaultStager(CheckpointStagerConfig(use_async_staging=False))
        staged_dict = stager.stage(state_dict)
        stager.close()

        # Asynchronous staging
        stager = DefaultStager(CheckpointStagerConfig(use_async_staging=True))
        future = stager.stage(state_dict)
        # ... do other work ...
        staged_dict = future.result()
        stager.close()

        # Context manager pattern (recommended)
        with DefaultStager(config) as stager:
            result = stager.stage(state_dict)
            # Automatic cleanup on exit

    Performance Considerations:
        - Async staging provides best performance when model computation
          can overlap with staging operations
        - Pinned memory improves CPU-GPU transfer speeds but uses more memory
        - Shared memory allows efficient IPC to checkpoint process
        - Non-blocking copies reduce GPU idle time during memory transfers

    Thread Safety:
        DefaultStager is not thread-safe. Each thread should use its own
        instance, or external synchronization should be provided.
    """

    def __init__(
        self,
        config: CheckpointStagerConfig | None = None,
    ):
        event_logger = EventLogger()
        if config is None:
            config = CheckpointStagerConfig()
        self._config = config
        self._metric_prefix = config.metric_prefix
        self._state_dict_stager = StateDictStager(
            pin_memory=config.use_pinned_memory,
            share_memory=config.use_shared_memory,
            use_non_blocking_copy=config.use_non_blocking_copy,
        )
        self._staging_executor: ThreadPoolExecutor | None = None
        self._staging_stream: torch.Stream | None = None
        self._accelerator_available = torch.accelerator.is_available()
        # Capture the main thread's CUDA device so we can replay it on the
        # staging worker. PyTorch's `set_device` is thread-local — without
        # this, the worker's current device defaults to 0, and the
        # `cudaHostRegister` calls from pin_memory create a phantom primary
        # context on dev 0 for every rank (≈ +0.5 GiB of fixed CUDA
        # runtime overhead each, ~3.5 GiB total per host with 8 GPUs).
        self._cuda_device: int | None = (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        )

        if self._config.use_async_staging:
            self._staging_executor = ThreadPoolExecutor(
                max_workers=1, initializer=self._init_worker_thread
            )
            if self._accelerator_available:
                # Note: stream needs to be initialized on the main thread after default cuda
                # stream is setup/used to avoid the risk of accidentally reusing the main
                # compute stream or in other cases kernels actually launching from the
                # main thread.
                self._staging_stream = torch.Stream()

        logger.info(
            "Initialized staging",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.staging.initialize_stream.latency_ms",
                end_to_end=True,
            ),
        )

        if self._config.use_non_blocking_copy:
            assert self._accelerator_available, (
                "Non-blocking copy requires that the current accelerator is available."
            )

    def _init_worker_thread(self) -> None:
        """Make the worker's thread-local CUDA current device match the main
        thread's. Required so `cudaHostRegister` calls from pin_memory
        target the rank's own primary context instead of accidentally
        creating a new one on dev 0.
        """
        if self._cuda_device is not None:
            torch.cuda.set_device(self._cuda_device)

    def stage(
        self,
        state_dict: STATE_DICT,
    ) -> STATE_DICT | Future[STATE_DICT]:
        if self._accelerator_available:
            # If staging runs on a separate stream, we need to have that stream wait
            # for all operations on the current stream (e.g. optimiser step) to finish.
            # This event lets us know when we are ready.
            current_stream_ready = torch.Event(enable_timing=True)
            current_stream_ready.record()
        else:
            current_stream_ready = None

        if self._config.use_async_staging:
            assert self._staging_executor is not None, (
                "Staging executor should be initialized for async staging"
            )
            # Propagate the trainer thread's checkpoint_logging_context (step
            # etc.) to the staging worker via ctx.run, so EventLogger calls
            # inside _stage pick up the correct step via the contextvar
            # fallback. Matches the pattern in CheckpointProcess._submit_write.
            ctx = contextvars.copy_context()
            return self._staging_executor.submit(
                ctx.run,
                self._stage,
                state_dict,
                current_stream_ready,
            )
        else:
            return self._stage(state_dict, current_stream_ready)

    def _stage(
        self, state_dict: STATE_DICT, current_stream_ready: torch.Event | None
    ) -> STATE_DICT:
        event_logger = EventLogger()
        logger.info(
            "Initiating checkpoint staging",
            extra=event_logger(event_type=EventType.CHECKPOINT_STAGING_START),
        )
        set_thread_name_safe(self._config.thread_name)

        if self._accelerator_available:
            assert current_stream_ready is not None
            with self._staging_stream or nullcontext():
                # Block GPU until the main stream is done with all operations.
                # This does NOT block CPU.
                current_stream_ready.wait()
                state_dict = self._state_dict_stager.stage(state_dict)
                staging_stream_end = torch.Event(enable_timing=True)
                staging_stream_end.record()

            # Block CPU thread until all staging work has finished. This is necessary
            # for measuring latency, regardless of which stream we used.
            staging_stream_end.synchronize()
            staging_work_time_ms = current_stream_ready.elapsed_time(staging_stream_end)

        else:
            state_dict = self._state_dict_stager.stage(state_dict)
            # Everything is happening synchronously, so measure latency via
            # EventLogger's walltimer
            staging_work_time_ms = None

        logger.info(
            "End to end initialized staging",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.staging.e2e.latency_ms",
                end_to_end=True,
                value=staging_work_time_ms,
            ),
        )
        logger.info(
            "Staging copy done.",
            extra=event_logger(
                EventType.CHECKPOINT_STAGING_END,
                metric_name=f"{self._metric_prefix}.execute.subprocess_comm.wait_for_tensor_copy.latency_ms",
            ),
        )

        return state_dict

    def pinned_num_bytes(self) -> int:
        """Return the total pinned bytes held by the staging storage pool."""
        return self._state_dict_stager._storage_manager.pinned_num_bytes()

    def close(self) -> None:
        """
        Clean up all resources used by the DefaultStager. Shuts down the ThreadPoolExecutor
        used for async staging operations and cleans up the underlying StateDictStager's
        cached storages. Should be called when the stager is no longer needed to prevent
        resource leaks, especially in long-running applications. After calling close(),
        the stager should not be used for further staging operations.

        state_dict should be deep-copyable object.

        Example:
            stager = DefaultStager(CheckpointStagerConfig(use_async_staging=True))
            # ... do staging operations ...
            stager.close()  # Clean up all resources
        """
        if self._staging_executor:
            self._staging_executor.shutdown(wait=True)

        # Check if StateDictStager has a close method before calling it
        if hasattr(self._state_dict_stager, "close"):
            self._state_dict_stager.close()
