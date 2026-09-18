# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import gc
import os
import tempfile
import time
from concurrent.futures import Future
from typing import Any
from unittest import mock

import torch
from torch.multiprocessing.spawn import ProcessExitedException
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_checkpointing.barriers import DefaultStoreBarrierConfig
from torch_checkpointing.checkpoint_base import CheckpointItem, CheckpointWriteInfo
from torch_checkpointing.checkpoint_process import (
    CheckpointProcess,
    CheckpointProcessConfig,
    RequestType,
    WorkerRequest,
    WorkerResponse,
)
from torch_checkpointing.checkpoint_writer import (
    CheckpointWriterArgs,
    CheckpointWriterConfig,
)
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo


def subprocess_init_fn(name: str, parent_pid: int) -> None:
    """Initialize the subprocess with some basic checks."""
    # Initialize basic logging for the subprocess to aid debugging and visibility
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(processName)s %(process)d] %(message)s",
    )
    logging.info(f"Subprocess {name} starting with parent pid {parent_pid}")

    assert name == "test-checkpointer", f"Unexpected subprocess name: {name}"
    assert os.getpid() != parent_pid, "This was supposed to run in a different process"
    assert os.getppid() == parent_pid, (
        "This was supposed to run as a child to main process"
    )


def failing_subprocess_init_fn(name: str, parent_pid: int) -> None:
    """Initialize function that raises an exception."""
    # Acknowledge parameters to avoid unused variable warnings
    _ = name
    _ = parent_pid
    raise RuntimeError("Subprocess initialization failed")


def timedout_subprocess_init_fn(**kwargs: Any) -> None:
    # Acknowledge parameters to avoid unused variable warnings
    _ = kwargs
    time.sleep(3)  # Simulate a long initialization


def shared_memory_test_subprocess_init_fn(name: str, parent_pid: int) -> None:
    """Initialize subprocess and monkey-patch CheckpointWriter to modify shared tensors.

    This allows us to test that shared memory tensors are actually shared between processes
    by having the subprocess modify the tensor and verifying the changes in the main process.
    """
    # First do the normal init checks
    subprocess_init_fn(name, parent_pid)

    # Now monkey-patch CheckpointWriter.write to modify the shared tensor
    from torch_checkpointing.checkpoint_writer import CheckpointWriter

    original_write = CheckpointWriter.write

    def patched_write(self, path, checkpoint_info, **kwargs):
        # Access state_dict from checkpoint_info
        state_dict = checkpoint_info.state_dict

        # Before writing, modify the shared tensor to prove shared memory works
        if "shared_tensor" in state_dict:
            shared_tensor = state_dict["shared_tensor"]
            # Verify it's in shared memory in the subprocess
            assert shared_tensor.is_shared(), (
                "Shared tensor should be in shared memory in subprocess"
            )
            # Modify it - this should be visible in the main process if truly shared
            shared_tensor[0][0] = 42.0

        if "regular_tensor" in state_dict:
            # Note: ForkingPickler moves regular tensors to shared memory during IPC
            assert state_dict["regular_tensor"].is_shared(), (
                "Regular tensor should also be in shared memory in subprocess"
            )

        # Call original write (which may not actually save anything if path is empty)
        return original_write(self, path, checkpoint_info, **kwargs)

    CheckpointWriter.write = patched_write


class TestRequestTypes(TestCase):
    """Test the request/response data structures."""

    def test_request_type_enum(self) -> None:
        """Test RequestType enum values."""
        self.assertEqual(RequestType.PING.value, "ping")
        self.assertEqual(RequestType.WRITE_CHECKPOINT.value, "write_checkpoint")
        self.assertEqual(RequestType.TERMINATE_PROCESS.value, "exit")

    def test_worker_request(self) -> None:
        """Test WorkerRequest dataclass."""
        request = WorkerRequest(request_type=RequestType.PING, payload={"test": "data"})
        self.assertEqual(request.request_type, RequestType.PING)
        self.assertEqual(request.payload["test"], "data")

    def test_worker_response(self) -> None:
        """Test WorkerResponse dataclass."""
        response = WorkerResponse(
            request_type=RequestType.PING,
            success=True,
            error_msg=None,
            payload={"result": "success"},
        )
        self.assertEqual(response.request_type, RequestType.PING)
        self.assertTrue(response.success)
        self.assertIsNone(response.error_msg)
        if response.payload is not None:
            self.assertEqual(response.payload["result"], "success")


class TestCheckpointProcessConfig(TestCase):
    """Test CheckpointProcessConfig configuration."""

    def test_default_options(self) -> None:
        """Test default CheckpointProcessConfig."""
        options = CheckpointProcessConfig()
        # Test default values
        self.assertEqual(options.subprocess_init_timeout_secs, 30)
        self.assertEqual(options.subprocess_shutdown_timeout_secs, 60)

    def test_custom_options(self) -> None:
        """Test custom CheckpointProcessConfig."""
        options = CheckpointProcessConfig(
            subprocess_init_timeout_secs=10, subprocess_shutdown_timeout_secs=30
        )
        self.assertEqual(options.subprocess_init_timeout_secs, 10)
        self.assertEqual(options.subprocess_shutdown_timeout_secs, 30)


class TestCheckpointProcess(TestCase):
    def setUp(self) -> None:
        """Set up common test fixtures."""
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
            role_rank=0,
            role_world_size=1,
        )
        self.writer_config = CheckpointWriterConfig()
        self.test_state_dict = {
            "checkpoint": {
                "model": torch.nn.Linear(10, 5).state_dict(),
                "optimizer": {"param_groups": [{"lr": 0.01}]},
                "epoch": 5,
                "step": 1000,
            }
        }
        # Initialize the desired storage backend
        self.storage_config = LocalFileSystemStorageConfig()

    def _create_checkpoint_process(
        self,
        subprocess_init_fn_override=None,
        subprocess_init_args_override=None,
        checkpoint_writer_args_override=None,
        subprocess_init_timeout_secs=30,
    ):
        """Helper to create CheckpointProcess."""
        config = CheckpointProcessConfig(
            subprocess_init_timeout_secs=subprocess_init_timeout_secs,
        )

        # Create default checkpoint writer args if not provided
        if checkpoint_writer_args_override is None:
            checkpoint_writer_args_override = CheckpointWriterArgs(
                config=self.writer_config,
                rank_info=self.rank_info,
                storage_config=self.storage_config,
            )

        return CheckpointProcess(
            rank_info=self.rank_info,
            config=config,
            subprocess_init_fn=subprocess_init_fn_override or subprocess_init_fn,
            subprocess_init_args=subprocess_init_args_override
            or (
                "test-checkpointer",
                os.getpid(),
            ),
            checkpoint_writer_args=checkpoint_writer_args_override,
        )

    def _build_checkpoint_info(self, state_dict: dict) -> CheckpointWriteInfo:
        """Helper to build CheckpointWriteInfo from a state dict."""
        items = {
            key: CheckpointItem(value=value, layout=None)
            for key, value in state_dict.items()
        }
        return CheckpointWriteInfo(checkpoint_items=items)

    def _run_subprocess_write_for_telemetry(
        self,
        write_error: BaseException | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        checkpoint_info = self._build_checkpoint_info(self.test_state_dict)
        request = WorkerRequest(
            request_type=RequestType.WRITE_CHECKPOINT,
            payload={
                "path": "/checkpoint/step_1",
                "checkpoint_info": checkpoint_info,
            },
        )
        terminate = WorkerRequest(
            request_type=RequestType.TERMINATE_PROCESS,
            payload={},
        )
        parent_pipe = mock.Mock()
        parent_pipe.recv.side_effect = [request, terminate]
        writer = mock.Mock()
        writer_args = mock.Mock()
        writer_args.build.return_value = writer
        events: list[dict[str, Any]] = []
        order: list[str] = []

        def write(**_: Any) -> None:
            order.append("write")
            if write_error is not None:
                raise write_error

        writer.write.side_effect = write

        class RecordingEventLogger:
            def __init__(self, logger_id: int) -> None:
                self._logger_id = logger_id

            def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                event_type = str(args[0])
                events.append(
                    {
                        **kwargs,
                        "event_type": event_type,
                        "logger_id": self._logger_id,
                    }
                )
                if (
                    kwargs.get("metric_name")
                    == "custom.checkpoint_write.execute.subprocess_writer.latency_ms"
                ):
                    order.append("writer_metric")
                elif event_type == "checkpoint_write_start":
                    order.append("start")
                elif event_type == "checkpoint_write_end":
                    order.append("end")
                return {}

        loggers: list[RecordingEventLogger] = []

        def event_logger_factory() -> RecordingEventLogger:
            result = RecordingEventLogger(len(loggers))
            loggers.append(result)
            return result

        with (
            mock.patch(
                "torch_checkpointing.checkpoint_process.EventLogger",
                side_effect=event_logger_factory,
            ),
            mock.patch(
                "torch_checkpointing.checkpoint_process.time.perf_counter_ns",
                side_effect=[1_000_000_000, 1_250_000_000],
            ),
        ):
            CheckpointProcess._subprocess(
                0,
                self.rank_info,
                parent_pipe,
                lambda: None,
                (),
                writer_args,
                "ckpt",
                "custom.checkpoint_write",
            )
        return events, order

    def test_subprocess_preserves_lifecycle_and_emits_writer_timing_after_success(
        self,
    ) -> None:
        events, order = self._run_subprocess_write_for_telemetry()

        writer_events = [
            event
            for event in events
            if event.get("metric_name")
            == "custom.checkpoint_write.execute.subprocess_writer.latency_ms"
        ]
        lifecycle_events = [
            event
            for event in events
            if event["event_type"] in {"checkpoint_write_start", "checkpoint_write_end"}
        ]
        self.assertEqual(order, ["start", "write", "end", "writer_metric"])
        self.assertEqual(len(writer_events), 1)
        self.assertEqual(writer_events[0]["value"], 250.0)
        self.assertIn("checkpoint_path:/checkpoint/step_1", writer_events[0]["context"])
        self.assertEqual({event["logger_id"] for event in lifecycle_events}, {0})
        self.assertEqual(writer_events[0]["logger_id"], 1)

    def test_subprocess_does_not_emit_writer_timing_after_failure(self) -> None:
        events, order = self._run_subprocess_write_for_telemetry(
            RuntimeError("write failed")
        )

        self.assertEqual(order, ["start", "write"])
        self.assertFalse(
            any(
                event.get("metric_name")
                == "custom.checkpoint_write.execute.subprocess_writer.latency_ms"
                for event in events
            )
        )

    def test_checkpoint_process_keeps_the_barrier_scope_until_close(self) -> None:
        with mock.patch.object(
            DefaultStoreBarrierConfig, "use_in_subprocess"
        ) as mock_use:
            checkpoint_process = self._create_checkpoint_process(
                checkpoint_writer_args_override=CheckpointWriterArgs(
                    config=CheckpointWriterConfig(
                        barrier_config=DefaultStoreBarrierConfig()
                    ),
                    rank_info=self.rank_info,
                    storage_config=self.storage_config,
                )
            )
            try:
                checkpoint_process.wait_for_init()
                self.assertEqual(mock_use.call_count, 1)
                self.assertEqual(mock_use.return_value.__enter__.call_count, 1)
                self.assertEqual(mock_use.return_value.__exit__.call_count, 0)
            finally:
                checkpoint_process.close()
            self.assertEqual(mock_use.return_value.__exit__.call_count, 1)

    def test_checkpoint_process_initialization(self) -> None:
        """Test that CheckpointProcess initializes and closes correctly."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for the process creation future to complete
        checkpoint_process.wait_for_init()

        # Verify process is alive
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())

        checkpoint_process.close()

        # Verify process is terminated
        self.assertFalse(checkpoint_process.process.processes[0].is_alive())

    def test_checkpoint_write_sync_state_dict(self) -> None:
        """Test writing a checkpoint with synchronous state dict."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process._creation_future.result()

        # Create a temporary directory for the checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Build CheckpointInfo from state dict
            checkpoint_info = self._build_checkpoint_info(self.test_state_dict)

            # Write checkpoint
            future = checkpoint_process.write(checkpoint_info, checkpoint_path)

            # Verify future is returned
            self.assertIsInstance(future, Future)

            # Wait for completion
            future.result()

            # Verify checkpoint file was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

            # Verify checkpoint content
            loaded_state_dict = torch.load(expected_file)
            self.assertIn("model", loaded_state_dict)
            self.assertIn("optimizer", loaded_state_dict)
            self.assertEqual(loaded_state_dict["epoch"], 5)
            self.assertEqual(loaded_state_dict["step"], 1000)

        checkpoint_process.close()

    def test_checkpoint_write_future_state_dict(self) -> None:
        """Test writing a checkpoint with Future CheckpointInfo."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process._creation_future.result()

        # Create a Future that resolves to CheckpointInfo
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)

        def get_checkpoint_info():
            time.sleep(0.1)  # Simulate some processing time
            return self._build_checkpoint_info(self.test_state_dict)

        future_checkpoint_info = executor.submit(get_checkpoint_info)

        # Create a temporary directory for the checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write checkpoint with Future CheckpointInfo
            write_future = checkpoint_process.write(
                future_checkpoint_info, checkpoint_path
            )

            # Wait for completion
            write_future.result()

            # Verify checkpoint file was created
            expected_file = os.path.join(
                checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(expected_file))

        executor.shutdown(wait=True)
        checkpoint_process.close()

    def test_subprocess_initialization_timeout(self) -> None:
        """Test subprocess initialization timeout."""

        # Create checkpoint process with a very short timeout by mocking the initialization
        checkpoint_process = self._create_checkpoint_process(
            subprocess_init_fn_override=timedout_subprocess_init_fn,
            subprocess_init_timeout_secs=1,
        )

        # This should timeout
        with self.assertRaises(TimeoutError) as cm:
            checkpoint_process._creation_future.result()

        self.assertIn("Timed out", str(cm.exception))

    def test_subprocess_initialization_failure(self) -> None:
        """Test subprocess initialization failure."""
        checkpoint_process = self._create_checkpoint_process(
            subprocess_init_fn_override=failing_subprocess_init_fn
        )

        # The subprocess should fail to initialize
        # We expect this to raise an exception when we try to use it
        with self.assertRaises(RuntimeError):
            checkpoint_process._creation_future.result()

    def test_graceful_termination(self) -> None:
        """Test graceful termination of subprocess."""
        checkpoint_process = self._create_checkpoint_process()

        checkpoint_process._creation_future.result()
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())
        checkpoint_process.close()
        self.assertFalse(checkpoint_process.process.processes[0].is_alive())

    def test_forced_termination(self) -> None:
        """Test forced termination when graceful termination fails."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process._creation_future.result()

        # Mock the join method to simulate timeout
        def mock_join(timeout=None):
            # Acknowledge timeout parameter to avoid unused variable warning
            _ = timeout
            return False  # Simulate timeout

        checkpoint_process.process.join = mock_join

        # This should trigger forced termination
        checkpoint_process.close()

        # Process should still be terminated (killed)
        # Note: This test might be flaky depending on timing

    def test_communication_error_handling(self):
        """Test handling of communication errors."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process._creation_future.result()

        # Close the pipe to simulate communication failure
        checkpoint_process._parent_end.close()

        # Build CheckpointInfo from state dict
        checkpoint_info = self._build_checkpoint_info(self.test_state_dict)

        # Attempting to write should raise an error
        future = checkpoint_process.write(checkpoint_info, "/tmp/test")
        with self.assertRaises(RuntimeError) as cm:
            future.result()

        self.assertIn("Child process terminated unexpectedly", str(cm.exception))

    def test_shared_memory_tensor_ipc(self):
        """Test that shared memory tensors are backed by the same memory across processes."""

        # Use custom subprocess init that monkey-patches the writer to modify shared tensors
        checkpoint_process = self._create_checkpoint_process(
            subprocess_init_fn_override=shared_memory_test_subprocess_init_fn,
        )

        checkpoint_process._creation_future.result()

        # Create tensors and put them in shared memory
        shared_tensor = torch.randn(100, 100)
        shared_tensor.share_memory_()

        shared_tensor_data_ptr = shared_tensor.data_ptr()

        regular_tensor = torch.randn(50, 50)
        # Don't put regular tensor in shared memory for comparison

        # Verify initial shared memory status
        self.assertTrue(
            shared_tensor.is_shared(), "Shared tensor should be in shared memory"
        )
        self.assertFalse(
            regular_tensor.is_shared(), "Regular tensor should not be in shared memory"
        )

        # Create state dict with mixed tensor types
        test_state_dict = {
            "shared_tensor": shared_tensor,
            "regular_tensor": regular_tensor,
        }

        # Build CheckpointInfo from state dict
        checkpoint_info = self._build_checkpoint_info(test_state_dict)

        # Create a temporary directory for the checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint")

            # Write to subprocess - the SharedTensorVerifier will:
            # 1. Verify the tensor is still in shared memory
            # 2. Check the marker value (42.0) to confirm same memory
            # 3. Modify specific positions to prove same memory access
            future = checkpoint_process.write(checkpoint_info, checkpoint_path)

            try:
                result = (
                    future.result()
                )  # This will raise an exception if the subprocess assertions fail
                self.assertIsNone(
                    result
                )  # SharedTensorVerifier returns None on success
            except Exception as e:
                self.fail(f"Subprocess assertions failed: {e}")

        # assert shared tensor is still in same shared memory
        self.assertEqual(
            shared_tensor_data_ptr,
            shared_tensor.data_ptr(),
            "Shared tensor should still be in same shared memory",
        )
        self.assertTrue(
            shared_tensor.is_shared(), "Shared tensor should still be in shared memory"
        )

        # CRITICAL TEST: Verify that modifications made by subprocess are visible in main process
        # This definitively proves that both processes access the same memory
        self.assertAlmostEqual(
            shared_tensor[0][0].item(),  # Convert tensor to float for comparison
            42.0,
            places=6,
            msg=f"Expected subprocess signature 42.0, got {shared_tensor[0][0].item()}. "
            f"Shared memory not working - subprocess modifications not visible!",
        )

        checkpoint_process.close()

    def test_close_method_with_process_exception(self) -> None:
        """Test close method behavior when process exits due to an exception."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process._creation_future.result()

        # Verify process is alive initially
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())

        # Simulate subprocess termination due to exception by killing the process
        # This mimics what happens when the subprocess crashes or exits due to an exception
        checkpoint_process.process.processes[0].kill()

        try:
            checkpoint_process.close()
        except ProcessExitedException:
            self.fail(
                "close() method should not raise ProcessExitedException when process has already exited"
            )

        # Verify the process is still not alive after close
        self.assertFalse(checkpoint_process.process.processes[0].is_alive())

    def test_subprocess_handles_parent_pipe_closed(self) -> None:
        """Test that subprocess exits gracefully when parent pipe is closed (EOFError)."""
        checkpoint_process = self._create_checkpoint_process()

        # Wait for initialization
        checkpoint_process._creation_future.result()

        # Verify process is alive initially
        self.assertTrue(checkpoint_process.process.processes[0].is_alive())

        # Close the pipe without sending termination, simulating parent dying
        checkpoint_process._parent_end.close()

        # Assert that the subprocess exited gracefully.
        # join() returns True if all processes joined successfully, False on timeout
        self.assertTrue(checkpoint_process.process.join(timeout=10))
        self.assertEqual(checkpoint_process.process.processes[0].exitcode, 0)

    def test_serving_a_request_releases_its_shared_memory(self) -> None:
        """The subprocess must not hold onto references to the payload after it is finished
        writing and waiting for the next request. Shared tensors must get freed if the
        parent process drops them.
        """
        payload_mib = 256
        n_tensors = 8
        rank_info = RankInfo(
            global_world_size=1, global_rank=0, role_rank=0, role_world_size=1
        )
        checkpoint_process = CheckpointProcess(
            rank_info=rank_info,
            config=CheckpointProcessConfig(subprocess_init_timeout_secs=60),
            subprocess_init_fn=subprocess_init_fn,
            subprocess_init_args=("test-checkpointer", os.getpid()),
            checkpoint_writer_args=CheckpointWriterArgs(
                config=CheckpointWriterConfig(),
                rank_info=rank_info,
                storage_config=LocalFileSystemStorageConfig(),
            ),
        )
        try:
            checkpoint_process._creation_future.result()
            child_pid = checkpoint_process.process.processes[0].pid
            try:
                baseline = _rss_mib(child_pid)
            except Exception as e:
                self.skipTest(
                    f"Could not measure RSS of the child process; it currently only works on Linux. {e!r}"
                )

            tensors = [
                torch.zeros(
                    payload_mib * 1024**2 // n_tensors // 4, dtype=torch.float32
                ).share_memory_()
                for _ in range(n_tensors)
            ]

            info = CheckpointWriteInfo(
                checkpoint_items={
                    f"t{i}": CheckpointItem(value=tensor, layout=None)
                    for i, tensor in enumerate(tensors)
                }
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                write_fut = checkpoint_process.write(
                    checkpoint_info=info, path=os.path.join(tmpdir, "ckpt")
                )

                mem_hi = _rss_mib(child_pid) - baseline
                del info
                del tensors
                gc.collect()

                # We expect the child process RSS to increase by about payload_mib when
                # the shared tensors get mapped and then drop back down to baseline when
                # they are released. Leave some wiggle room.
                upper_threshold = payload_mib * 0.75
                lower_threshold = payload_mib * 0.25
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    retained = _rss_mib(child_pid) - baseline
                    mem_hi = max(mem_hi, retained)
                    if mem_hi > upper_threshold and retained < lower_threshold:
                        break
                    time.sleep(0.05)

                assert mem_hi > upper_threshold, (
                    f"Expected to see a memory spike of {payload_mib} MiB but only saw {mem_hi} MiB"
                )
                assert retained < lower_threshold, (
                    f"Expected memory to be freed, but still holding onto {retained} MiB"
                )
                write_fut.result()  # Assert no throw
        finally:
            checkpoint_process.close()


def _rss_mib(pid: int) -> float:
    with open(f"/proc/{pid}/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError(f"no VmRSS for pid {pid}")


if __name__ == "__main__":
    run_tests()
