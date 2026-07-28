# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase
from torch_checkpointing import _pin_memory_utils as pin_memory_utils
from torch_checkpointing._state_dict_stager import StorageManager
from torch_checkpointing.staging import (
    CheckpointStagerConfig,
    DefaultStager,
)
from torch_checkpointing.utils import ensure_future


class TestDefaultStager(TestCase):
    def setUp(self) -> None:
        # Create a test state dictionary with various data types
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
            "tensor": torch.randn(3, 4),
            "nested": {"inner_tensor": torch.ones(2, 2), "inner_value": 42},
        }

    @requires_cuda
    @pytest.mark.gpus_needed_1
    def test_sync_staging(self) -> None:
        """Test synchronous staging."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        # Stage the state dict
        staged_dict = stager.stage(self.state_dict)

        # Verify that a state dict is returned (not a Future)
        self.assertIsInstance(staged_dict, dict)

        # Verify the staged state dictionary
        assert isinstance(staged_dict, dict)
        self.assertIn("model", staged_dict)
        self.assertIn("optimizer", staged_dict)
        self.assertEqual(staged_dict["epoch"], 5)
        self.assertEqual(staged_dict["step"], 1000)
        self.assertIn("tensor", staged_dict)
        self.assertIn("nested", staged_dict)

        # Clean up
        stager.close()

    @requires_cuda
    @pytest.mark.gpus_needed_1
    def test_async_staging(self) -> None:
        """Test asynchronous staging."""
        options = CheckpointStagerConfig(use_async_staging=True)
        stager = DefaultStager(options)

        # Stage the state dict
        result = stager.stage(self.state_dict)

        # Verify that a Future is returned
        self.assertIsInstance(result, Future)

        # Wait for the Future to complete
        if isinstance(result, Future):
            staged_dict = result.result()

            # Verify the staged state dictionary
            self.assertIsInstance(staged_dict, dict)
            self.assertIn("model", staged_dict)
            self.assertIn("optimizer", staged_dict)
            self.assertEqual(staged_dict["epoch"], 5)
            self.assertEqual(staged_dict["step"], 1000)
        else:
            self.fail("Expected Future but got dict")

        # Clean up
        stager.close()

    def test_stage_passes_through_keys_not_requiring_copy(self) -> None:
        """stage(state_dict, keys_not_requiring_copy=[...]) stages the other
        keys (distinct CPU copies) while passing the listed keys through
        unaffected (same object); the full key set is always returned.
        """
        for use_async_staging in (False, True):
            with self.subTest(use_async_staging=use_async_staging):
                copied = torch.randn(4, 4)
                not_copied = torch.randn(4, 4)
                state_dict = {"copied": copied, "not_copied": not_copied}

                stager = DefaultStager(
                    CheckpointStagerConfig(
                        use_async_staging=use_async_staging,
                        use_pinned_memory=False,
                        use_shared_memory=False,
                        use_non_blocking_copy=False,
                    )
                )
                try:
                    result = ensure_future(
                        stager.stage(state_dict, keys_not_requiring_copy=["not_copied"])
                    ).result()
                    self.assertIsInstance(result, dict)
                    self.assertEqual(set(result.keys()), {"copied", "not_copied"})

                    # Copy-required key: distinct storage, equal values.
                    self.assertIsNot(result["copied"], copied)
                    self.assertNotEqual(result["copied"].data_ptr(), copied.data_ptr())
                    torch.testing.assert_close(result["copied"], copied)

                    # Passed-through key: same object, unaffected.
                    self.assertIs(result["not_copied"], not_copied)
                finally:
                    stager.close()

    def test_get_staged_state_dict_none_before_staging(self) -> None:
        """get_staged_state_dict() returns None before any stage() call."""
        for use_async_staging in (False, True):
            with self.subTest(use_async_staging=use_async_staging):
                stager = DefaultStager(
                    CheckpointStagerConfig(
                        use_async_staging=use_async_staging,
                        use_pinned_memory=False,
                        use_shared_memory=False,
                        use_non_blocking_copy=False,
                    )
                )
                try:
                    self.assertIsNone(stager.get_staged_state_dict())
                finally:
                    stager.close()

    def test_get_staged_state_dict_returns_latest_staged(self) -> None:
        """get_staged_state_dict() returns the exact dict produced by the most
        recent successful stage(); a second stage() supersedes the first.
        """
        for use_async_staging in (False, True):
            with self.subTest(use_async_staging=use_async_staging):
                stager = DefaultStager(
                    CheckpointStagerConfig(
                        use_async_staging=use_async_staging,
                        use_pinned_memory=False,
                        use_shared_memory=False,
                        use_non_blocking_copy=False,
                    )
                )
                try:
                    first = ensure_future(
                        stager.stage({"tensor": torch.randn(4, 4), "step": 1})
                    ).result()
                    self.assertIs(stager.get_staged_state_dict(), first)

                    second = ensure_future(
                        stager.stage({"tensor": torch.randn(4, 4), "step": 2})
                    ).result()
                    self.assertIs(stager.get_staged_state_dict(), second)
                    self.assertIsNot(first, second)
                finally:
                    stager.close()

    def test_get_staged_state_dict_none_after_failure(self) -> None:
        """A failed stage() resets the cached dict to None, so a stale dict from
        an earlier successful stage() is never returned.
        """
        for use_async_staging in (False, True):
            with self.subTest(use_async_staging=use_async_staging):
                stager = DefaultStager(
                    CheckpointStagerConfig(
                        use_async_staging=use_async_staging,
                        use_pinned_memory=False,
                        use_shared_memory=False,
                        use_non_blocking_copy=False,
                    )
                )
                try:
                    # A successful stage populates the cached dict.
                    ensure_future(stager.stage({"tensor": torch.randn(4, 4)})).result()
                    self.assertIsNotNone(stager.get_staged_state_dict())

                    # A stage that raises must clear it back to None.
                    with patch.object(
                        stager._state_dict_stager,
                        "stage",
                        side_effect=RuntimeError("boom"),
                    ):
                        with self.assertRaises(RuntimeError):
                            ensure_future(
                                stager.stage({"tensor": torch.randn(4, 4)})
                            ).result()

                    self.assertIsNone(stager.get_staged_state_dict())
                finally:
                    stager.close()

    def test_cuda_non_blocking_without_cuda(self) -> None:
        """Test that non-blocking copy fails when CUDA is not available."""
        if torch.cuda.is_available():
            self.skipTest("CUDA is available, cannot test CUDA unavailable scenario")

        options = CheckpointStagerConfig(use_non_blocking_copy=True)
        with self.assertRaises(AssertionError):
            DefaultStager(options)

    def test_different_option_combinations(self) -> None:
        """Test various combinations of staging options."""
        test_cases = [
            # All disabled
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=False,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
            # Only pinned memory
            CheckpointStagerConfig(
                use_pinned_memory=True,
                use_shared_memory=False,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
            # Only shared memory
            CheckpointStagerConfig(
                use_pinned_memory=False,
                use_shared_memory=True,
                use_async_staging=False,
                use_non_blocking_copy=False,
            ),
        ]

        if torch.cuda.is_available():
            # Only async staging
            test_cases.append(
                CheckpointStagerConfig(
                    use_pinned_memory=torch.accelerator.is_available(),
                    use_shared_memory=False,
                    use_async_staging=True,
                    use_non_blocking_copy=False,
                )
            )
            # Only CUDA non-blocking copy
            test_cases.append(
                CheckpointStagerConfig(
                    use_pinned_memory=torch.accelerator.is_available(),
                    use_shared_memory=False,
                    use_async_staging=False,
                    use_non_blocking_copy=torch.accelerator.is_available(),
                )
            )

        for options in test_cases:
            with self.subTest(options=options):
                stager = DefaultStager(options)

                # Test staging works with these options
                if options.use_async_staging and torch.accelerator.is_available():
                    result = stager.stage(self.state_dict)
                    self.assertIsInstance(result, Future)
                    if isinstance(result, Future):
                        staged_dict = result.result()

                        self.assertIsNotNone(staged_dict)
                        self.assertIsInstance(staged_dict, dict)
                        self.assertIn("model", staged_dict)
                    else:
                        self.fail("Expected Future but got dict")
                else:
                    result = stager.stage(self.state_dict)
                    self.assertIsInstance(result, dict)
                    if isinstance(result, dict):
                        self.assertIn("model", result)
                    else:
                        self.fail("Expected dict but got Future")

                stager.close()

    @requires_cuda
    @pytest.mark.gpus_needed_1
    def test_cuda_tensors_staging(self) -> None:
        """Test staging with CUDA tensors."""
        # Create state dict with CUDA tensors
        cuda_state_dict = {
            "cuda_tensor": torch.randn(3, 4).cuda(),
            "cpu_tensor": torch.randn(2, 3),
            "mixed_model": {
                "weight": torch.randn(5, 5).cuda(),
                "bias": torch.randn(5).cuda(),
            },
        }

        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        staged_dict = stager.stage(cuda_state_dict)
        assert isinstance(staged_dict, dict)

        # Verify tensors are staged (should be moved to CPU)
        self.assertIn("cuda_tensor", staged_dict)
        self.assertIn("cpu_tensor", staged_dict)
        self.assertIn("mixed_model", staged_dict)

        stager.close()

    @requires_cuda
    @pytest.mark.gpus_needed_1
    def test_resource_cleanup(self) -> None:
        """Test that resources are properly cleaned up."""
        options = CheckpointStagerConfig(use_async_staging=False)
        stager = DefaultStager(options)

        # Verify initial state
        self.assertIsNotNone(stager._state_dict_stager)

        # Close and verify cleanup
        stager.close()

    def test_multiple_staging_operations(self) -> None:
        """Test multiple staging operations with the same stager."""
        options = CheckpointStagerConfig(
            use_async_staging=False,
            use_pinned_memory=torch.accelerator.is_available(),
            use_shared_memory=False,
            use_non_blocking_copy=torch.accelerator.is_available(),
        )
        stager = DefaultStager(options)

        # Stage multiple different state dicts
        state_dicts = [
            {"model1": torch.nn.Linear(5, 3).state_dict()},
            {"model2": torch.nn.Conv2d(3, 16, 3).state_dict()},
            {"optimizer": {"lr": 0.001, "momentum": 0.9}},
        ]

        staged_results = []
        for state_dict in state_dicts:
            staged_dict = stager.stage(state_dict)
            staged_results.append(staged_dict)

        # Verify all staging operations succeeded
        self.assertEqual(len(staged_results), 3)
        for i, result in enumerate(staged_results):
            self.assertIsInstance(result, dict)
            assert isinstance(result, dict)
            # Verify the result contains the expected keys
            for key in state_dicts[i].keys():
                self.assertIn(key, result)

        stager.close()

    @requires_cuda
    @pytest.mark.gpus_needed_1
    def test_cached_storage_reuse_with_new_tensor(self) -> None:
        """Test that cached storage is reused for new tensors with same FQN and shape.

        This test verifies that:
        1. On the first staging, a new storage is created and pinned
        2. On subsequent stagings with newly allocated tensors (same FQN/shape),
           the cached storage is reused and NO new pinning occurs
        """
        options = CheckpointStagerConfig(
            use_async_staging=False,
            use_pinned_memory=True,
            use_shared_memory=True,
            use_non_blocking_copy=False,
        )
        stager = DefaultStager(options)

        with (
            patch(
                "torch_checkpointing._pin_memory_utils.pin_memory"
            ) as mock_pin_memory,
            patch("torch_checkpointing._pin_memory_utils.unpin_memory"),
        ):
            num_staging_attempts = 3
            staged_dicts = []

            for i in range(num_staging_attempts):
                # Reset mock to track only this iteration's calls
                mock_pin_memory.reset_mock()

                # Allocate NEW tensors with same FQN and shape for each iteration
                state_dict = {
                    "model": {
                        "weight": torch.randn(100, 100).cuda(),
                        "bias": torch.randn(100).cuda(),
                    }
                }

                # Stage the state dict
                staged_dict = stager.stage(state_dict)
                assert isinstance(staged_dict, dict)
                staged_dicts.append(staged_dict)

                # Check pinning behavior for this iteration only
                pin_calls_this_iteration = mock_pin_memory.call_count

                if i == 0:
                    # First staging: pinning should occur for exactly 2 tensors (weight and bias)
                    self.assertEqual(
                        pin_calls_this_iteration,
                        2,
                        f"Expected exactly 2 pin_memory calls on first staging (iteration {i})",
                    )
                else:
                    # Subsequent stagings: NO new pinning should occur (cached storage reused)
                    # Verify new tensors were actually allocated (different data_ptr from previous)
                    self.assertNotEqual(
                        staged_dicts[i - 1]["model"]["weight"]
                        .untyped_storage()
                        .data_ptr(),
                        state_dict["model"]["weight"].untyped_storage().data_ptr(),
                        f"Iteration {i}: Original tensors should have different data pointers",
                    )

                    # No new pinning should have happened (cached storage reused)
                    self.assertEqual(
                        pin_calls_this_iteration,
                        0,
                        f"Expected 0 pin_memory calls on staging iteration {i} (cached storage should be reused)",
                    )

                # Verify all staged dicts share the same cached storage
                if i > 0:
                    self.assertEqual(
                        staged_dicts[0]["model"]["weight"].data_ptr(),
                        staged_dict["model"]["weight"].data_ptr(),
                        f"Iteration {i}: Staged weight tensors should share the same cached storage",
                    )
                    self.assertEqual(
                        staged_dicts[0]["model"]["bias"].data_ptr(),
                        staged_dict["model"]["bias"].data_ptr(),
                        f"Iteration {i}: Staged bias tensors should share the same cached storage",
                    )

        stager.close()

    @requires_cuda
    @pytest.mark.gpus_needed_1
    def test_non_blocking_copy_race_condition(self) -> None:
        """Test that non-blocking correctly captures state of tensors that are being
        updated on the default stream. Necessary because use_non_blocking_copy=True
        runs staging on an alternate stream and we need to manage synchronisation.
        """
        options = CheckpointStagerConfig(
            use_async_staging=True,
            use_pinned_memory=True,
            use_shared_memory=True,
            use_non_blocking_copy=True,
        )
        stager = DefaultStager(options)
        # Force eager initialization of the thread pool
        ensure_future(stager.stage({})).result()

        # Do some expensive computation on the default stream
        size = 4096
        a = torch.randn(size, size).cuda()
        b = torch.randn(size, size).cuda()
        for _ in range(100):
            a = (b @ a).clamp_(-1, 1)

        fut = stager.stage({"model": {"tensor": a}})

        staged_tensor = ensure_future(fut).result()["model"]["tensor"]
        self.assertEqual(staged_tensor.device, torch.device("cpu"))

        # staged_tensor should reflect the tensor's final value.  If we see an
        # intermediate value then we have hit the race condition.
        torch.testing.assert_close(staged_tensor, a.cpu())
        stager.close()

    @requires_cuda
    @pytest.mark.gpus_needed_1
    @patch.object(pin_memory_utils, "pin_memory", wraps=pin_memory_utils.pin_memory)
    @patch.object(pin_memory_utils, "unpin_memory", wraps=pin_memory_utils.unpin_memory)
    def test_storages_unpinned_at_close(self, mock_unpin, mock_pin) -> None:
        """Test that storages are unpinned when the stager is closed."""
        # Real pin and unpin, wrapping in mock so we can check the calls
        options = CheckpointStagerConfig(
            use_async_staging=True,
            use_pinned_memory=True,
            use_shared_memory=True,
            use_non_blocking_copy=True,
        )
        stager = DefaultStager(options)
        for _ in range(3):
            state_dict = {
                "model": {
                    "a": torch.randn(10).cuda(),
                    "b": torch.randn(10).cuda(),
                    "c": torch.randn(20).cuda(),
                    "d": torch.randn(30).cuda(),
                }
            }
            fut = stager.stage(state_dict)
            staged_state_dict = ensure_future(fut).result()
            for k, v in staged_state_dict["model"].items():
                torch.equal(v, state_dict["model"][k].cpu())

            assert mock_pin.call_count == 4  # Always 4, does not increase
            assert mock_unpin.call_count == 0

        stager.close()
        assert mock_unpin.call_count == 4
        # unpin_memory should be called on the same storage pointers as pin_memory
        pinned_ptrs = {call.args[0] for call in mock_pin.call_args_list}
        unpinned_ptrs = {call.args[0] for call in mock_unpin.call_args_list}
        assert pinned_ptrs == unpinned_ptrs
        assert stager.pinned_num_bytes() == 0

        # Repeated cleanup must not unpin or release the same storage twice.
        stager.close()
        assert mock_unpin.call_count == 4
        assert stager.pinned_num_bytes() == 0
        assert {call.args[0] for call in mock_unpin.call_args_list} == unpinned_ptrs


@requires_cuda
@pytest.mark.gpus_needed_1
def test_default_stager_copies_run_on_side_stream(tmp_path: Path) -> None:
    stager = DefaultStager(
        CheckpointStagerConfig(
            use_async_staging=True,
            use_pinned_memory=True,
            use_shared_memory=True,
            use_non_blocking_copy=True,
        )
    )
    try:
        size = 2048
        left = torch.randn(size, size, device="cuda")
        right = torch.randn(size, size, device="cuda")
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            result = left @ right
            staged = ensure_future(stager.stage({"result": result})).result()
            torch.cuda.synchronize()

        torch.testing.assert_close(staged["result"], result.cpu())

        trace_path = tmp_path / "staging_trace.json"
        profiler.export_chrome_trace(str(trace_path))
        trace_events: list[dict[str, Any]] = json.loads(trace_path.read_text())[
            "traceEvents"
        ]

        def event_stream(event: dict[str, Any]) -> int | None:
            stream = event.get("args", {}).get("stream")
            return stream if isinstance(stream, int) else None

        dtoh_streams = {
            stream
            for event in trace_events
            if "dtoh" in str(event.get("name", "")).lower()
            and (stream := event_stream(event)) is not None
        }
        compute_streams = {
            stream
            for event in trace_events
            if str(event.get("cat", "")).lower() == "kernel"
            and (stream := event_stream(event)) is not None
        }
        assert dtoh_streams, "Expected a profiled device-to-host staging copy"
        assert compute_streams, "Expected a profiled CUDA compute kernel"
        assert len(dtoh_streams) == 1
        assert dtoh_streams.isdisjoint(compute_streams)
    finally:
        stager.close()


class TestStorageManager(TestCase):
    def test_reuse(self):
        storages_seen = []
        manager = StorageManager(
            pin_memory=False, share_memory=True, pin_memory_min_bytes=0
        )

        # These should all be new storages
        for size in [5, 10, 20, 10, 7, 10]:
            storage = manager.get(size)
            assert storage.nbytes() == size
            assert storage.is_shared()
            assert storage not in storages_seen
            storages_seen.append(storage)

        manager.reset()

        # These should be reused
        for size in [20, 10, 5, 10]:
            storage = manager.get(size)
            assert storage.nbytes() == size
            assert storage.is_shared()
            assert storage in storages_seen
        # These should be new
        for size in [20, 5, 12]:
            storage = manager.get(size)
            assert storage.nbytes() == size
            assert storage.is_shared()
            assert storage not in storages_seen
            storages_seen.append(storage)

        manager.delete_unused()
        manager.reset()

        # I didn't use the 7 last round, so it should have gotten cleaned up
        for size in [7]:
            storage = manager.get(size)
            assert storage.nbytes() == size
            assert storage.is_shared()
            assert storage not in storages_seen

    @requires_cuda
    @pytest.mark.gpus_needed_1
    def test_pin_unpin(self):
        manager = StorageManager(
            pin_memory=True, share_memory=True, pin_memory_min_bytes=5
        )
        storage = manager.get(10)
        assert storage.is_pinned()
        assert storage.is_shared()
        manager.reset()
        storage2 = manager.get(10)
        # Should be the same object as before
        assert storage2 is storage
        assert storage2.is_pinned()
        assert storage2.is_shared()

        # Too small to pin
        storage3 = manager.get(3)
        assert not storage3.is_pinned()
        assert storage3.is_shared()

        # Pinned storage should get unpinned
        manager.close()
        assert not storage.is_pinned()
        assert not storage2.is_pinned()
        assert not storage3.is_pinned()

    def test_num_bytes_pin_disabled(self):
        """pinned_num_bytes always returns 0 when pin_memory is False."""
        manager = StorageManager(
            pin_memory=False, share_memory=False, pin_memory_min_bytes=0
        )
        manager.get(100)
        manager.get(200)
        assert manager.total_num_bytes() == 300
        assert manager.pinned_num_bytes() == 0
        manager.close()

    def test_num_bytes_pin_enabled(self):
        """pinned_num_bytes counts only storages whose size meets the pinning threshold."""
        with (
            patch("torch_checkpointing._pin_memory_utils.pin_memory"),
            patch("torch_checkpointing._pin_memory_utils.unpin_memory"),
        ):
            manager = StorageManager(
                pin_memory=True, share_memory=False, pin_memory_min_bytes=10
            )
            # below threshold: not counted as pinned
            manager.get(5)
            assert manager.total_num_bytes() == 5
            assert manager.pinned_num_bytes() == 0

            # at threshold: counted
            manager.get(10)
            assert manager.total_num_bytes() == 15
            assert manager.pinned_num_bytes() == 10

            # above threshold: counted
            manager.get(100)
            assert manager.total_num_bytes() == 115
            assert manager.pinned_num_bytes() == 110

            # multiple storages of the same size scale linearly
            manager.get(100)
            assert manager.total_num_bytes() == 215
            assert manager.pinned_num_bytes() == 210

            # Reset does not clear memory
            manager.reset()
            assert manager.total_num_bytes() == 215
            assert manager.pinned_num_bytes() == 210

            # delete_unused clears memory
            manager.delete_unused()
            assert manager.total_num_bytes() == 0
            assert manager.pinned_num_bytes() == 0

            manager.close()


if __name__ == "__main__":
    run_tests()
