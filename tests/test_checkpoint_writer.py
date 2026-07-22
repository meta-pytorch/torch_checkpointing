# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from unittest import mock

import torch
import torch_checkpointing.checkpoint_writer as checkpoint_writer_module
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_checkpointing.checkpoint_base import (
    CheckpointItem,
    CheckpointWriteInfo,
)
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    RawSerialization,
    TorchSerialization,
)
from torch_checkpointing.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterArgs,
    CheckpointWriterConfig,
)
from torch_checkpointing.logging_utils import EventLogger
from torch_checkpointing.storage.filesystem import (
    LocalFileSystemStorage,
    LocalFileSystemStorageConfig,
)
from torch_checkpointing.types import RankInfo


def simple_layout(rank: int) -> dict[str, LayoutInfo]:
    """Simple test layout that splits model and metadata."""
    return {
        "model": LayoutInfo(
            file_path=f"model_rank_{rank}.pt",
            serialization_format=TorchSerialization(),
        ),
        "optimizer": LayoutInfo(
            file_path=f"optimizer_rank_{rank}.pt",
            serialization_format=TorchSerialization(),
        ),
        "epoch": LayoutInfo(
            file_path=f"epoch_rank_{rank}.json",
            serialization_format=JsonSerialization(str),
        ),
        "step": LayoutInfo(
            file_path=f"step_rank_{rank}.json",
            serialization_format=JsonSerialization(str),
        ),
    }


def global_file_layout(rank: int) -> dict[str, LayoutInfo]:
    """Test layout that demonstrates global file functionality."""
    return {
        "model": LayoutInfo(
            file_path=f"model_rank_{rank}.pt",  # Per-rank file
            serialization_format=TorchSerialization(),
        ),
        "global_config": LayoutInfo(
            file_path="config.json",  # Global file - same for all ranks
            serialization_format=JsonSerialization(str),
        ),
        "optimizer": LayoutInfo(
            file_path=f"optimizer_rank_{rank}.pt",  # Per-rank file
            serialization_format=TorchSerialization(),
        ),
    }


class MockCallback:
    """Mock implementation of callback functions for testing."""

    def __init__(self) -> None:
        self.pre_finalize_called: bool = False
        self.finalize_called: bool = False
        self.pre_finalize_path: str | None = None
        self.finalize_path: str | None = None
        self.pre_finalize_event_logger: EventLogger | None = None
        self.finalize_event_logger: EventLogger | None = None

    def pre_finalize_callback(self, path: str, event_logger: EventLogger):
        self.pre_finalize_called = True
        self.pre_finalize_path = path
        self.pre_finalize_event_logger = event_logger

    def finalize_callback(self, path: str, event_logger: EventLogger):
        self.finalize_called = True
        self.finalize_path = path
        self.finalize_event_logger = event_logger


class _ConcurrencyProbe:
    def __init__(self, expected_concurrent_writes: int) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._barrier_entries_remaining = expected_concurrent_writes
        self._barrier = (
            threading.Barrier(expected_concurrent_writes)
            if expected_concurrent_writes > 1
            else None
        )
        self.max_active = 0

    def __enter__(self):
        should_wait = False
        barrier = self._barrier
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if barrier is not None and self._barrier_entries_remaining > 0:
                self._barrier_entries_remaining -= 1
                should_wait = True
        if should_wait:
            assert barrier is not None
            barrier.wait(timeout=5)

    def __exit__(self, exc_type, exc_value, traceback):
        with self._lock:
            self._active -= 1


class _ProbeStorage(LocalFileSystemStorage):
    def __init__(self, config: "_ProbeStorageConfig") -> None:
        super().__init__(config)
        self._config = config
        self._probe = config.probe

    def mkdir(self, path: Path, recursive: bool = True) -> None:
        self._config.mkdir_paths.append(path)
        super().mkdir(path, recursive)

    def write(self, path: Path, data) -> None:
        with self._probe:
            super().write(path, data)


class _ProbeStorageConfig(LocalFileSystemStorageConfig):
    def __init__(self, probe: _ConcurrencyProbe) -> None:
        super().__init__(use_direct_io=False)
        self.probe = probe
        self.mkdir_paths: list[Path] = []

    def create_storage(self) -> _ProbeStorage:
        return _ProbeStorage(self)


class TestCheckpointWriterConfig(TestCase):
    def test_default_values(self):
        """Test that CheckpointWriterConfig has the correct default values."""
        options = CheckpointWriterConfig()
        self.assertEqual(options.checkpoint_write_barrier_timeout_sec, 600)
        self.assertIsNone(options.barrier_config)
        self.assertEqual(options.file_write_max_threads, 1)

    def test_custom_values(self):
        """Test that CheckpointWriterConfig can be initialized with custom values."""
        options = CheckpointWriterConfig(checkpoint_write_barrier_timeout_sec=450)
        self.assertEqual(options.checkpoint_write_barrier_timeout_sec, 450)

    def test_file_write_max_threads_rejects_non_positive_values(self):
        """Test that file_write_max_threads must be positive."""
        with self.assertRaisesRegex(ValueError, "file_write_max_threads"):
            CheckpointWriterConfig(file_write_max_threads=0)


class TestCheckpointWriter(TestCase):
    def setUp(self):
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test objects
        self.rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
            role_rank=0,
            role_world_size=1,
        )
        self.config = CheckpointWriterConfig()
        self.mock_callback = MockCallback()

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }
        # Create the storage backend for the writer
        self.storage_config = LocalFileSystemStorageConfig()
        self.storage = self.storage_config.create_storage()

    def _writer(
        self,
        probe: _ConcurrencyProbe,
        *,
        file_write_max_threads: int = 1,
        storage_config: _ProbeStorageConfig | None = None,
    ) -> CheckpointWriter:
        if storage_config is None:
            storage_config = _ProbeStorageConfig(probe)
        return CheckpointWriter(
            CheckpointWriterArgs(
                config=CheckpointWriterConfig(
                    file_write_max_threads=file_write_max_threads
                ),
                rank_info=self.rank_info,
                storage_config=storage_config,
            )
        )

    def _raw_checkpoint_info(self) -> CheckpointWriteInfo:
        return CheckpointWriteInfo(
            checkpoint_items={
                "first": CheckpointItem(
                    value=b"first",
                    layout=LayoutInfo("first.bin", RawSerialization()),
                ),
                "second": CheckpointItem(
                    value=b"second",
                    layout=LayoutInfo("second.bin", RawSerialization()),
                ),
                "metadata": CheckpointItem(
                    value=b"metadata",
                    requires_copy=False,
                    layout=LayoutInfo("metadata.bin", RawSerialization()),
                ),
            }
        )

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_file_write_max_threads_controls_parallel_key_writes(self):
        """Test that file_write_max_threads controls concurrent key writes."""
        for max_threads, expected_max_active in ((1, 1), (2, 2), (3, 3)):
            with self.subTest(max_threads=max_threads):
                probe = _ConcurrencyProbe(expected_max_active)

                self._writer(probe, file_write_max_threads=max_threads).write(
                    path=os.path.join(self.temp_dir, f"checkpoint_{max_threads}"),
                    checkpoint_info=self._raw_checkpoint_info(),
                )

                self.assertEqual(probe.max_active, expected_max_active)

    def test_write_prepares_shared_checkpoint_parent_once(self):
        """Test that shared checkpoint parent directories are deduplicated."""
        probe = _ConcurrencyProbe(expected_concurrent_writes=1)
        storage_config = _ProbeStorageConfig(probe)

        self._writer(probe, storage_config=storage_config).write(
            path=os.path.join(self.temp_dir, "checkpoint_mkdir"),
            checkpoint_info=self._raw_checkpoint_info(),
        )

        self.assertEqual(
            storage_config.mkdir_paths,
            [Path(self.temp_dir) / "checkpoint_mkdir"],
        )

    def test_key_write_metrics_emit_save_task_metric(self):
        """Test that per-key write metrics use the save_task event name."""
        probe = _ConcurrencyProbe(expected_concurrent_writes=1)

        with mock.patch.object(checkpoint_writer_module.logger, "info") as info:
            self._writer(probe).write(
                path=os.path.join(self.temp_dir, "checkpoint_metrics"),
                checkpoint_info=self._raw_checkpoint_info(),
            )

        metric_names = {
            call.kwargs["extra"].get("metric_name")
            for call in info.call_args_list
            if "extra" in call.kwargs
        }

        self.assertIn(
            "train.checkpoint_write.execute.storage.first.save_task.e2e.latency_ms",
            metric_names,
        )
        self.assertIn(
            "train.checkpoint_write.execute.storage.mkdir.latency_ms",
            metric_names,
        )

    def test_write_calls_callbacks(self):
        """Test that write calls the callbacks with correct parameters."""
        # Create writer with callbacks
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
            pre_finalize_callback=self.mock_callback.pre_finalize_callback,
            finalize_callback=self.mock_callback.finalize_callback,
        )
        writer = CheckpointWriter(args=args)

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint")

        # Build CheckpointInfo from state_dict
        items = {
            key: CheckpointItem(value=value, layout=None)
            for key, value in self.state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Call write
        writer.write(checkpoint_path, checkpoint_info)

        # Verify callbacks were called
        self.assertTrue(self.mock_callback.pre_finalize_called)
        self.assertEqual(self.mock_callback.pre_finalize_path, checkpoint_path)
        self.assertIsNotNone(self.mock_callback.pre_finalize_event_logger)
        self.assertIsInstance(self.mock_callback.pre_finalize_event_logger, EventLogger)

        self.assertTrue(self.mock_callback.finalize_called)
        self.assertEqual(self.mock_callback.finalize_path, checkpoint_path)
        self.assertIsNotNone(self.mock_callback.finalize_event_logger)
        self.assertIsInstance(self.mock_callback.finalize_event_logger, EventLogger)

    def test_write_without_callbacks(self):
        """Test that write works correctly without callbacks."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_no_callbacks")

        # Build CheckpointInfo from state_dict
        items = {
            key: CheckpointItem(value=value, layout=None)
            for key, value in self.state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Should not raise any errors
        writer.write(checkpoint_path, checkpoint_info)

    def test_close(self):
        """Test that close doesn't raise any exceptions."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)
        # This is a no-op in the base class, so just verify it doesn't raise
        writer.close()

    def test_write_with_simple_layout(self):
        """Test writing checkpoint with simple layout."""
        # Create a writer without checkpoint_layout
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        # Create test state dict that matches the layout keys
        state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": "5",  # JSON needs string representation
            "step": "1000",
        }

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_layout")

        # Compute layout_info_mappings by calling simple_layout with rank
        layout_info_mappings = simple_layout(self.rank_info.global_rank)

        # Build CheckpointInfo with layout_info_mappings
        items = {
            key: CheckpointItem(
                value=value,
                layout=layout_info_mappings.get(key),
            )
            for key, value in state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Write checkpoint with CheckpointInfo
        writer.write(checkpoint_path, checkpoint_info)

        # Verify files exist based on layout
        model_file = os.path.join(
            checkpoint_path, f"model_rank_{self.rank_info.global_rank}.pt"
        )
        optimizer_file = os.path.join(
            checkpoint_path, f"optimizer_rank_{self.rank_info.global_rank}.pt"
        )
        epoch_file = os.path.join(
            checkpoint_path, f"epoch_rank_{self.rank_info.global_rank}.json"
        )
        step_file = os.path.join(
            checkpoint_path, f"step_rank_{self.rank_info.global_rank}.json"
        )

        self.assertTrue(os.path.exists(model_file))
        self.assertTrue(os.path.exists(optimizer_file))
        self.assertTrue(os.path.exists(epoch_file))
        self.assertTrue(os.path.exists(step_file))

        # Verify content of torch files
        loaded_model = torch.load(model_file)
        loaded_optimizer = torch.load(optimizer_file)
        self.assertIn("weight", loaded_model)
        self.assertEqual(loaded_optimizer["param_groups"][0]["lr"], 0.01)

        # Verify content of JSON files
        with open(epoch_file, "r") as f:
            epoch_content = json.load(f)
            self.assertEqual(epoch_content, "5")  # epoch

        with open(step_file, "r") as f:
            step_content = json.load(f)
            self.assertEqual(step_content, "1000")  # step

    def test_write_with_layout_extra_keys(self):
        """Test that writer ignores extra keys when no layout is provided for them."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        # Create state dict with extra keys not covered by layout
        extra_keys_state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": "5",
            "step": "1000",
            "extra_key": "not_covered_by_layout",  # This key is not in the layout
        }

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_extra_keys")

        # Compute layout_info_mappings by calling simple_layout with rank
        # This layout doesn't include "extra_key"
        layout_info_mappings = simple_layout(self.rank_info.global_rank)

        # Build CheckpointInfo - only include keys that are in the layout
        # extra_key is not in layout, so it's excluded from checkpoint_info
        items = {
            key: CheckpointItem(
                value=value,
                layout=layout_info_mappings.get(key),
            )
            for key, value in extra_keys_state_dict.items()
            if key in layout_info_mappings
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Write checkpoint with CheckpointInfo - extra keys should be ignored
        writer.write(checkpoint_path, checkpoint_info)

        # Verify that files for layout-covered keys exist
        model_file = os.path.join(
            checkpoint_path, f"model_rank_{self.rank_info.global_rank}.pt"
        )
        self.assertTrue(os.path.exists(model_file))

        # Verify that extra_key was NOT written to any file (it was ignored)
        # We just check that the covered files exist and don't check for extra_key

    def test_write_with_global_file_layout(self):
        """Test writing checkpoint with global file layout."""
        args = CheckpointWriterArgs(
            config=self.config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )
        writer = CheckpointWriter(args=args)

        # Create test state dict that matches the global layout keys
        state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "global_config": "some_global_config_value",
        }

        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_global_layout")

        # Compute layout_info_mappings by calling global_file_layout with rank
        layout_info_mappings = global_file_layout(self.rank_info.global_rank)

        # Build CheckpointInfo with layout_info_mappings
        items = {
            key: CheckpointItem(
                value=value,
                layout=layout_info_mappings.get(key),
            )
            for key, value in state_dict.items()
        }
        checkpoint_info = CheckpointWriteInfo(checkpoint_items=items)

        # Write checkpoint with CheckpointInfo
        writer.write(checkpoint_path, checkpoint_info)

        # Verify per-rank files exist with rank suffix
        model_file = os.path.join(
            checkpoint_path, f"model_rank_{self.rank_info.global_rank}.pt"
        )
        optimizer_file = os.path.join(
            checkpoint_path, f"optimizer_rank_{self.rank_info.global_rank}.pt"
        )

        # Verify global file exists WITHOUT rank suffix
        global_config_file = os.path.join(checkpoint_path, "config.json")

        self.assertTrue(os.path.exists(model_file))
        self.assertTrue(os.path.exists(optimizer_file))
        self.assertTrue(
            os.path.exists(global_config_file)
        )  # No rank suffix for global files

        # Verify content of torch files
        loaded_model = torch.load(model_file)
        loaded_optimizer = torch.load(optimizer_file)
        self.assertIn("weight", loaded_model)
        self.assertEqual(loaded_optimizer["param_groups"][0]["lr"], 0.01)

        # Verify content of global JSON file
        with open(global_config_file, "r") as f:
            import json

            global_data = json.load(f)
            # The JSON file contains just the raw value, not wrapped in an object
            self.assertEqual(global_data, "some_global_config_value")


if __name__ == "__main__":
    run_tests()
