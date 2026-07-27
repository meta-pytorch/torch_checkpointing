# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_checkpointing.builder import (
    make_async_checkpoint_saver,
    make_sync_checkpoint_saver,
)
from torch_checkpointing.checkpoint_base import CheckpointItem
from torch_checkpointing.checkpoint_loader import CheckpointLoader
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.checkpoint_saver import (
    AsyncCheckpointSaver,
    CheckpointBase,
    SyncCheckpointSaver,
)
from torch_checkpointing.checkpoint_writer import CheckpointWriterConfig
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo, STATE_DICT


class SimpleCheckpoint(CheckpointBase):
    """Simple checkpoint wrapper for testing that wraps a state dict."""

    def __init__(self, state_dict, keys_requiring_copy_list=None):
        self._state_dict = state_dict
        self._keys_requiring_copy_list = keys_requiring_copy_list

    def get_items(self) -> dict[str, CheckpointItem]:
        """Return a dict of CheckpointItem objects representing the checkpoint."""
        items = {}
        for key, value in self._state_dict.items():
            requires_copy = True
            if self._keys_requiring_copy_list is not None:
                requires_copy = key in self._keys_requiring_copy_list
            items[key] = CheckpointItem(
                value=value,
                requires_copy=requires_copy,
            )
        return items

    def load_state_dict(self, state_dict: STATE_DICT) -> None:
        """Load the state from a loaded state dictionary into this checkpoint object."""
        self._state_dict.update(state_dict)


class TestMakeCheckpointer(TestCase):
    def setUp(self) -> None:
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create real objects for testing
        self.rank_info = RankInfo(
            global_world_size=1,
            global_rank=0,
            role_rank=0,
            role_world_size=1,
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": 5,
            "step": 1000,
        }

        # Specify the storage backend to use for testing
        self.storage_config = LocalFileSystemStorageConfig()

    def tearDown(self) -> None:
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_make_sync_checkpoint_saver(self) -> None:
        """Test creating a synchronous checkpointer using make_sync_checkpoint_saver."""

        # Create sync checkpointer using factory function with no barrier
        writer_config = CheckpointWriterConfig()
        config = SyncCheckpointSaverConfig(writer_config=writer_config)
        checkpointer = make_sync_checkpoint_saver(
            config=config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )

        # Verify it's a SyncCheckpointSaver instance
        self.assertIsInstance(checkpointer, SyncCheckpointSaver)

        # Test that it works for sync operations
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint_factory_sync")
        result = checkpointer.save(checkpoint_path, SimpleCheckpoint(self.state_dict))
        self.assertIsNone(result)  # Sync mode returns None

        # Verify checkpoint was created
        for key in self.state_dict.keys():
            checkpoint_file = os.path.join(
                checkpoint_path, f"{key}_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(checkpoint_file))

        # Test loading using CheckpointLoader
        loader = CheckpointLoader(
            reader=CheckpointReader(
                rank_info=self.rank_info,
                storage_config=self.storage_config,
            ),
        )
        loaded_checkpoint = SimpleCheckpoint({"epoch": 0})
        loader.load(checkpoint_path, loaded_checkpoint)
        self.assertEqual(loaded_checkpoint._state_dict["epoch"], 5)
        loader.close()

    def test_make_sync_checkpoint_saver_with_config_first(self) -> None:
        """Test creating a synchronous checkpointer with config as first parameter."""
        # Create sync checkpointer with config as first parameter
        writer_config = CheckpointWriterConfig()
        config = SyncCheckpointSaverConfig(writer_config=writer_config)
        checkpointer = make_sync_checkpoint_saver(
            config=config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )

        # Verify it's a SyncCheckpointSaver instance
        self.assertIsInstance(checkpointer, SyncCheckpointSaver)

        # Test that it works for sync operations
        checkpoint_path = os.path.join(
            self.temp_dir, "checkpoint_factory_sync_config_first"
        )
        result = checkpointer.save(checkpoint_path, SimpleCheckpoint(self.state_dict))
        self.assertIsNone(result)  # Sync mode returns None

        # Verify checkpoint was created
        for key in self.state_dict.keys():
            checkpoint_file = os.path.join(
                checkpoint_path, f"{key}_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(checkpoint_file))

    def test_make_sync_checkpoint_saver_with_custom_config(self) -> None:
        """Test creating a synchronous checkpointer with a custom config."""
        # Create a custom config with no barrier
        writer_config = CheckpointWriterConfig()
        config = SyncCheckpointSaverConfig(writer_config=writer_config)

        # Create sync checkpointer with the custom config
        checkpointer = make_sync_checkpoint_saver(
            rank_info=self.rank_info,
            config=config,
            storage_config=self.storage_config,
        )

        # Verify it's a SyncCheckpointSaver instance
        self.assertIsInstance(checkpointer, SyncCheckpointSaver)

        # Test that it works for sync operations
        checkpoint_path = os.path.join(
            self.temp_dir, "checkpoint_factory_sync_custom_config"
        )
        result = checkpointer.save(checkpoint_path, SimpleCheckpoint(self.state_dict))
        self.assertIsNone(result)  # Sync mode returns None

        # Verify checkpoint was created
        for key in self.state_dict.keys():
            checkpoint_file = os.path.join(
                checkpoint_path, f"{key}_{self.rank_info.global_rank}.pt"
            )
            self.assertTrue(os.path.exists(checkpoint_file))

        # Test loading using CheckpointLoader
        loader = CheckpointLoader(
            reader=CheckpointReader(
                rank_info=self.rank_info,
                storage_config=self.storage_config,
            ),
        )
        loaded_checkpoint = SimpleCheckpoint({"epoch": 0})
        loader.load(checkpoint_path, loaded_checkpoint)
        self.assertEqual(loaded_checkpoint._state_dict["epoch"], 5)
        loader.close()

    def test_make_async_checkpoint_saver(self) -> None:
        """Test creating an asynchronous checkpointer using make_async_checkpoint_saver."""
        # Create async checkpointer using factory function with default parameters
        config = AsyncCheckpointSaverConfig(
            staging_config=CheckpointStagerConfig(
                use_non_blocking_copy=torch.accelerator.is_available(),
                use_pinned_memory=torch.accelerator.is_available(),
            )
        )
        checkpointer = make_async_checkpoint_saver(
            config=config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )

        try:
            # Verify it's an AsyncCheckpointSaver instance
            self.assertIsInstance(checkpointer, AsyncCheckpointSaver)

            # Test that it works for async operations
            checkpoint_path = os.path.join(self.temp_dir, "checkpoint_factory_async")
            result = checkpointer.save(
                checkpoint_path, SimpleCheckpoint(self.state_dict)
            )

            # For async checkpointer, result should be a tuple of futures
            self.assertIsNotNone(result)
            self.assertIsInstance(result, tuple)

            if result is not None:
                self.assertEqual(len(result), 2)
                stage_future, write_future = result

                # Verify futures are returned
                self.assertIsNotNone(stage_future)
                self.assertIsNotNone(write_future)

                # Wait for completion
                stage_future.result()
                write_future.result()
            else:
                self.fail("Expected tuple of futures but got None")

            # Verify checkpoint was created
            for key in self.state_dict.keys():
                checkpoint_file = os.path.join(
                    checkpoint_path, f"{key}_{self.rank_info.global_rank}.pt"
                )
                self.assertTrue(os.path.exists(checkpoint_file))

            # Test loading using CheckpointLoader
            loader = CheckpointLoader(
                reader=CheckpointReader(
                    rank_info=self.rank_info,
                    storage_config=self.storage_config,
                ),
            )
            loaded_checkpoint = SimpleCheckpoint({"epoch": 0})
            loader.load(checkpoint_path, loaded_checkpoint)
            self.assertEqual(loaded_checkpoint._state_dict["epoch"], 5)
            loader.close()

        finally:
            # Clean up
            checkpointer.close()

    def test_make_async_checkpoint_saver_propagates_zero_wait_timeout(self) -> None:
        config = AsyncCheckpointSaverConfig(
            wait_timeout_secs=0,
            staging_config=CheckpointStagerConfig(
                use_non_blocking_copy=False,
                use_pinned_memory=False,
            ),
        )
        checkpointer = make_async_checkpoint_saver(
            config=config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )

        try:
            with checkpointer.staging_lock.read:
                with self.assertRaisesRegex(RuntimeError, "after 0 seconds"):
                    checkpointer.save(
                        os.path.join(self.temp_dir, "timeout_checkpoint"),
                        SimpleCheckpoint(self.state_dict),
                    )
        finally:
            checkpointer.close()

    def test_make_async_checkpoint_saver_uses_default_lock_timeout_for_none(
        self,
    ) -> None:
        config = AsyncCheckpointSaverConfig(
            wait_timeout_secs=None,
            staging_config=CheckpointStagerConfig(
                use_non_blocking_copy=False,
                use_pinned_memory=False,
            ),
        )
        checkpointer = make_async_checkpoint_saver(
            config=config,
            rank_info=self.rank_info,
            storage_config=self.storage_config,
        )

        try:
            with checkpointer.staging_lock.read:
                self.assertEqual(checkpointer.staging_lock.locked_mode(), "read")
        finally:
            checkpointer.close()


if __name__ == "__main__":
    run_tests()
