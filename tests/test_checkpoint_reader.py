# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil
import tempfile
from typing import Any

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_checkpointing.checkpoint_base import (
    CheckpointBase,
    CheckpointInfo,
    CheckpointItem,
)
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    TorchSerialization,
)
from torch_checkpointing.checkpoint_reader import (
    CheckpointReader,
)
from torch_checkpointing.distributed_metadata import ShardingMetadata
from torch_checkpointing.resharding import Resharder
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import NestedPath, RankInfo, STATE_DICT


def _read_wrapper(
    reader: CheckpointReader,
    path: str,
    checkpoint: CheckpointBase,
    map_location: Any = None,
) -> tuple[STATE_DICT, list[str]]:
    """
    Helper function to call CheckpointReader.read().

    This extracts CheckpointInfo from the checkpoint and calls the read() API.
    """
    items = checkpoint.get_items()
    checkpoint_info = CheckpointInfo(checkpoint_items=items)

    return reader.read(
        path=path,
        checkpoint_info=checkpoint_info.for_reads(),
        map_location=map_location,
    )


class NoOpResharder(Resharder):
    """A no-op resharder that never reshards but forces normal code path.

    When configured on checkpoint items, this resharder ensures that the
    CheckpointReader goes through the normal path (with metadata loading)
    rather than the fast path (which skips metadata and uses full file reads).
    """

    def __init__(self, skip_resharding: bool = False):
        self._skip_resharding = skip_resharding

    @property
    def skip_resharding(self) -> bool:
        return self._skip_resharding

    def extract_sharding_metadata(
        self, item_key: str, item_value: Any
    ) -> dict[NestedPath, ShardingMetadata]:
        """Extract metadata - returns empty dict since no-op."""
        return {}

    def load(
        self,
        source_path,
        item_key,
        target_metadata,
        source_metadata,
        target,
        storage,
    ) -> list[NestedPath]:
        # No-op: never actually loads anything
        return []

    def should_reshard(self, source_metadata, target_metadata):
        return False  # Never actually reshard


class SimpleCheckpoint(CheckpointBase):
    """Simple checkpoint wrapper for testing that wraps a state dict."""

    def __init__(
        self,
        state_dict,
        layout_info_mappings=None,
        use_resharder=True,
        skip_resharding: bool = False,
    ):
        self._state_dict = state_dict
        self._layout_info_mappings = layout_info_mappings
        self._use_resharder = use_resharder
        self._skip_resharding = skip_resharding

    def get_items(self) -> dict[str, CheckpointItem]:
        """Return a dict of CheckpointItem objects representing the checkpoint."""
        items = {}
        for key, value in self._state_dict.items():
            layout = None
            if self._layout_info_mappings is not None:
                layout = self._layout_info_mappings.get(key)
            resharder = None
            if self._use_resharder:
                resharder = NoOpResharder(skip_resharding=self._skip_resharding)
            items[key] = CheckpointItem(
                value=value,
                layout=layout,
                resharder=resharder,
            )
        return items

    def load_state_dict(self, state_dict: STATE_DICT) -> None:
        """Load the state from a loaded state dictionary into this checkpoint object."""
        self._state_dict.update(state_dict)

    @classmethod
    def from_dict(cls, state_dict):
        """Create a SimpleCheckpoint from a state dictionary."""
        return cls(state_dict)


class TestCheckpointReader(TestCase):
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

        # Create the checkpoint reader
        self.reader = CheckpointReader(
            rank_info=self.rank_info,
            storage_config=LocalFileSystemStorageConfig(),
        )

        # Create a test state dictionary
        self.state_dict = {
            "model": {
                "weight": torch.randn(10, 5),
                "bias": torch.randn(5),
                "test_list": [torch.randn(2), torch.randn(2)],
            },
            "optimizer": {
                "param_groups": [
                    {"lr": 0.01, "test_list": [torch.randn(2), torch.randn(2)]}
                ]
            },
            "epoch": 5,
            "step": 1000,
        }

        # Create a test checkpoint file
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoint")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            self.checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        torch.save(self.state_dict, checkpoint_file)

    def move_tensors_to_device(self, state_dict: Any, device: str) -> Any:
        """
        Recursively move all tensors in a nested dictionary to CUDA.

        Args:
            state_dict (dict): A dictionary potentially containing nested dictionaries and tensors.

        Returns:
            dict: A new dictionary with all tensors moved to CUDA.
        """
        if isinstance(state_dict, dict):
            return {
                key: self.move_tensors_to_device(value, device)
                for key, value in state_dict.items()
            }
        elif isinstance(state_dict, list):
            return [self.move_tensors_to_device(item, device) for item in state_dict]
        elif isinstance(state_dict, torch.Tensor):
            return state_dict.cuda() if device == "cpu" else state_dict.cpu()
        else:
            return state_dict

    def deep_compare(self, obj1: Any, obj2: Any) -> bool:
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            if obj1.keys() != obj2.keys():
                return False
            return all(self.deep_compare(obj1[key], obj2[key]) for key in obj1)
        elif isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            return all(
                self.deep_compare(item1, item2) for item1, item2 in zip(obj1, obj2)
            )
        elif isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
            return torch.equal(obj1, obj2)
        else:
            return obj1 == obj2

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_read_checkpoint(self):
        """Test that read correctly reads a checkpoint file."""

        # Create a template checkpoint with layout for all keys
        template_state_dict = {k: None for k in self.state_dict.keys()}
        template_checkpoint = SimpleCheckpoint({"checkpoint": template_state_dict})

        # Call read
        read_state_dict, missing_keys = _read_wrapper(
            self.reader, self.checkpoint_path, template_checkpoint
        )
        read_state_dict = read_state_dict["checkpoint"]
        self.assertEqual(missing_keys, [])

        # Verify that the read state dictionary contains the expected values
        self.assertIn("model", read_state_dict)
        self.assertIn("optimizer", read_state_dict)
        self.assertTrue(self.deep_compare(read_state_dict, self.state_dict))

        # No hooks to verify since we removed them

    def test_read_with_map_location(self):
        """Test that read correctly uses the map_location parameter."""
        # Call read with map_location='cpu'
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        template_state_dict = {k: None for k in self.state_dict.keys()}
        template_checkpoint = SimpleCheckpoint({"checkpoint": template_state_dict})
        read_state_dict, _ = _read_wrapper(
            self.reader,
            self.checkpoint_path,
            template_checkpoint,
            map_location=map_location,
        )
        read_state_dict = read_state_dict["checkpoint"]

        # Verify that the read state dictionary contains the expected values
        self.assertIn("model", read_state_dict)
        self.assertIn("optimizer", read_state_dict)
        self.assertEqual(read_state_dict["epoch"], 5)
        self.assertEqual(read_state_dict["step"], 1000)
        self.assertEqual(read_state_dict["model"]["weight"].device.type, map_location)

    def test_read_nonexistent_checkpoint(self):
        """Test that read raises FileNotFoundError for a nonexistent checkpoint."""
        # Set up a path to a nonexistent checkpoint
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent_checkpoint")

        # Call read and expect a FileNotFoundError
        template_state_dict = {k: None for k in self.state_dict.keys()}
        template_checkpoint = SimpleCheckpoint({"checkpoint": template_state_dict})
        with self.assertRaises(FileNotFoundError):
            _read_wrapper(self.reader, nonexistent_path, template_checkpoint)

    def test_partial_read(self):
        """Test that read with state_dict correctly loads only the requested keys."""
        # Create a partial state dictionary with only some keys
        partial_state_dict = {}
        partial_state_dict["optimizer"] = None
        partial_state_dict["model"] = {"weight": torch.randn(10, 5)}
        partial_state_dict["epoch"] = None
        # Create layout for partial read
        layout_info = {
            "checkpoint": LayoutInfo(
                file_path=f"checkpoint_{self.rank_info.global_rank}.pt",
                serialization_format=TorchSerialization(),
            )
        }
        partial_checkpoint = SimpleCheckpoint(
            {"checkpoint": partial_state_dict}, layout_info_mappings=layout_info
        )
        # Call read with state_dict
        updated_state_dict, missing = _read_wrapper(
            self.reader,
            self.checkpoint_path,
            partial_checkpoint,
        )
        updated_state_dict = updated_state_dict["checkpoint"]

        # Verify that the updated state dictionary contains values from both dictionaries
        self.assertIn("model", updated_state_dict)
        self.assertIn("epoch", updated_state_dict)
        self.assertTrue(
            torch.equal(
                updated_state_dict["model"]["weight"],
                self.state_dict["model"]["weight"],
            )
        )

        self.assertTrue(
            self.deep_compare(
                updated_state_dict["optimizer"],
                self.state_dict["optimizer"],
            )
        )
        self.assertEqual(updated_state_dict["epoch"], 5)  # From checkpoint

        self.assertNotIn("bias", updated_state_dict["model"])
        self.assertNotIn("step", updated_state_dict)

    def test_partial_read_missing_keys(self):
        """Test that partial_read correctly reports missing keys."""
        # Create a partial state dictionary with keys that don't exist in the checkpoint
        partial_state_dict = {
            "model": None,
            "nonexistent_key": None,  # This key doesn't exist in the checkpoint
            "another_missing_key": {"nested": None},  # This key also doesn't exist
        }

        # Create layout for partial read
        # use_resharder=True (default) ensures we go through normal path with missing key tracking
        partial_checkpoint = SimpleCheckpoint({"checkpoint": partial_state_dict})

        # Call read with state_dict
        _, missing_keys = _read_wrapper(
            self.reader,
            self.checkpoint_path,
            partial_checkpoint,
        )

        # Verify that missing keys are correctly reported
        self.assertIn("checkpoint::nonexistent_key", missing_keys)
        self.assertIn("checkpoint::another_missing_key", missing_keys)

        # Verify that keys that exist in the checkpoint are not in missing_keys
        self.assertNotIn("checkpoint::model", missing_keys)

    def test_read_different_dtypes(self):
        """Test that read correctly handles different tensor dtypes."""
        # Create a state dictionary with tensors of different dtypes
        dtype_state_dict = {
            "float32": torch.randn(10, 10, dtype=torch.float32),
            "float64": torch.randn(10, 10, dtype=torch.float64),
            "int32": torch.randint(-100, 100, (10, 10), dtype=torch.int32),
            "int64": torch.randint(-100, 100, (10, 10), dtype=torch.int64),
            "bool": torch.randint(0, 2, (10, 10), dtype=torch.bool),
        }

        # Save the state dictionary
        dtype_checkpoint_path = os.path.join(self.temp_dir, "dtype_checkpoint")
        os.makedirs(dtype_checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            dtype_checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        torch.save(dtype_state_dict, checkpoint_file)

        # Create a partial state dictionary requesting tensors of each dtype
        partial_state_dict = {
            "float32": torch.randn(10, 10, dtype=torch.float32),
            "float64": None,
            "int32": None,
            "int64": None,
            "bool": None,
        }

        partial_checkpoint = SimpleCheckpoint({"checkpoint": partial_state_dict})

        # Load the partial state dictionary
        updated_state_dict, _ = _read_wrapper(
            self.reader,
            os.path.dirname(checkpoint_file),
            partial_checkpoint,
        )
        updated_state_dict = updated_state_dict["checkpoint"]

        # Verify that tensors of each dtype were loaded correctly
        for key in dtype_state_dict:
            self.assertIn(key, updated_state_dict)
            self.assertEqual(updated_state_dict[key].dtype, dtype_state_dict[key].dtype)
            self.assertTrue(
                torch.allclose(updated_state_dict[key], dtype_state_dict[key])
            )

    def test_read_with_simple_layout(self):
        """Test reading checkpoint with simple layout."""
        # Create test data that will be stored according to layout
        test_data = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": "5",
            "step": "1000",
        }

        # Create checkpoint files in the expected format
        layout_checkpoint_path = os.path.join(self.temp_dir, "layout_checkpoint")
        os.makedirs(layout_checkpoint_path, exist_ok=True)

        # Create the individual files in format
        model_file = os.path.join(
            layout_checkpoint_path, f"model_{self.rank_info.global_rank}.pt"
        )
        optimizer_file = os.path.join(
            layout_checkpoint_path, f"optimizer_{self.rank_info.global_rank}.pt"
        )
        epoch_file = os.path.join(
            layout_checkpoint_path, f"epoch_{self.rank_info.global_rank}.json"
        )
        step_file = os.path.join(
            layout_checkpoint_path, f"step_{self.rank_info.global_rank}.json"
        )

        torch.save(test_data["model"], model_file)
        torch.save(test_data["optimizer"], optimizer_file)
        with open(epoch_file, "w") as f:
            json.dump(test_data["epoch"], f)
        with open(step_file, "w") as f:
            json.dump(test_data["step"], f)

        # Create reader
        reader = CheckpointReader(
            rank_info=self.rank_info,
            storage_config=LocalFileSystemStorageConfig(),
        )

        # Create layout info for the checkpoint
        layout_info = {
            "model": LayoutInfo(
                file_path=f"model_{self.rank_info.global_rank}.pt",
                serialization_format=TorchSerialization(),
            ),
            "optimizer": LayoutInfo(
                file_path=f"optimizer_{self.rank_info.global_rank}.pt",
                serialization_format=TorchSerialization(),
            ),
            "epoch": LayoutInfo(
                file_path=f"epoch_{self.rank_info.global_rank}.json",
                serialization_format=JsonSerialization(str),
            ),
            "step": LayoutInfo(
                file_path=f"step_{self.rank_info.global_rank}.json",
                serialization_format=JsonSerialization(str),
            ),
        }
        template_state_dict = {k: None for k in test_data.keys()}
        template_checkpoint = SimpleCheckpoint(
            template_state_dict, layout_info_mappings=layout_info
        )

        # Read the entire checkpoint
        loaded_state_dict, missing_keys = _read_wrapper(
            reader, layout_checkpoint_path, template_checkpoint
        )

        # Verify no missing keys
        self.assertEqual(missing_keys, [])

        # Verify all expected keys are present
        self.assertIn("model", loaded_state_dict)
        self.assertIn("optimizer", loaded_state_dict)
        self.assertIn("epoch", loaded_state_dict)
        self.assertIn("step", loaded_state_dict)

        # Verify content
        self.assertEqual(loaded_state_dict["optimizer"]["param_groups"][0]["lr"], 0.01)

    def test_read_partial_with_layout(self):
        """Test partial reading checkpoint with layout."""
        # Create test data that will be stored according to layout
        test_data = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "optimizer": {"param_groups": [{"lr": 0.01}]},
            "epoch": "5",
            "step": "1000",
        }

        # Create checkpoint files in the expected format
        layout_checkpoint_path = os.path.join(
            self.temp_dir, "partial_layout_checkpoint"
        )
        os.makedirs(layout_checkpoint_path, exist_ok=True)

        # Create the individual files
        model_file = os.path.join(
            layout_checkpoint_path, f"model_{self.rank_info.global_rank}.pt"
        )
        optimizer_file = os.path.join(
            layout_checkpoint_path, f"optimizer_{self.rank_info.global_rank}.pt"
        )
        epoch_file = os.path.join(
            layout_checkpoint_path, f"epoch_{self.rank_info.global_rank}.json"
        )
        step_file = os.path.join(
            layout_checkpoint_path, f"step_{self.rank_info.global_rank}.json"
        )

        torch.save(test_data["model"], model_file)
        torch.save(test_data["optimizer"], optimizer_file)
        with open(epoch_file, "w") as f:
            json.dump(test_data["epoch"], f)
        with open(step_file, "w") as f:
            json.dump(test_data["step"], f)

        # Create reader
        reader = CheckpointReader(
            rank_info=self.rank_info,
            storage_config=LocalFileSystemStorageConfig(),
        )

        # Create partial state dict - only request model and epoch
        partial_state_dict = {
            "model": torch.nn.Linear(10, 5).state_dict(),
            "epoch": None,
        }

        # Create layout info for partial read
        layout_info = {
            "model": LayoutInfo(
                file_path=f"model_{self.rank_info.global_rank}.pt",
                serialization_format=TorchSerialization(),
            ),
            "epoch": LayoutInfo(
                file_path=f"epoch_{self.rank_info.global_rank}.json",
                serialization_format=JsonSerialization(str),
            ),
        }
        partial_checkpoint = SimpleCheckpoint(
            partial_state_dict, layout_info_mappings=layout_info
        )

        # Read partially
        loaded_state_dict, missing_keys = _read_wrapper(
            reader, layout_checkpoint_path, partial_checkpoint
        )

        # Verify content - should only have model and epoch
        self.assertIn("model", loaded_state_dict)
        self.assertIn("epoch", loaded_state_dict)
        # Should not have optimizer or step since not requested
        self.assertNotIn("optimizer", loaded_state_dict)
        self.assertNotIn("step", loaded_state_dict)

        # Verify no missing keys for requested items
        self.assertEqual(missing_keys, [])

    def test_read_empty_tensor(self):
        """Test that read correctly handles tensors with 0 elements (tensor_len == 0)."""
        # Create a state dictionary with empty tensors
        empty_tensor_state_dict = {
            "empty_tensor": torch.empty(0),
            "empty_2d": torch.empty(0, 5),
            "empty_3d": torch.empty(2, 0, 3),
            "normal_tensor": torch.randn(3, 3),
        }

        # Save the state dictionary
        empty_checkpoint_path = os.path.join(self.temp_dir, "empty_checkpoint")
        os.makedirs(empty_checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(
            empty_checkpoint_path, f"checkpoint_{self.rank_info.global_rank}.pt"
        )
        torch.save(empty_tensor_state_dict, checkpoint_file)

        # Create a partial state dictionary requesting the empty tensors
        partial_state_dict = {
            "empty_tensor": None,
            "empty_2d": None,
            "empty_3d": None,
            "normal_tensor": None,
        }

        # Load the partial state dictionary
        partial_checkpoint = SimpleCheckpoint({"checkpoint": partial_state_dict})
        updated_state_dict, missing_keys = _read_wrapper(
            self.reader,
            os.path.dirname(checkpoint_file),
            partial_checkpoint,
        )
        updated_state_dict = updated_state_dict["checkpoint"]

        for key, value in empty_tensor_state_dict.items():
            self.assertIn(key, updated_state_dict)
            self.assertTrue(torch.allclose(value, updated_state_dict[key]))
            self.assertEqual(value.numel(), updated_state_dict[key].numel())
            self.assertEqual(value.shape, updated_state_dict[key].shape)

    def test_read_without_resharders_skips_metadata(self):
        """Test that read without resharders configured skips metadata loading."""
        # Create a template checkpoint with layout for all keys but NO resharders
        template_state_dict = dict.fromkeys(self.state_dict.keys(), None)
        template_checkpoint = SimpleCheckpoint(
            {"checkpoint": template_state_dict}, use_resharder=False
        )

        # Get checkpoint info - SimpleCheckpoint doesn't configure resharders
        items = template_checkpoint.get_items()
        checkpoint_info = CheckpointInfo(checkpoint_items=items)

        # Verify no resharders are configured
        for item in checkpoint_info.checkpoint_items.values():
            self.assertIsNone(item.resharder)

        # Call read - should skip metadata loading and use direct file reads
        read_state_dict, missing_keys = self.reader.read(
            path=self.checkpoint_path,
            checkpoint_info=checkpoint_info,
        )
        read_state_dict = read_state_dict["checkpoint"]

        # Verify no missing keys
        self.assertEqual(missing_keys, [])

        # Verify that the read state dictionary contains the expected values
        self.assertIn("model", read_state_dict)
        self.assertIn("optimizer", read_state_dict)
        self.assertIn("epoch", read_state_dict)
        self.assertTrue(self.deep_compare(read_state_dict, self.state_dict))

    def test_no_resharders_uses_full_file_read(self):
        """Test that no resharders uses full file read, then filters to requested keys."""
        # Create a partial state dictionary - requesting only some keys
        # Without resharders, we use the fast path which does full file reads
        # but then filters to only the requested structure
        partial_state_dict = {
            "model": {"weight": torch.randn(10, 5)},  # Only request weight
        }

        layout_info = {
            "checkpoint": LayoutInfo(
                file_path=f"checkpoint_{self.rank_info.global_rank}.pt",
                serialization_format=TorchSerialization(),
            )
        }
        partial_checkpoint = SimpleCheckpoint(
            {"checkpoint": partial_state_dict},
            layout_info_mappings=layout_info,
            use_resharder=False,
        )

        # Get checkpoint info - no resharders configured
        items = partial_checkpoint.get_items()
        checkpoint_info = CheckpointInfo(checkpoint_items=items)

        # Verify no resharders are configured
        for item in checkpoint_info.checkpoint_items.values():
            self.assertIsNone(item.resharder)

        # Call read - should use fast path with filtering
        read_state_dict, _ = self.reader.read(
            path=self.checkpoint_path,
            checkpoint_info=checkpoint_info,
        )
        read_state_dict = read_state_dict["checkpoint"]

        # Fast path loads full file, then filters to requested keys
        # So we should only get the requested structure (model with weight)
        # not the entire checkpoint
        self.assertIn("model", read_state_dict)
        self.assertIn("weight", read_state_dict["model"])
        # Keys not in the requested structure are filtered out
        self.assertNotIn("optimizer", read_state_dict)
        self.assertNotIn("epoch", read_state_dict)
        # Verify model only contains requested keys (weight, not bias)
        self.assertNotIn("bias", read_state_dict["model"])

    def test_no_resharders_nonexistent_checkpoint(self):
        """Test that no resharders configured raises FileNotFoundError for nonexistent checkpoint."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent_checkpoint")

        template_state_dict = dict.fromkeys(self.state_dict.keys(), None)
        template_checkpoint = SimpleCheckpoint(
            {"checkpoint": template_state_dict}, use_resharder=False
        )

        items = template_checkpoint.get_items()
        checkpoint_info = CheckpointInfo(checkpoint_items=items)

        with self.assertRaises(FileNotFoundError):
            self.reader.read(
                path=nonexistent_path,
                checkpoint_info=checkpoint_info,
            )

    def test_resharder_skip_resharding_uses_fast_path(self):
        """Test that resharders with skip_resharding=True use the fast path."""
        # Create a template checkpoint with resharders that have skip_resharding=True
        template_state_dict = dict.fromkeys(self.state_dict.keys(), None)
        template_checkpoint = SimpleCheckpoint(
            {"checkpoint": template_state_dict},
            use_resharder=True,
            skip_resharding=True,  # This should trigger fast path
        )

        # Get checkpoint info - resharders are configured but with skip_resharding=True
        items = template_checkpoint.get_items()
        checkpoint_info = CheckpointInfo(checkpoint_items=items)

        # Verify resharders are configured with skip_resharding=True
        for item in checkpoint_info.checkpoint_items.values():
            self.assertIsNotNone(item.resharder)
            self.assertTrue(item.resharder.skip_resharding)

        # Call read - should skip metadata loading and use direct file reads
        read_state_dict, missing_keys = self.reader.read(
            path=self.checkpoint_path,
            checkpoint_info=checkpoint_info,
        )
        read_state_dict = read_state_dict["checkpoint"]

        # Verify no missing keys
        self.assertEqual(missing_keys, [])

        # Verify that the read state dictionary contains the expected values
        self.assertIn("model", read_state_dict)
        self.assertIn("optimizer", read_state_dict)
        self.assertIn("epoch", read_state_dict)
        self.assertTrue(self.deep_compare(read_state_dict, self.state_dict))

    def test_missing_keys_reported(self):
        """Test behavior when requesting keys that don't exist in checkpoint.

        Scenario 1: Request only keys that exist - should succeed with no missing keys.
        Scenario 2: Request a key whose file doesn't exist - should raise RuntimeError.
        """
        # Create a checkpoint that only has 'model' saved
        single_item_path = os.path.join(self.temp_dir, "single_item_checkpoint")
        os.makedirs(single_item_path, exist_ok=True)

        # Only save model
        model_layout = LayoutInfo(
            file_path=f"model_{self.rank_info.global_rank}.pt",
            serialization_format=TorchSerialization(),
        )
        torch.save(
            self.state_dict["model"],
            os.path.join(single_item_path, model_layout.file_path),
        )

        # Create CheckpointInfo with only model (optimizer not requested)
        checkpoint_items = {
            "model": CheckpointItem(
                value=None,
                layout=model_layout,
                resharder=None,
            ),
        }
        checkpoint_info = CheckpointInfo(checkpoint_items=checkpoint_items)

        # Read the checkpoint - should only load model
        loaded_state_dict, missing_keys = self.reader._read_without_resharding(
            path=single_item_path,
            checkpoint_info=checkpoint_info,
        )

        # Model should be loaded
        self.assertIn("model", loaded_state_dict)
        # No missing keys since we only requested what exists
        self.assertEqual(len(missing_keys), 0)

        # Now test that requesting a non-existent file raises RuntimeError
        checkpoint_items_with_missing = {
            "model": CheckpointItem(
                value=None,
                layout=model_layout,
                resharder=None,
            ),
            "optimizer": CheckpointItem(
                value=self.state_dict["optimizer"],
                layout=None,  # Will use default layout, file won't exist
                resharder=None,
            ),
        }
        checkpoint_info_with_missing = CheckpointInfo(
            checkpoint_items=checkpoint_items_with_missing
        )

        # Should raise RuntimeError for missing optimizer file
        with self.assertRaises(RuntimeError) as context:
            self.reader._read_without_resharding(
                path=single_item_path,
                checkpoint_info=checkpoint_info_with_missing,
            )
        self.assertIn("Missing file", str(context.exception))
        self.assertIn("optimizer", str(context.exception))


if __name__ == "__main__":
    run_tests()
