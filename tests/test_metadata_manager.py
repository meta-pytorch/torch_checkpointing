"""
Tests for the metadata manager in distributed checkpointing.

This module tests the metadata extraction, aggregation, and validation
functionality for distributed tensor checkpointing across various configurations.
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor.placement_types import (
    Replicate as DTensorReplicate,
    Shard as DTensorShard,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._api import DTensor
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch_checkpointing.checkpoint_base import CheckpointInfo
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    TorchSerialization,
)
from torch_checkpointing.distributed_metadata import (
    DistributedMetadata,
    ShardingMetadata,
)
from torch_checkpointing.dtensor_metadata import DTensorShardingMetadata
from torch_checkpointing.metadata_manager import DefaultMetadataManager
from torch_checkpointing.types import RankInfo

from .resharding_test_utils import (
    CustomObject,
    CustomShardingMetadata,
    CustomTensorObject,
    make_checkpoint_info_from_dict,
)


class TestMetadataManager(DTensorTestBase):
    """
    Test suite for metadata manager functionality.

    Tests various scenarios:
    1. Object metadata extraction (DTensor, custom, unknown)
    2. Multi-rank metadata aggregation
    3. Cross-rank validation
    4. Different metadata types and flags
    5. Error handling and edge cases
    """

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @property
    def world_size(self) -> int:
        return 2

    def _create_rank_info(self) -> RankInfo:
        """Create RankInfo for current distributed setup."""
        if dist.is_initialized():
            return RankInfo(
                global_rank=dist.get_rank(),
                global_world_size=dist.get_world_size(),
                role_rank=dist.get_rank(),
                role_world_size=dist.get_world_size(),
            )
        else:
            return RankInfo(
                global_rank=0,
                global_world_size=1,
                role_rank=0,
                role_world_size=1,
            )

    def _create_simple_state_dict(self, device_mesh: DeviceMesh) -> dict:
        """Create a simple state dict with DTensors for testing."""
        # Get current rank, defaulting to 0 if distributed is not initialized
        current_rank = dist.get_rank() if dist.is_initialized() else 0

        # Create local tensors with different shapes for different placements
        local_tensor_1 = torch.randn(4, 8, device=self.device_type) + current_rank
        local_tensor_2 = torch.randn(6, 4, device=self.device_type) + current_rank * 10

        # Create DTensors with different sharding strategies
        dtensor_1 = DTensor.from_local(local_tensor_1, device_mesh, [DTensorShard(0)])
        dtensor_2 = DTensor.from_local(local_tensor_2, device_mesh, [DTensorShard(1)])

        return {
            "model": {
                "weight": dtensor_1,
                "bias": dtensor_2,
            },
            "optimizer": {
                "lr": 0.001,
            },
            "step": 100,
            "nested": {
                "param": dtensor_1,
                "scalar": 42,
            },
        }

    @with_comms
    @pytest.mark.gpus_needed_2
    def test_extract_item_metadata_basic(self):
        """Test basic item metadata extraction functionality."""
        # Create a simple device mesh for single rank
        device_mesh = DeviceMesh(self.device_type, [0])

        # Create DTensor
        local_tensor = torch.randn(4, 8, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [DTensorShard(0)])

        # Create state dict with various object types
        state_dict = {
            "dtensor": dtensor,
            "custom": CustomObject("test_data"),
            "custom_tensor": CustomTensorObject(torch.randn(2, 2)),
            "regular_tensor": torch.randn(3, 3),
            "scalar": 42,
            "string": "test",
            "nested": {"dtensor": dtensor, "list": [1, 2, 3]},
        }

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        manager = DefaultMetadataManager(
            self._create_rank_info(),
        )
        item_to_metadata = manager.extract_object_metadata(checkpoint_info)

        # Verify we got per-item metadata dictionary
        self.assertIsInstance(item_to_metadata, dict)
        self.assertGreater(len(item_to_metadata), 0)

        # Verify each item's nested_path_to_metadata structure
        for item_key, nested_path_to_metadata in item_to_metadata.items():
            self.assertIsInstance(item_key, str)
            self.assertIsInstance(nested_path_to_metadata, dict)
            for nested_path, metadata in nested_path_to_metadata.items():
                self.assertIsInstance(nested_path, tuple)
                self.assertIsInstance(metadata, ShardingMetadata)

        # Verify DTensor metadata (item_key="dtensor", nested_path=())
        self.assertIn("dtensor", item_to_metadata)
        dtensor_nested = item_to_metadata["dtensor"]
        self.assertIn((), dtensor_nested)
        dtensor_meta = dtensor_nested[()]
        self.assertIsInstance(dtensor_meta, DTensorShardingMetadata)

        # Verify custom metadata (item_key="custom", nested_path=())
        self.assertIn("custom", item_to_metadata)
        custom_nested = item_to_metadata["custom"]
        self.assertIn((), custom_nested)
        custom_meta = custom_nested[()]
        self.assertIsInstance(custom_meta, CustomShardingMetadata)

        # Verify custom tensor metadata (uses DTensorShardingMetadata via protocol)
        self.assertIn("custom_tensor", item_to_metadata)
        custom_tensor_nested = item_to_metadata["custom_tensor"]
        self.assertIn((), custom_tensor_nested)
        custom_tensor_meta = custom_tensor_nested[()]
        self.assertIsInstance(custom_tensor_meta, DTensorShardingMetadata)

        # regular_tensor, scalar, string should NOT be in metadata (no sharding info)
        self.assertNotIn("regular_tensor", item_to_metadata)
        self.assertNotIn("scalar", item_to_metadata)
        self.assertNotIn("string", item_to_metadata)

    @with_comms
    @pytest.mark.gpus_needed_2
    def test_extract_item_metadata_dtensor_properties(self):
        """Test that DTensor metadata is extracted correctly."""
        device_mesh = DeviceMesh(self.device_type, [0])
        local_tensor = torch.randn(4, 8, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [DTensorShard(0)])

        state_dict = {"dtensor": dtensor}

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        manager = DefaultMetadataManager(
            self._create_rank_info(),
        )
        item_to_metadata = manager.extract_object_metadata(checkpoint_info)

        # Verify DTensor metadata (item_key="dtensor", nested_path=())
        self.assertIn("dtensor", item_to_metadata)
        dtensor_nested = item_to_metadata["dtensor"]
        self.assertIn((), dtensor_nested)
        dtensor_meta = dtensor_nested[()]
        self.assertIsInstance(dtensor_meta, DTensorShardingMetadata)

        self.assertEqual(dtensor_meta.global_shape, (4, 8))
        self.assertEqual(dtensor_meta.dtype, str(dtensor.dtype))
        self.assertEqual(dtensor_meta.stride, tuple(dtensor.stride()))
        self.assertEqual(len(dtensor_meta.placements), 1)

    @with_comms
    @pytest.mark.gpus_needed_2
    def test_extract_item_metadata_nested_structures(self):
        """Test extraction from deeply nested structures."""
        device_mesh = DeviceMesh(self.device_type, [0])
        local_tensor = torch.randn(2, 2, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [DTensorReplicate()])

        state_dict = {
            "level1": {
                "level2": {
                    "dtensor": dtensor,
                    "list": [dtensor, {"nested_dtensor": dtensor}],
                    "tuple": (1, 2, dtensor),
                }
            }
        }

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        manager = DefaultMetadataManager(
            self._create_rank_info(),
        )
        item_to_metadata = manager.extract_object_metadata(checkpoint_info)

        # Verify DTensor paths are extracted correctly
        # Nested dtensor path: item_key="level1", nested_path=("level2", "list", 1, "nested_dtensor")
        self.assertIn("level1", item_to_metadata)
        level1_nested = item_to_metadata["level1"]
        nested_path = ("level2", "list", 1, "nested_dtensor")
        self.assertIn(nested_path, level1_nested)
        self.assertIsInstance(level1_nested[nested_path], DTensorShardingMetadata)

    def test_custom_metadata_protocol(self):
        """Test objects implementing __pt_sharding_metadata__ protocol."""
        custom_obj = CustomObject({"test": "data"}, "test_type")
        custom_tensor_obj = CustomTensorObject(torch.randn(3, 4))

        state_dict = {
            "custom": custom_obj,
            "custom_tensor": custom_tensor_obj,
        }

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        manager = DefaultMetadataManager(
            self._create_rank_info(),
        )
        item_to_metadata = manager.extract_object_metadata(checkpoint_info)

        # Verify custom metadata object (item_key="custom", nested_path=())
        self.assertIn("custom", item_to_metadata)
        custom_nested = item_to_metadata["custom"]
        self.assertIn((), custom_nested)
        custom_meta = custom_nested[()]
        self.assertIsInstance(custom_meta, CustomShardingMetadata)
        self.assertIsNotNone(custom_meta.custom_info)
        self.assertEqual(custom_meta.custom_type, "test_type")

        # Verify custom tensor metadata object (item_key="custom_tensor", nested_path=())
        self.assertIn("custom_tensor", item_to_metadata)
        custom_tensor_nested = item_to_metadata["custom_tensor"]
        self.assertIn((), custom_tensor_nested)
        custom_tensor_meta = custom_tensor_nested[()]
        self.assertIsInstance(custom_tensor_meta, DTensorShardingMetadata)
        self.assertEqual(custom_tensor_meta.global_shape, (3, 4))

    @with_comms
    @pytest.mark.gpus_needed_2
    def test_single_rank_compute_metadata(self):
        """Test complete metadata computation for single rank with layout info."""
        device_mesh = DeviceMesh(self.device_type, [0])
        state_dict = self._create_simple_state_dict(device_mesh)

        # Add layout info (common path)
        layout_info_mappings = {
            "model": LayoutInfo(
                file_path="model.pt",
                serialization_format=TorchSerialization(),
            ),
            "nested": LayoutInfo(
                file_path="nested.pt",
                serialization_format=TorchSerialization(),
            ),
            "optimizer": LayoutInfo(
                file_path="config.json",
                serialization_format=JsonSerialization(cls=dict),
            ),
        }

        # Create CheckpointInfo from state_dict with layout info and appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(
            state_dict, layout_info_mappings
        )

        rank_info = RankInfo(
            global_rank=0, global_world_size=1, role_rank=0, role_world_size=1
        )

        manager = DefaultMetadataManager(rank_info)

        metadata = manager.compute_metadata(checkpoint_info)

        # Verify metadata structure
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata.distributed_metadata, DistributedMetadata)
        self.assertIsInstance(metadata.distributed_metadata.metadata, dict)
        self.assertGreater(len(metadata.distributed_metadata.metadata), 0)

        # Verify layout info is present per-item (v2.0 format)
        # Layout info is now in each DistributedItemMetadata
        model_item = metadata.distributed_metadata.metadata.get("model")
        self.assertIsNotNone(model_item)
        self.assertIn(0, model_item.rank_to_layout_info)
        self.assertEqual(model_item.rank_to_layout_info[0].file_path, "model.pt")
        self.assertIsInstance(
            model_item.rank_to_layout_info[0].serialization_format, TorchSerialization
        )

        # Verify DTensors are found in grouped format
        # metadata.distributed_metadata.metadata is dict[str, DistributedItemMetadata]
        # Each DistributedItemMetadata has nested_path_to_metadata: dict[NestedPath, list[GlobalObjectMetadata]]
        dtensor_items = [
            ("model", ("weight",)),
            ("model", ("bias",)),
            ("nested", ("param",)),
        ]

        for item_key, nested_path in dtensor_items:
            # Item should exist in metadata
            self.assertIn(item_key, metadata.distributed_metadata.metadata)
            item_metadata = metadata.distributed_metadata.metadata[item_key]
            # Nested path should exist
            self.assertIn(nested_path, item_metadata.nested_path_to_metadata)
            path_groups = item_metadata.nested_path_to_metadata[nested_path]
            # For single rank, should have one GlobalObjectMetadata entry
            self.assertEqual(len(path_groups), 1)
            group = path_groups[0]
            # The sharding_metadata should be DTensorShardingMetadata
            self.assertIsInstance(group.sharding_metadata, DTensorShardingMetadata)
            # For single rank, ranks should be (0,)
            self.assertEqual(group.ranks, (0,))

    @with_comms
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @pytest.mark.gpus_needed_2
    def test_multi_rank_compute_metadata(self):
        """Test metadata computation across multiple ranks with layout info."""
        world_size = dist.get_world_size()
        current_rank = dist.get_rank()
        device_mesh = DeviceMesh(self.device_type, list(range(world_size)))
        state_dict = self._create_simple_state_dict(device_mesh)

        # Add layout info (common path)
        layout_info_mappings = {
            "model": LayoutInfo(
                file_path=f"model_rank_{current_rank}.pt",
                serialization_format=TorchSerialization(),
            ),
        }

        # Create CheckpointInfo from state_dict with layout info and appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(
            state_dict, layout_info_mappings
        )

        rank_info = self._create_rank_info()

        manager = DefaultMetadataManager(rank_info)

        metadata = manager.compute_metadata(checkpoint_info)

        # Verify metadata is aggregated correctly
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata.distributed_metadata, DistributedMetadata)
        self.assertGreater(len(metadata.distributed_metadata.metadata), 0)

        # Verify layout info from all ranks is aggregated per-item (v2.0 format)
        model_item = metadata.distributed_metadata.metadata.get("model")
        self.assertIsNotNone(model_item)
        rank_to_layout = model_item.rank_to_layout_info
        self.assertEqual(len(rank_to_layout), world_size)

        # Verify each rank's layout info
        for rank in range(world_size):
            self.assertIn(rank, rank_to_layout)
            rank_layout = rank_to_layout[rank]
            self.assertIsNotNone(rank_layout)
            self.assertEqual(rank_layout.file_path, f"model_rank_{rank}.pt")

    @with_comms
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @pytest.mark.gpus_needed_2
    def test_empty_state_dict(self):
        """Test handling of empty state dictionary."""
        # Create CheckpointInfo from empty state_dict
        checkpoint_info = CheckpointInfo({})

        rank_info = RankInfo(
            global_rank=0, global_world_size=1, role_rank=0, role_world_size=1
        )

        manager = DefaultMetadataManager(rank_info)

        metadata = manager.compute_metadata(checkpoint_info)

        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata.distributed_metadata, DistributedMetadata)
        self.assertEqual(len(metadata.distributed_metadata.metadata), 0)

    def test_unknown_objects_only_state_dict(self):
        """Test state dict with only unknown objects (no DTensor or custom metadata).

        Objects without sharding metadata are NOT stored
        in the metadata. This test verifies that behavior.
        """
        state_dict = {
            "scalar": 42,
            "string": "test",
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tensor": torch.randn(4, 4, device=self.device_type),
        }

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        rank_info = RankInfo(
            global_rank=0, global_world_size=1, role_rank=0, role_world_size=1
        )

        manager = DefaultMetadataManager(rank_info)

        metadata = manager.compute_metadata(checkpoint_info)

        # With no sharding metadata to extract, the metadata dict should be empty
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata.distributed_metadata, DistributedMetadata)
        # Objects without sharding metadata are not stored
        self.assertEqual(len(metadata.distributed_metadata.metadata), 0)

    @with_comms
    @pytest.mark.gpus_needed_2
    def test_mixed_object_types(self):
        """Test state dict with mixed object types."""
        device_mesh = DeviceMesh(self.device_type, [0])
        local_tensor = torch.randn(2, 2, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [DTensorReplicate()])

        state_dict = {
            "dtensor": dtensor,
            "custom": CustomObject("test"),
            "unknown": torch.randn(3, 3),
            "scalar": 42,
        }

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        manager = DefaultMetadataManager(
            self._create_rank_info(),
        )
        item_to_metadata = manager.extract_object_metadata(checkpoint_info)

        # Verify DTensor and custom types are extracted (item_key -> nested_path -> metadata)
        self.assertIn("dtensor", item_to_metadata)
        self.assertIn((), item_to_metadata["dtensor"])
        self.assertIsInstance(item_to_metadata["dtensor"][()], DTensorShardingMetadata)

        self.assertIn("custom", item_to_metadata)
        self.assertIn((), item_to_metadata["custom"])
        self.assertIsInstance(item_to_metadata["custom"][()], CustomShardingMetadata)

        # Unknown objects (regular tensor, scalar) are NOT stored
        self.assertNotIn("unknown", item_to_metadata)
        self.assertNotIn("scalar", item_to_metadata)

    def test_nested_path_construction(self):
        """Test that nested paths are constructed correctly."""
        # Use objects that have sharding metadata so they appear in the result
        custom_objects = {
            "simple": CustomObject("simple"),
            "nested": {
                "param": CustomObject("param"),
                "deep": {"value": CustomObject("value")},
            },
            "list": [CustomObject("0"), CustomObject("1")],
            "mixed": {"list": [{"item": CustomObject("item")}]},
        }

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(custom_objects)

        manager = DefaultMetadataManager(
            self._create_rank_info(),
        )
        item_to_metadata = manager.extract_object_metadata(checkpoint_info)

        # Verify all custom objects have their metadata extracted
        # item_to_metadata is dict[str, dict[NestedPath, ShardingMetadata]]
        for item_key, nested_path_to_metadata in item_to_metadata.items():
            self.assertIsInstance(item_key, str)
            for nested_path, sharding_meta in nested_path_to_metadata.items():
                self.assertIsInstance(nested_path, tuple)
                self.assertIsInstance(sharding_meta, ShardingMetadata)

    def test_local_metadata_cache_invalidated(self):
        """Test that local metadata cache is invalidated when state dict changes."""
        state_dict = {
            "custom": CustomObject("test_data"),
            "custom_tensor": CustomTensorObject(torch.ones(2, 2)),
            "regular_tensor": torch.ones(3, 3),
            "scalar": 42,
            "string": "test",
        }

        manager = DefaultMetadataManager(
            self._create_rank_info(),
        )

        # Create CheckpointInfo from state_dict with appropriate resharders
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        # First call should populate cache
        manager.compute_metadata(checkpoint_info)
        self.assertTrue(manager._cached_local_metadata is not None)

        # Subsequent call without metadata changes should not raise error
        manager.compute_metadata(checkpoint_info)
        state_dict["scalar"] = 1
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)
        manager.compute_metadata(checkpoint_info)

        # Change state dict's metadata by changing tensor shape
        state_dict["custom_tensor"] = CustomTensorObject(torch.ones(4, 2))
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        with self.assertRaises(RuntimeError) as context:
            manager.compute_metadata(checkpoint_info)
        self.assertIn(
            "State dictionary has changed since last checkpoint.",
            str(context.exception),
        )

    @with_comms
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @pytest.mark.gpus_needed_2
    def test_compact_dtensor_metadata_representative_rank(self):
        """Test that _compact only keeps DTensor entries for representative rank."""
        world_size = dist.get_world_size()
        current_rank = dist.get_rank()
        device_mesh = DeviceMesh(self.device_type, list(range(world_size)))

        # Create DTensor
        local_tensor = torch.randn(4, 8, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [DTensorShard(0)])

        state_dict = {"model": {"weight": dtensor}}
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        rank_info = self._create_rank_info()
        manager = DefaultMetadataManager(rank_info)

        # Extract local metadata (grouped by item_key)
        local_metadata = manager.extract_object_metadata(checkpoint_info)

        # Compact the metadata
        compacted = manager._compact(local_metadata)

        # Only rank 0 (smallest in mesh) should have DTensor entries
        if current_rank == 0:
            # Representative rank keeps DTensor entries
            self.assertEqual(len(compacted), len(local_metadata))
        else:
            # Non-representative ranks should have DTensor entries removed
            # So compacted should be empty or have no DTensorShardingMetadata
            for _item_key, nested_path_to_meta in compacted.items():
                for _nested_path, sharding_meta in nested_path_to_meta.items():
                    # DTensorShardingMetadata entries should be removed for non-representative ranks
                    self.assertNotIsInstance(sharding_meta, DTensorShardingMetadata)

    def test_compact_preserves_non_dtensor_metadata(self):
        """Test that _compact preserves non-DTensor metadata for all ranks."""
        state_dict = {
            "custom": CustomObject("test_data"),
            "scalar": 42,
            "tensor": torch.randn(3, 3),
        }
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        # Test for various ranks
        for test_rank in [0, 1, 5]:
            rank_info = RankInfo(
                global_rank=test_rank,
                global_world_size=8,
                role_rank=test_rank,
                role_world_size=8,
            )
            manager = DefaultMetadataManager(rank_info)

            local_metadata = manager.extract_object_metadata(checkpoint_info)
            compacted = manager._compact(local_metadata)

            # All non-DTensor entries should be preserved
            self.assertEqual(len(compacted), len(local_metadata))
            for item_key in local_metadata:
                self.assertIn(item_key, compacted)

    @with_comms
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @pytest.mark.gpus_needed_2
    def test_merge_dtensor_metadata_uses_equivalent_ranks(self):
        """Test that merged DTensor metadata has ranks from equivalent_ranks."""
        world_size = dist.get_world_size()
        device_mesh = DeviceMesh(self.device_type, list(range(world_size)))

        local_tensor = torch.randn(4, 8, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [DTensorShard(0)])

        state_dict = {"model": {"weight": dtensor}}
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        rank_info = self._create_rank_info()
        manager = DefaultMetadataManager(rank_info)

        metadata = manager.compute_metadata(checkpoint_info)

        # Verify DTensor metadata has ranks from equivalent_ranks (mesh_data)
        # Grouped format: metadata is dict[str, DistributedItemMetadata]
        self.assertIn("model", metadata.distributed_metadata.metadata)
        model_item = metadata.distributed_metadata.metadata["model"]
        self.assertIn(("weight",), model_item.nested_path_to_metadata)
        object_metadata_groups = model_item.nested_path_to_metadata[("weight",)]
        self.assertEqual(len(object_metadata_groups), 1)

        object_group = object_metadata_groups[0]
        # Check sharding_metadata is DTensorShardingMetadata
        self.assertIsInstance(object_group.sharding_metadata, DTensorShardingMetadata)
        # ranks should contain all ranks from the mesh (from equivalent_ranks)
        self.assertEqual(object_group.ranks, tuple(range(world_size)))

    @with_comms
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @pytest.mark.gpus_needed_2
    def test_multi_rank_metadata_aggregation_with_compaction(self):
        """Test end-to-end metadata aggregation verifying compaction works correctly."""
        world_size = dist.get_world_size()
        device_mesh = DeviceMesh(self.device_type, list(range(world_size)))

        # Create mixed state dict with DTensor and custom objects
        local_tensor = torch.randn(4, 8, device=self.device_type)
        dtensor = DTensor.from_local(local_tensor, device_mesh, [DTensorShard(0)])

        state_dict = {
            "model": {"weight": dtensor},
            "custom": CustomObject("test"),  # Custom object with sharding metadata
        }
        checkpoint_info = make_checkpoint_info_from_dict(state_dict)

        rank_info = self._create_rank_info()
        manager = DefaultMetadataManager(rank_info)

        metadata = manager.compute_metadata(checkpoint_info)

        # Verify all items are present in aggregated metadata
        self.assertIsNotNone(metadata)
        distributed_meta = metadata.distributed_metadata

        # DTensor path "model::weight" should exist with ranks from equivalent_ranks
        self.assertIn("model", distributed_meta.metadata)
        model_item = distributed_meta.metadata["model"]
        self.assertIn(("weight",), model_item.nested_path_to_metadata)
        model_groups = model_item.nested_path_to_metadata[("weight",)]
        self.assertEqual(len(model_groups), 1)
        self.assertEqual(model_groups[0].ranks, tuple(range(world_size)))

        # Custom object "custom" should exist with proper ranks tracking
        self.assertIn("custom", distributed_meta.metadata)
        custom_item = distributed_meta.metadata["custom"]
        self.assertIn((), custom_item.nested_path_to_metadata)
        custom_groups = custom_item.nested_path_to_metadata[()]
        # Custom objects (without equivalent_ranks) should have ranks from aggregation
        self.assertEqual(custom_groups[0].ranks, tuple(range(world_size)))
