# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch_checkpointing.checkpoint_base import CheckpointInfo
from torch_checkpointing.checkpoint_layout import (
    LayoutInfo,
    TorchSerialization,
)
from torch_checkpointing.default_metadata import DefaultShardingMetadata
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
    StridedShardSpec,
)
from torch_checkpointing.metadata_manager import DefaultMetadataManager
from torch_checkpointing.types import CheckpointPath, NestedPath, RankInfo

from .resharding_test_utils import CustomTensorObject, make_checkpoint_item


def test_default_sharding_metadata_serialization_round_trip() -> None:
    metadata = DefaultShardingMetadata(
        global_shape=(8, 6),
        global_offsets=((0, 1), (4, 1)),
        local_offsets=((1, 0), (5, 0)),
        local_sizes=((4, 5), (4, 5)),
        dtype="torch.float32",
    )
    object_metadata = GlobalObjectMetadata(
        sharding_metadata=metadata,
        ranks=(3,),
    )

    serialized = object_metadata.to_dict()
    restored = GlobalObjectMetadata.from_dict(serialized)

    assert serialized["sharding_metadata"]["type"] == "default_tensor_v1"
    assert serialized["sharding_metadata"]["data"] == {
        "global_shape": [8, 6],
        "global_offsets": [[0, 1], [4, 1]],
        "local_offsets": [[1, 0], [5, 0]],
        "local_sizes": [[4, 5], [4, 5]],
        "dtype": "torch.float32",
    }
    assert restored == object_metadata
    assert restored.sharding_metadata.equivalent_ranks is None


@pytest.mark.parametrize(
    (
        "global_shape",
        "global_offsets",
        "local_offsets",
        "local_sizes",
        "error_match",
    ),
    [
        pytest.param(
            (8,),
            ((0,),),
            (),
            ((4,),),
            "same number of shards",
            id="mismatched-shard-count",
        ),
        pytest.param(
            (8, 4),
            ((0,),),
            ((0, 0),),
            ((4, 4),),
            "must have 2 dimensions",
            id="mismatched-dimensions",
        ),
        pytest.param(
            (8,),
            ((6,),),
            ((0,),),
            ((4,),),
            "extends beyond the logical global tensor",
            id="shard-exceeds-global-shape",
        ),
    ],
)
def test_default_sharding_metadata_rejects_invalid_geometry(
    global_shape: tuple[int, ...],
    global_offsets: tuple[tuple[int, ...], ...],
    local_offsets: tuple[tuple[int, ...], ...],
    local_sizes: tuple[tuple[int, ...], ...],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        DefaultShardingMetadata(
            global_shape=global_shape,
            global_offsets=global_offsets,
            local_offsets=local_offsets,
            local_sizes=local_sizes,
            dtype="torch.float32",
        )


def test_default_sharding_metadata_allows_scalar_and_empty_shards() -> None:
    scalar = DefaultShardingMetadata(
        global_shape=(),
        global_offsets=((),),
        local_offsets=((),),
        local_sizes=((),),
        dtype="torch.float32",
    )
    empty = DefaultShardingMetadata(
        global_shape=(0, 3),
        global_offsets=((0, 0),),
        local_offsets=((2, 1),),
        local_sizes=((0, 2),),
        dtype="torch.float32",
    )

    assert scalar.local_sizes == ((),)
    assert empty.local_sizes == ((0, 2),)


def test_missing_rank_layout_uses_default_torch_rank_path() -> None:
    metadata = DistributedItemMetadata(
        nested_path_to_metadata={},
        rank_to_layout_info={7: None},
    )

    assert metadata.get_layout_info(7, "model") == LayoutInfo(
        file_path="model_7.pt",
        serialization_format=TorchSerialization(),
    )


def test_to_and_from_dict():
    """Test that metadata is correctly converted to and from a dictionary."""
    state_dict = {
        "scalar": 42,
        "string": "test",
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "tensor": torch.ones(4, 4, device="cpu"),
        "custom_tensor": CustomTensorObject(torch.ones(2, 2)),
    }

    # Create CheckpointInfo from state_dict
    checkpoint_info = CheckpointInfo(
        {k: make_checkpoint_item(value=v) for k, v in state_dict.items()}
    )

    manager = DefaultMetadataManager(
        rank_info=RankInfo(
            global_rank=0, global_world_size=1, role_rank=0, role_world_size=1
        ),
    )
    manager.compute_distributed_metadata(checkpoint_info)
    md = manager.get_computed_metadata().distributed_metadata

    # Convert to dictionary
    metadata_dict = md.to_dict()

    # Top-level keys
    assert "metadata" in metadata_dict
    assert "world_size" in metadata_dict
    assert "version" in metadata_dict

    assert metadata_dict["world_size"] == 1
    assert metadata_dict["version"] == "2.0"

    # Validate metadata structure: dict with item_key keys
    metadata = metadata_dict["metadata"]
    assert isinstance(metadata, dict)
    assert len(metadata) == 1  # Only custom_tensor has sharding metadata

    # Validate custom_tensor has correct structure (v2.0 format)
    assert "custom_tensor" in metadata
    custom_tensor_item = metadata["custom_tensor"]
    assert "nested_path_to_metadata" in custom_tensor_item
    assert "rank_to_layout_info" in custom_tensor_item

    # Check nested_path_to_metadata has empty path for leaf value
    nested_path_to_metadata = custom_tensor_item["nested_path_to_metadata"]
    assert len(nested_path_to_metadata) == 1
    # Empty nested path serialized as "[]"
    path_key = "[]"
    assert path_key in nested_path_to_metadata
    groups = nested_path_to_metadata[path_key]
    assert len(groups) == 1

    group = groups[0]
    # Validate ranks field is present and equals (0,) for single rank
    assert "ranks" in group
    assert group["ranks"] == (0,)

    # Check sharding_metadata contains the metadata
    assert "sharding_metadata" in group
    sharding_meta = group["sharding_metadata"]
    placements = sharding_meta["data"]["placements"]
    assert len(placements) == 1
    assert placements[0]["type"] == "Replicate"

    # Convert back to metadata
    md2 = DistributedMetadata.from_dict(metadata_dict)

    # Check that the two metadata objects are equal
    assert md == md2


def test_dtensor_strided_shard_metadata_round_trip():
    metadata_dict = {
        "global_shape": [8, 4],
        "dtype": "torch.float32",
        "stride": [4, 1],
        "mesh_spec": {
            "device_type": "cpu",
            "mesh_shape": (2, 2),
            "mesh_data": (0, 1, 2, 3),
            "mesh_dim_names": ("dp", "tp"),
        },
        "placements": [
            {"type": "StridedShard", "dim": 0, "split_factor": 2},
            {"type": "Shard", "dim": 0},
        ],
    }

    metadata = DTensorShardingMetadata.from_dict(metadata_dict)

    assert isinstance(metadata.placements[0], StridedShardSpec)
    assert metadata.to_dict() == metadata_dict
    restored = DTensorShardingMetadata.from_dict(metadata.to_dict())
    assert metadata == restored
    assert hash(metadata) == hash(restored)

    different_split_factor = {
        **metadata_dict,
        "placements": [
            {"type": "StridedShard", "dim": 0, "split_factor": 4},
            {"type": "Shard", "dim": 0},
        ],
    }
    assert metadata != DTensorShardingMetadata.from_dict(different_split_factor)


def test_dtensor_metadata_rejects_unknown_placement_type():
    with pytest.raises(ValueError, match="Unknown DTensor placement type"):
        DTensorShardingMetadata.from_dict(
            {
                "global_shape": [8],
                "dtype": "torch.float32",
                "stride": [1],
                "mesh_spec": {
                    "device_type": "cpu",
                    "mesh_shape": (1,),
                    "mesh_data": (0,),
                    "mesh_dim_names": None,
                },
                "placements": [{"type": "Unknown"}],
            }
        )


def test_distributed_metadata_with_layout_info():
    """Test DistributedMetadata with layout info covering multiple ranks, different serialization formats, and None values."""
    # Create simple tensor sharding metadata
    mesh_spec = DeviceMeshSpec.from_mesh(
        device_type="cpu",
        mesh=torch.tensor([[0, 1]]),
        mesh_dim_names=("dp", "tp"),
    )
    tensor_metadata = DTensorShardingMetadata(
        global_shape=(
            8,
            16,
        ),
        dtype=str(torch.float32),
        stride=(
            16,
            1,
        ),
        mesh_spec=mesh_spec,
        placements=(ShardSpec(dim=0), ReplicateSpec()),
    )

    # Create layout info for the "model" item with multiple ranks
    model_rank_to_layout_info = {
        0: LayoutInfo(
            file_path="model_rank_0.pt",
            serialization_format=TorchSerialization(),
        ),
        1: LayoutInfo(
            file_path="model_rank_1.pt",
            serialization_format=TorchSerialization(),
        ),
    }

    # Create DistributedMetadata with layout info using v2.0 format
    nested_path: NestedPath = ("weight",)
    object_metadata_with_ranks = GlobalObjectMetadata(
        sharding_metadata=tensor_metadata,
        ranks=(0, 1),  # All ranks as tuple
    )
    model_item_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            nested_path: [object_metadata_with_ranks],
        },
        rank_to_layout_info=model_rank_to_layout_info,
    )
    metadata = DistributedMetadata(
        metadata={
            "model": model_item_metadata,
        },
        world_size=2,
    )

    # Serialize to dict
    metadata_dict = metadata.to_dict()

    # Verify v2.0 structure
    assert "metadata" in metadata_dict
    assert "model" in metadata_dict["metadata"]
    model_dict = metadata_dict["metadata"]["model"]
    assert "nested_path_to_metadata" in model_dict
    assert "rank_to_layout_info" in model_dict

    # Verify layout info is in the per-item structure
    layout_info_dict = model_dict["rank_to_layout_info"]
    assert 0 in layout_info_dict
    assert 1 in layout_info_dict
    assert layout_info_dict[0]["file_path"] == "model_rank_0.pt"
    assert layout_info_dict[1]["file_path"] == "model_rank_1.pt"

    # Deserialize back to DistributedMetadata
    metadata_restored = DistributedMetadata.from_dict(metadata_dict)

    # Verify deserialization
    assert metadata_restored.world_size == 2
    assert metadata_restored.version == "2.0"
    assert "model" in metadata_restored.metadata

    # Verify layout info is restored correctly
    restored_model_metadata = metadata_restored.metadata["model"]
    assert 0 in restored_model_metadata.rank_to_layout_info
    assert 1 in restored_model_metadata.rank_to_layout_info
    assert restored_model_metadata.rank_to_layout_info[0].file_path == "model_rank_0.pt"
    assert restored_model_metadata.rank_to_layout_info[1].file_path == "model_rank_1.pt"
    assert isinstance(
        restored_model_metadata.rank_to_layout_info[0].serialization_format,
        TorchSerialization,
    )

    # Verify equality works
    assert metadata == metadata_restored


def test_distributed_metadata_with_empty_layout_info():
    """Test DistributedMetadata serialization with rank present but no layout items."""
    mesh_spec = DeviceMeshSpec.from_mesh(
        device_type="cpu",
        mesh=torch.tensor([0]),
        mesh_dim_names=None,
    )
    tensor_metadata = DTensorShardingMetadata(
        global_shape=(
            4,
            8,
        ),
        dtype=str(torch.float32),
        stride=(
            8,
            1,
        ),
        mesh_spec=mesh_spec,
        placements=(ReplicateSpec(),),
    )

    # Create metadata with rank present but None layout
    nested_path: NestedPath = ("bias",)
    object_metadata_with_ranks = GlobalObjectMetadata(
        sharding_metadata=tensor_metadata,
        ranks=(0,),  # Single rank as tuple
    )
    model_item_metadata = DistributedItemMetadata(
        nested_path_to_metadata={nested_path: [object_metadata_with_ranks]},
        rank_to_layout_info={0: None},  # Rank 0 present but None layout
    )
    metadata = DistributedMetadata(
        metadata={"model": model_item_metadata},
        world_size=1,
    )

    # Serialize
    metadata_dict = metadata.to_dict()

    # Verify structure
    assert "metadata" in metadata_dict
    assert "model" in metadata_dict["metadata"]
    model_dict = metadata_dict["metadata"]["model"]
    assert model_dict["rank_to_layout_info"] == {0: None}

    # Deserialize
    metadata_restored = DistributedMetadata.from_dict(metadata_dict)

    # Verify layout info has rank 0 with None
    assert metadata_restored.metadata["model"].rank_to_layout_info == {0: None}
    assert metadata == metadata_restored


def test_checkpoint_path():
    """Test CheckpointPath construction and serialization."""
    # Test leaf path (no nested path)
    leaf_path = CheckpointPath("step", ())
    assert leaf_path.item_key == "step"
    assert leaf_path.nested_path == ()
    assert str(leaf_path) == "step"

    # Test nested path
    nested_path = CheckpointPath("model", ("encoder", "weight"))
    assert nested_path.item_key == "model"
    assert nested_path.nested_path == ("encoder", "weight")
    assert str(nested_path) == "model::encoder.weight"

    # Test serialization/deserialization
    path_str = nested_path.serialize()
    assert path_str == '["model","encoder","weight"]'

    restored_path = CheckpointPath.deserialize(path_str)
    assert restored_path == nested_path


def test_pickle_deduplication():
    """Test that pickle deduplicates DeviceMeshSpec references.

    This test verifies that when many DTensorShardingMetadata objects share
    the same DeviceMeshSpec (via get_device_mesh_spec caching), pickle's memo
    mechanism properly deduplicates them during serialization, resulting in
    much smaller serialized sizes than if each mesh were serialized separately.
    """
    import pickle

    # Create mesh spec (cached via get_device_mesh_spec)
    # 1 DTensor
    single = DTensorShardingMetadata(
        global_shape=(1024,),
        dtype="torch.float32",
        stride=(1,),
        mesh_spec=get_device_mesh_spec("cuda", (8,), tuple(range(8)), ("dp",)),
        placements=(ReplicateSpec(),),
    )

    # 1000 DTensors with same mesh (all reference the same cached mesh_spec)
    many = [
        DTensorShardingMetadata(
            global_shape=(1024,),
            dtype="torch.float32",
            stride=(1,),
            mesh_spec=get_device_mesh_spec("cuda", (8,), tuple(range(8)), ("dp",)),
            placements=(ReplicateSpec(),),
        )
        for _ in range(1000)
    ]

    single_size = len(pickle.dumps(single))
    many_size = len(pickle.dumps(many))

    # Size should not be 1000x due to pickle's memo deduplication
    # The shared mesh_spec should be serialized once and referenced thereafter
    # With deduplication working, we expect the ratio to be much less than 1000x
    # (without dedup it would be ~1000x, with dedup we expect <200x)
    assert many_size < single_size * 200, (
        f"Deduplication failed: 1000 items is {many_size / single_size:.1f}x "
        f"the size of 1 item (expected <200x due to pickle memo)"
    )


def test_distributed_metadata_rejects_duplicate_file_paths():
    """Test that DistributedMetadata raises error when two ranks write to the same file."""
    mesh_spec = DeviceMeshSpec.from_mesh(
        device_type="cpu",
        mesh=torch.tensor([0, 1]),
        mesh_dim_names=("dp",),
    )
    tensor_metadata = DTensorShardingMetadata(
        global_shape=(8,),
        dtype=str(torch.float32),
        stride=(1,),
        mesh_spec=mesh_spec,
        placements=(ReplicateSpec(),),
    )

    # Create layout info where two ranks write to the same file
    nested_path: NestedPath = ("weight",)
    object_metadata = GlobalObjectMetadata(
        sharding_metadata=tensor_metadata,
        ranks=(0, 1),
    )
    model_item_metadata = DistributedItemMetadata(
        nested_path_to_metadata={nested_path: [object_metadata]},
        rank_to_layout_info={
            0: LayoutInfo(
                file_path="shared_file.pt",  # Same file as rank 1
                serialization_format=TorchSerialization(),
            ),
            1: LayoutInfo(
                file_path="shared_file.pt",  # Same file as rank 0 - should fail
                serialization_format=TorchSerialization(),
            ),
        },
    )

    # Validation should reject this
    with pytest.raises(RuntimeError, match="both write to the same file path"):
        DistributedMetadata(
            metadata={"model": model_item_metadata},
            world_size=2,
        )


def test_distributed_metadata_rejects_missing_ranks():
    """Test that DistributedMetadata raises error when ranks are missing."""
    mesh_spec = DeviceMeshSpec.from_mesh(
        device_type="cpu",
        mesh=torch.tensor([0, 1]),
        mesh_dim_names=("dp",),
    )
    tensor_metadata = DTensorShardingMetadata(
        global_shape=(8,),
        dtype=str(torch.float32),
        stride=(1,),
        mesh_spec=mesh_spec,
        placements=(ReplicateSpec(),),
    )

    nested_path: NestedPath = ("weight",)
    object_metadata = GlobalObjectMetadata(
        sharding_metadata=tensor_metadata,
        ranks=(0, 1),
    )
    model_item_metadata = DistributedItemMetadata(
        nested_path_to_metadata={nested_path: [object_metadata]},
        rank_to_layout_info={0: None},  # Missing rank 1
    )

    # world_size=2 but only rank 0 present - should fail
    with pytest.raises(RuntimeError, match="Missing ranks"):
        DistributedMetadata(
            metadata={"model": model_item_metadata},
            world_size=2,
        )


def test_v1_to_v2_conversion():
    """Test that v1.0 format checkpoints are correctly converted to v2.0 format."""
    # Construct a v1.0 format dictionary manually
    # v1.0 format has:
    # - "metadata": dict with CheckpointPath serialized keys
    # - "rank_to_layout_info_mappings": dict[rank, dict[item_key, LayoutInfo]]
    # - "world_size": int
    # - "version": "1.0"

    v1_dict = {
        "metadata": {
            # CheckpointPath("model", ("weight",)).serialize() = '["model","weight"]'
            '["model","weight"]': [
                {
                    "sharding_metadata": {
                        "type": "DTensorShardingMetadata",
                        "data": {
                            "global_shape": (8, 16),
                            "dtype": "torch.float32",
                            "stride": (16, 1),
                            "mesh_spec": {
                                "device_type": "cpu",
                                "mesh_shape": (2,),
                                "mesh_data": (0, 1),
                                "mesh_dim_names": ("dp",),
                            },
                            "placements": [{"type": "Shard", "dim": 0}],
                        },
                    },
                    "ranks": (0, 1),
                }
            ],
            # CheckpointPath("model", ("bias",)).serialize() = '["model","bias"]'
            '["model","bias"]': [
                {
                    "sharding_metadata": {
                        "type": "DTensorShardingMetadata",
                        "data": {
                            "global_shape": (16,),
                            "dtype": "torch.float32",
                            "stride": (1,),
                            "mesh_spec": {
                                "device_type": "cpu",
                                "mesh_shape": (2,),
                                "mesh_data": (0, 1),
                                "mesh_dim_names": ("dp",),
                            },
                            "placements": [{"type": "Replicate"}],
                        },
                    },
                    "ranks": (0, 1),
                }
            ],
            # CheckpointPath("optimizer", ("state", 0, "exp_avg")).serialize()
            '["optimizer","state",0,"exp_avg"]': [
                {
                    "sharding_metadata": {
                        "type": "DTensorShardingMetadata",
                        "data": {
                            "global_shape": (8, 16),
                            "dtype": "torch.float32",
                            "stride": (16, 1),
                            "mesh_spec": {
                                "device_type": "cpu",
                                "mesh_shape": (2,),
                                "mesh_data": (0, 1),
                                "mesh_dim_names": ("dp",),
                            },
                            "placements": [{"type": "Shard", "dim": 0}],
                        },
                    },
                    "ranks": (0, 1),
                }
            ],
        },
        "rank_to_layout_info_mappings": {
            0: {
                "model": {
                    "file_path": "model_rank_0.pt",
                    "serialization_format": {"type": "TorchSerialization"},
                },
                "optimizer": {
                    "file_path": "optimizer_rank_0.pt",
                    "serialization_format": {"type": "TorchSerialization"},
                },
            },
            1: {
                "model": {
                    "file_path": "model_rank_1.pt",
                    "serialization_format": {"type": "TorchSerialization"},
                },
                "optimizer": {
                    "file_path": "optimizer_rank_1.pt",
                    "serialization_format": {"type": "TorchSerialization"},
                },
            },
        },
        "world_size": 2,
        "version": "1.0",
    }

    # Load from v1.0 format
    metadata = DistributedMetadata.from_dict(v1_dict)

    # Verify it was converted to v2.0 format
    assert metadata.version == "2.0"
    assert metadata.world_size == 2

    # Verify items are grouped correctly
    assert "model" in metadata.metadata
    assert "optimizer" in metadata.metadata

    # Verify model item has both weight and bias nested paths
    model_item = metadata.metadata["model"]
    assert ("weight",) in model_item.nested_path_to_metadata
    assert ("bias",) in model_item.nested_path_to_metadata

    # Verify optimizer item has the nested path
    optimizer_item = metadata.metadata["optimizer"]
    assert ("state", 0, "exp_avg") in optimizer_item.nested_path_to_metadata

    # Verify layout info was correctly associated with items
    assert 0 in model_item.rank_to_layout_info
    assert 1 in model_item.rank_to_layout_info
    assert model_item.rank_to_layout_info[0].file_path == "model_rank_0.pt"
    assert model_item.rank_to_layout_info[1].file_path == "model_rank_1.pt"

    assert 0 in optimizer_item.rank_to_layout_info
    assert 1 in optimizer_item.rank_to_layout_info
    assert optimizer_item.rank_to_layout_info[0].file_path == "optimizer_rank_0.pt"
    assert optimizer_item.rank_to_layout_info[1].file_path == "optimizer_rank_1.pt"

    # Verify sharding metadata was preserved
    weight_groups = model_item.nested_path_to_metadata[("weight",)]
    assert len(weight_groups) == 1
    assert weight_groups[0].ranks == (0, 1)
    assert weight_groups[0].sharding_metadata.global_shape == (8, 16)


def _coordinate_by_mesh_search(
    spec: DeviceMeshSpec, rank: int
) -> tuple[int, ...] | None:
    indices = torch.where(spec.mesh == rank)
    if indices[0].numel() == 0:
        return None
    return tuple(int(index[0]) for index in indices)


@pytest.mark.parametrize(
    ("mesh_shape", "mesh_data"),
    [
        ((4,), (0, 1, 2, 3)),
        ((2, 2), (0, 2, 1, 3)),
        ((2, 3), (6, 7, 8, 9, 10, 11)),
        ((2, 2, 2), (7, 6, 5, 4, 3, 2, 1, 0)),
    ],
)
def test_get_coordinate_matches_a_search_of_the_mesh(
    mesh_shape: tuple[int, ...], mesh_data: tuple[int, ...]
) -> None:
    spec = DeviceMeshSpec(device_type="cpu", mesh_shape=mesh_shape, mesh_data=mesh_data)
    for rank in (*mesh_data, max(mesh_data) + 1):
        assert spec.get_coordinate(rank) == _coordinate_by_mesh_search(spec, rank)


@pytest.mark.parametrize(
    ("mesh_shape", "mesh_data", "mesh_dim_names", "match"),
    [
        ((2, 2), (0, 1, 2), None, "needs 4"),
        ((2, 2), (0, 1, 2, 3, 4), None, "needs 4"),
        ((2, 2), (0, 1, 0, 2), None, "duplicate ranks"),
        ((2, 2), (0, 1, 2, 3), ("dp",), "do not match"),
    ],
)
def test_device_mesh_spec_rejects_invalid_meshes(
    mesh_shape: tuple[int, ...],
    mesh_data: tuple[int, ...],
    mesh_dim_names: tuple[str, ...] | None,
    match: str,
) -> None:
    fields = {
        "device_type": "cpu",
        "mesh_shape": mesh_shape,
        "mesh_data": mesh_data,
        "mesh_dim_names": mesh_dim_names,
    }
    with pytest.raises(ValueError, match=match):
        DeviceMeshSpec(**fields)
    with pytest.raises(ValueError, match=match):
        DeviceMeshSpec.from_dict(fields)


@pytest.fixture
def fake_process_group() -> Iterator[None]:
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_from_dtensor_builds_one_mesh_spec_per_device_mesh(
    fake_process_group: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    expected = DeviceMeshSpec.from_mesh(
        device_type="cpu", mesh=mesh.mesh, mesh_dim_names=("dp", "tp")
    )
    from_mesh = DeviceMeshSpec.from_mesh
    calls: list[str] = []

    def counting_from_mesh(
        device_type: str,
        mesh: torch.Tensor,
        mesh_dim_names: tuple[str, ...] | None = None,
    ) -> DeviceMeshSpec:
        calls.append("from_mesh")
        return from_mesh(device_type, mesh, mesh_dim_names)

    monkeypatch.setattr(DeviceMeshSpec, "from_mesh", staticmethod(counting_from_mesh))
    specs = [
        DTensorShardingMetadata.from_dtensor(
            DTensor.from_local(
                torch.zeros(2, 2), mesh, [Shard(0), Shard(1)], run_check=False
            )
        ).mesh_spec
        for _ in range(3)
    ]

    assert len(calls) == 1
    assert all(spec is specs[0] for spec in specs)
    assert specs[0] == expected


def test_from_dtensor_keeps_specs_of_different_meshes_apart(
    fake_process_group: None,
) -> None:
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    full = DTensorShardingMetadata.from_dtensor(
        DTensor.from_local(
            torch.zeros(2, 2), mesh, [Shard(0), Shard(1)], run_check=False
        )
    ).mesh_spec
    sub = DTensorShardingMetadata.from_dtensor(
        DTensor.from_local(torch.zeros(2), mesh["tp"], [Shard(0)], run_check=False)
    ).mesh_spec

    assert (full.mesh_shape, full.mesh_data) == ((2, 2), (0, 1, 2, 3))
    assert (sub.mesh_shape, sub.mesh_data, sub.mesh_dim_names) == (
        (2,),
        (0, 1),
        ("tp",),
    )
