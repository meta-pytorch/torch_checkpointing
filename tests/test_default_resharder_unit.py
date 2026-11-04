# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch_checkpointing.checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
    TorchSerialization,
)
from torch_checkpointing.default_resharder import (
    _world_mesh_spec,
    compute_local_shard_info,
    DefaultResharder,
)
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    GlobalObjectMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
)
from torch_checkpointing.resharding import LoadPlan
from torch_checkpointing.safetensors_metadata import SafetensorsFileMetadata
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import NestedPath

from .resharding_test_utils import plan_and_execute


def test_import_does_not_require_safetensors() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.modules['safetensors'] = None; "
            "sys.modules['safetensors.torch'] = None; "
            "import torch_checkpointing.default_load_plan_executor; "
            "import torch_checkpointing.default_resharder",
        ],
        check=True,
    )


def test_extract_sharding_metadata_treats_plain_tensors_as_replicated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_item = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "scalar": torch.tensor(7, dtype=torch.int64),
        "epoch": 4,
    }

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.default_resharder.dist.get_world_size",
            return_value=4,
        ),
        caplog.at_level(logging.WARNING),
    ):
        metadata = DefaultResharder().extract_sharding_metadata(
            "model",
            checkpoint_item,
        )

    assert set(metadata) == {("weight",), ("scalar",)}
    weight_metadata = metadata[("weight",)]
    assert isinstance(weight_metadata, DTensorShardingMetadata)
    assert weight_metadata.global_shape == (2, 3)
    assert weight_metadata.dtype == "torch.float32"
    assert weight_metadata.stride == (3, 1)
    assert weight_metadata.mesh_spec.device_type == "cpu"
    assert weight_metadata.mesh_spec.mesh_shape == (4,)
    assert weight_metadata.mesh_spec.mesh_data == (0, 1, 2, 3)
    assert weight_metadata.placements == (ReplicateSpec(),)
    assert weight_metadata.equivalent_ranks == (0, 1, 2, 3)

    scalar_metadata = metadata[("scalar",)]
    assert isinstance(scalar_metadata, DTensorShardingMetadata)
    assert scalar_metadata.global_shape == ()
    assert scalar_metadata.stride == ()
    assert scalar_metadata.dtype == "torch.int64"
    assert scalar_metadata.placements == (ReplicateSpec(),)

    assert "Found 2 plain tensors" in caplog.text
    assert "treating them as replicated tensors" in caplog.text


def test_extract_sharding_metadata_builds_the_replicated_mesh_once() -> None:
    _world_mesh_spec.cache_clear()
    checkpoint_item = {f"tensor_{i}": torch.zeros(2) for i in range(3)}

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.default_resharder.dist.get_world_size",
            return_value=4,
        ),
        patch(
            "torch_checkpointing.default_resharder.get_device_mesh_spec",
            wraps=get_device_mesh_spec,
        ) as build_mesh_spec,
    ):
        metadata = DefaultResharder().extract_sharding_metadata(
            "model", checkpoint_item
        )

    assert build_mesh_spec.call_count == 1
    mesh_specs = []
    for tensor_metadata in metadata.values():
        assert isinstance(tensor_metadata, DTensorShardingMetadata)
        mesh_specs.append(tensor_metadata.mesh_spec)
    assert all(spec is mesh_specs[0] for spec in mesh_specs)
    assert mesh_specs[0].mesh_data == (0, 1, 2, 3)


@pytest.fixture
def mesh_tensor_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if a mesh tensor is rebuilt, which costs O(mesh size) per call."""

    def rebuild(self: DeviceMeshSpec) -> torch.Tensor:
        raise AssertionError("mesh tensor rebuilt")

    monkeypatch.setattr(DeviceMeshSpec, "mesh", property(rebuild))


@pytest.mark.parametrize(
    ("rank", "global_offset"),
    [(0, (0, 0)), (2, (0, 4)), (1, (4, 0)), (3, (4, 4))],
)
def test_compute_local_shard_info_resolves_rank_without_mesh_tensor(
    mesh_tensor_forbidden: None, rank: int, global_offset: tuple[int, ...]
) -> None:
    metadata = DTensorShardingMetadata(
        global_shape=(8, 8),
        dtype="torch.float32",
        stride=(8, 1),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu", mesh_shape=(2, 2), mesh_data=(0, 2, 1, 3)
        ),
        placements=(ShardSpec(0), ShardSpec(1)),
    )

    assert compute_local_shard_info(metadata, rank) == ((4, 4), global_offset)


def test_generate_load_plans_resolves_source_ranks_without_mesh_tensor(
    mesh_tensor_forbidden: None,
) -> None:
    mesh_spec = DeviceMeshSpec(
        device_type="cpu", mesh_shape=(4,), mesh_data=(0, 1, 2, 3)
    )

    def metadata(placement: ShardSpec | ReplicateSpec) -> DTensorShardingMetadata:
        return DTensorShardingMetadata(
            global_shape=(8, 8),
            dtype="torch.float32",
            stride=(8, 1),
            mesh_spec=mesh_spec,
            placements=(placement,),
        )

    source = DistributedItemMetadata(
        nested_path_to_metadata={
            ("weight",): [
                GlobalObjectMetadata(
                    sharding_metadata=metadata(ShardSpec(0)), ranks=(0, 1, 2, 3)
                )
            ]
        },
        rank_to_layout_info=dict.fromkeys(range(4)),
    )

    info = DefaultResharder().generate_load_plans(
        "model",
        {"weight": torch.zeros(8, 8)},
        {("weight",): metadata(ReplicateSpec())},
        source,
    )

    assert info.non_reshardable_paths == []
    plans = sorted(
        info.nested_path_to_load_plans[("weight",)], key=lambda p: p.src_rank
    )
    assert plans == [
        LoadPlan(
            offsets=(2 * rank, 0),
            sizes=(2, 8),
            src_rank=rank,
            src_fqn="weight",
            src_offsets=(0, 0),
            src_sizes=(2, 8),
        )
        for rank in range(4)
    ]


def _replicated(shape: tuple[int, ...]) -> DTensorShardingMetadata:
    return DTensorShardingMetadata(
        global_shape=shape,
        dtype="torch.float32",
        stride=torch.empty(shape).stride(),
        mesh_spec=DeviceMeshSpec(device_type="cpu", mesh_shape=(1,), mesh_data=(0,)),
        placements=(ReplicateSpec(),),
    )


def test_generate_load_plans_binds_planned_paths_to_their_target_tensors() -> None:
    weight = torch.zeros(8, 8)
    target = {"weight": weight, "unsaved": torch.zeros(2), "step": 3}
    source = DistributedItemMetadata(
        nested_path_to_metadata={
            ("weight",): [
                GlobalObjectMetadata(sharding_metadata=_replicated((8, 8)), ranks=(0,))
            ]
        },
        rank_to_layout_info={0: None},
    )

    info = DefaultResharder().generate_load_plans(
        "model",
        target,
        {("weight",): _replicated((8, 8)), ("unsaved",): _replicated((2,))},
        source,
    )

    assert info.nested_path_to_load_plans.keys() == {("weight",)}
    assert info.non_reshardable_paths == [("unsaved",)]
    assert info.nested_path_to_target_tensors[("weight",)] is weight


@dataclass
class _ColumnSlice:
    """A target leaf that loads into a range of columns of a wider tensor."""

    backing: torch.Tensor
    columns: slice


class _ColumnSliceResharder(DefaultResharder):
    def resolve_targets(
        self, item_key: str, target: Any
    ) -> dict[NestedPath, torch.Tensor]:
        return {(name,): leaf.backing[:, leaf.columns] for name, leaf in target.items()}


def test_load_writes_through_a_non_contiguous_target_view(tmp_path: Path) -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    torch.save({"weight": source}, tmp_path / "checkpoint.pt")
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    backing = torch.full((4, 4), -1.0)
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            ("weight",): [
                GlobalObjectMetadata(sharding_metadata=_replicated((4, 2)), ranks=(0,))
            ]
        },
        rank_to_layout_info={0: LayoutInfo("checkpoint.pt", TorchSerialization())},
    )

    missing_paths = plan_and_execute(
        _ColumnSliceResharder(),
        source_path=tmp_path,
        item_key="model",
        target_metadata={("weight",): _replicated((4, 2))},
        source_metadata=source_metadata,
        target={"weight": _ColumnSlice(backing, slice(1, 3))},
        storage=storage,
    )

    assert missing_paths == []
    torch.testing.assert_close(backing[:, 1:3], source)
    torch.testing.assert_close(backing[:, [0, 3]], torch.full((4, 2), -1.0))


def test_load_fills_a_planned_target_that_an_unplanned_path_also_binds(
    tmp_path: Path,
) -> None:
    source = torch.arange(4, dtype=torch.float32)
    torch.save({"weight": source}, tmp_path / "checkpoint.pt")
    tied = torch.zeros(4)
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            ("weight",): [
                GlobalObjectMetadata(sharding_metadata=_replicated((4,)), ranks=(0,))
            ]
        },
        rank_to_layout_info={0: LayoutInfo("checkpoint.pt", TorchSerialization())},
    )

    # "unsaved" binds the same tensor after "weight", but has no plans.
    missing_paths = plan_and_execute(
        DefaultResharder(),
        source_path=tmp_path,
        item_key="model",
        target_metadata={
            ("weight",): _replicated((4,)),
            ("unsaved",): _replicated((4,)),
        },
        source_metadata=source_metadata,
        target={"weight": tied, "unsaved": tied},
        storage=LocalFileSystemStorageConfig(use_direct_io=False).create_storage(),
    )

    assert missing_paths == [("unsaved",)]
    torch.testing.assert_close(tied, source)


def test_load_reshards_safetensors_shards_described_by_native_metadata(
    tmp_path: Path,
) -> None:
    """An unconsolidated checkpoint: safetensors shards, native metadata, no HF."""
    whole = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    for rank in range(2):
        save_file(
            {"weight": whole[rank * 3 : (rank + 1) * 3].contiguous()},
            tmp_path / f"model_{rank}.safetensors",
        )

    def sharded(mesh_shape: tuple[int, ...], mesh_data: tuple[int, ...], placements):
        return DTensorShardingMetadata(
            global_shape=(6, 4),
            dtype="torch.float32",
            stride=(4, 1),
            mesh_spec=DeviceMeshSpec(
                device_type="cpu", mesh_shape=mesh_shape, mesh_data=mesh_data
            ),
            placements=placements,
        )

    source_sharding = sharded((2,), (0, 1), (ShardSpec(0),))
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            ("weight",): [
                GlobalObjectMetadata(sharding_metadata=source_sharding, ranks=(0, 1))
            ]
        },
        rank_to_layout_info={
            rank: LayoutInfo(f"model_{rank}.safetensors", SafetensorsSerialization())
            for rank in range(2)
        },
    )
    target = {"weight": torch.zeros((6, 4), dtype=torch.float32)}
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch("torch_checkpointing.default_resharder.dist.get_rank", return_value=0),
        patch(
            "torch_checkpointing.default_resharder.dist.get_world_size", return_value=1
        ),
    ):
        missing = plan_and_execute(
            DefaultResharder(),
            source_path=tmp_path,
            item_key="model",
            target_metadata={("weight",): sharded((1,), (0,), (ReplicateSpec(),))},
            source_metadata=source_metadata,
            target=target,
            storage=storage,
        )

    assert missing == []
    torch.testing.assert_close(target["weight"], whole)


def test_safetensors_meta_tensors_name_the_file_for_a_missing_fqn() -> None:
    metadata = SafetensorsFileMetadata(file_path="model.safetensors", tensors={})

    with pytest.raises(
        KeyError,
        match="'missing' is not in the safetensors header of model.safetensors",
    ):
        metadata.as_meta_tensors(["missing"])
