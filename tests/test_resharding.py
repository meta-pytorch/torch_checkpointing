# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    ShardingMetadata,
)
from torch_checkpointing.resharding import LoadPlan, Resharder, ReshardingInfo
from torch_checkpointing.types import NestedPath


class _MinimalResharder(Resharder):
    """Implements only the abstract methods, to test the base class defaults."""

    def extract_sharding_metadata(
        self, item_key: str, item_value: Any
    ) -> dict[NestedPath, ShardingMetadata]:
        return {}

    def generate_load_plans(
        self,
        item_key: str,
        target: Any,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
    ) -> ReshardingInfo:
        return ReshardingInfo(nested_path_to_load_plans={}, non_reshardable_paths=[])


@pytest.fixture
def cpu_mesh() -> Iterator[DeviceMesh]:
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _plan() -> LoadPlan:
    return LoadPlan(
        offsets=(0,),
        sizes=(1,),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=(1,),
    )


def test_resharding_info_requires_a_target_tensor_for_each_planned_path() -> None:
    with pytest.raises(ValueError, match=r"no target tensor: \[\('weight',\)\]"):
        ReshardingInfo(
            nested_path_to_load_plans={("weight",): [_plan()]},
            non_reshardable_paths=[],
            nested_path_to_target_tensors={("bias",): torch.zeros(1)},
        )


def test_resharding_info_allows_target_tensors_for_unplanned_paths() -> None:
    weight = torch.zeros(1)

    info = ReshardingInfo(
        nested_path_to_load_plans={},
        non_reshardable_paths=[("weight",)],
        nested_path_to_target_tensors={("weight",): weight},
    )

    assert info.nested_path_to_target_tensors[("weight",)] is weight


def test_resolve_targets_returns_each_tensor_leaf_itself() -> None:
    weight = torch.zeros(2, 3)
    column_view = torch.zeros(4, 4)[:, 1:3]
    target = {"weight": weight, "layers": [{"view": column_view}], "step": 7}

    resolved = _MinimalResharder().resolve_targets("model", target)

    assert resolved.keys() == {("weight",), ("layers", 0, "view")}
    assert resolved[("weight",)] is weight
    assert resolved[("layers", 0, "view")] is column_view


def test_resolve_targets_replaces_a_dtensor_with_its_local_shard(
    cpu_mesh: DeviceMesh,
) -> None:
    dtensor = distribute_tensor(torch.arange(4.0), cpu_mesh, [Replicate()])

    resolved = _MinimalResharder().resolve_targets("model", {"weight": dtensor})

    assert resolved[("weight",)] is dtensor._local_tensor
