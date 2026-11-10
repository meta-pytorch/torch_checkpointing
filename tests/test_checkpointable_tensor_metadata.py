# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Sequence

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch_checkpointing.checkpointable_tensor_metadata import (
    CheckpointableTensorShardingMetadata,
)
from torch_checkpointing.distributed_metadata import GlobalObjectMetadata

from .resharding_test_utils import dtensor_local_shard, fake_process_group


def test_checkpointable_tensor_sharding_metadata_serialization_round_trip() -> None:
    metadata = CheckpointableTensorShardingMetadata(
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

    assert serialized["sharding_metadata"]["type"] == "checkpointable_tensor_v1"
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
def test_checkpointable_tensor_sharding_metadata_rejects_invalid_geometry(
    global_shape: tuple[int, ...],
    global_offsets: tuple[tuple[int, ...], ...],
    local_offsets: tuple[tuple[int, ...], ...],
    local_sizes: tuple[tuple[int, ...], ...],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        CheckpointableTensorShardingMetadata(
            global_shape=global_shape,
            global_offsets=global_offsets,
            local_offsets=local_offsets,
            local_sizes=local_sizes,
            dtype="torch.float32",
        )


def test_checkpointable_tensor_sharding_metadata_allows_scalar_and_empty_shards() -> (
    None
):
    scalar = CheckpointableTensorShardingMetadata(
        global_shape=(),
        global_offsets=((),),
        local_offsets=((),),
        local_sizes=((),),
        dtype="torch.float32",
    )
    empty = CheckpointableTensorShardingMetadata(
        global_shape=(0, 3),
        global_offsets=((0, 0),),
        local_offsets=((2, 1),),
        local_sizes=((0, 2),),
        dtype="torch.float32",
    )

    assert scalar.local_sizes == ((),)
    assert empty.local_sizes == ((0, 2),)


def _dtensor_at(
    rank: int,
    global_tensor: torch.Tensor,
    mesh_shape: tuple[int, ...],
    placements: Sequence[Placement],
) -> DTensor:
    """The DTensor that ``rank`` holds; call inside its fake process group."""
    mesh = init_device_mesh("cpu", mesh_shape)
    coordinate = mesh.get_coordinate()
    assert coordinate is not None
    local = dtensor_local_shard(global_tensor, mesh_shape, coordinate, placements)
    return DTensor.from_local(
        local.clone(),
        mesh,
        placements,
        run_check=False,
        shape=global_tensor.shape,
        stride=global_tensor.stride(),
    )


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize(
    ("global_shape", "mesh_shape", "placements"),
    [
        pytest.param((8, 6), (4,), [Shard(0)], id="shard"),
        pytest.param((5,), (4,), [Shard(0)], id="uneven-shard-with-empty-chunk"),
        pytest.param((7, 5), (2, 2), [Shard(0), Shard(1)], id="2d-uneven"),
        pytest.param((9, 2), (2, 2), [Shard(0), Shard(0)], id="same-dim-twice"),
        pytest.param((4, 4), (2, 2), [Replicate(), Shard(1)], id="replicate-shard"),
        pytest.param(
            (8, 2),
            (2, 2),
            [_StridedShard(0, split_factor=2), Shard(0)],
            id="strided-then-shard",
        ),
        pytest.param(
            (8, 2),
            (2, 2),
            [Shard(0), _StridedShard(0, split_factor=2)],
            id="shard-then-strided",
        ),
        pytest.param(
            (3, 11),
            (2, 2),
            [_StridedShard(1, split_factor=2), Shard(1)],
            id="uneven-strided",
        ),
        pytest.param(
            (13,),
            (2, 2),
            [Shard(0), _StridedShard(0, split_factor=3)],
            id="strided-split-factor-3",
        ),
        pytest.param((), (4,), [Replicate()], id="scalar"),
        pytest.param((0, 4), (4,), [Shard(0)], id="empty"),
    ],
)
def test_from_dtensor_shards_tile_the_local_tensor(
    rank: int,
    global_shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    placements: list[Placement],
) -> None:
    global_tensor = torch.arange(math.prod(global_shape)).reshape(global_shape)
    with fake_process_group(rank, world_size=4):
        dtensor = _dtensor_at(rank, global_tensor, mesh_shape, placements)
        metadata = CheckpointableTensorShardingMetadata.from_dtensor(dtensor)

    local = dtensor._local_tensor
    assert metadata.global_shape == global_shape
    assert metadata.dtype == str(global_tensor.dtype)
    assert all(0 not in sizes for sizes in metadata.local_sizes)
    covered = torch.zeros(local.shape, dtype=torch.int64)
    for global_offset, local_offset, sizes in zip(
        metadata.global_offsets, metadata.local_offsets, metadata.local_sizes
    ):
        local_block = tuple(
            slice(offset, offset + size) for offset, size in zip(local_offset, sizes)
        )
        global_block = tuple(
            slice(offset, offset + size) for offset, size in zip(global_offset, sizes)
        )
        torch.testing.assert_close(local[local_block], global_tensor[global_block])
        covered[local_block] += 1
    assert bool((covered == 1).all())


@pytest.mark.parametrize(
    ("placements", "rank", "expected"),
    [
        pytest.param(
            # compute_local_shape_and_chunks also returns an empty chunk at
            # global offset (2, 0), which the metadata leaves out.
            [_StridedShard(0, split_factor=2), Shard(0)],
            1,
            CheckpointableTensorShardingMetadata(
                global_shape=(8, 2),
                global_offsets=((4, 0),),
                local_offsets=((0, 0),),
                local_sizes=((2, 2),),
                dtype="torch.int64",
            ),
            id="right-to-left-shards-drop-empty-chunk",
        ),
        pytest.param(
            [Shard(0), _StridedShard(0, split_factor=2)],
            1,
            CheckpointableTensorShardingMetadata(
                global_shape=(8, 2),
                global_offsets=((1, 0), (3, 0)),
                local_offsets=((0, 0), (1, 0)),
                local_sizes=((1, 2), (1, 2)),
                dtype="torch.int64",
            ),
            id="interleaved-rows-become-two-shards",
        ),
    ],
)
def test_from_dtensor_describes_strided_shards(
    placements: list[Placement],
    rank: int,
    expected: CheckpointableTensorShardingMetadata,
) -> None:
    global_tensor = torch.arange(16).reshape(8, 2)
    with fake_process_group(rank, world_size=4):
        dtensor = _dtensor_at(rank, global_tensor, (2, 2), placements)

        assert CheckpointableTensorShardingMetadata.from_dtensor(dtensor) == expected


def test_from_dtensor_rejects_partial_placements() -> None:
    with fake_process_group(0, world_size=4):
        mesh = init_device_mesh("cpu", (4,))
        dtensor = DTensor.from_local(torch.zeros(4), mesh, [Partial()], run_check=False)

        with pytest.raises(ValueError, match="only Shard, StridedShard and Replicate"):
            CheckpointableTensorShardingMetadata.from_dtensor(dtensor)


def test_from_dtensor_rejects_a_local_tensor_its_placements_do_not_describe() -> None:
    with fake_process_group(0, world_size=4):
        mesh = init_device_mesh("cpu", (4,))
        dtensor = DTensor.from_local(
            torch.zeros(3, 2),
            mesh,
            [Shard(0)],
            run_check=False,
            shape=torch.Size((8, 2)),
            stride=(2, 1),
        )

        with pytest.raises(ValueError, match=r"local tensor of shape \(2, 2\)"):
            CheckpointableTensorShardingMetadata.from_dtensor(dtensor)
