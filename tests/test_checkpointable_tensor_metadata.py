# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch_checkpointing.checkpointable_tensor_metadata import (
    CheckpointableTensorShardingMetadata,
)
from torch_checkpointing.distributed_metadata import GlobalObjectMetadata


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
