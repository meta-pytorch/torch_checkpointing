# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Protocol for local tensors that describe their logical global shards."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class CheckpointableTensor(Protocol):
    """Shard geometry exposed by a local tensor for checkpointing.

    Attributes:
        global_shape: Shape of the logical global tensor.
        global_offsets: Global start coordinate of each local shard.
        local_offsets: Start coordinate of each shard in the local tensor.
        local_sizes: Shape of each local shard.
    """

    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]
