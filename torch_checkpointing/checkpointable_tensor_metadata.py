# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding metadata for local tensors with ``CheckpointableTensor`` annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    _StridedShard as DTensorStridedShard,
    Replicate as DTensorReplicate,
    Shard as DTensorShard,
)

from ._dtensor_utils import compute_local_shape_and_chunks
from .checkpointable import CheckpointableTensor
from .distributed_metadata import ShardingMetadata


@dataclass(frozen=True)
class CheckpointableTensorShardingMetadata(
    ShardingMetadata, type_name="checkpointable_tensor_v1"
):
    """Metadata representing a local tensor with ``CheckpointableTensor`` annotations.

    The annotations place one or more shards of the local tensor within a logical
    global tensor; see ``CheckpointableTensor`` for the meaning of each field.
    """

    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]
    dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_shape", tuple(self.global_shape))
        object.__setattr__(
            self,
            "global_offsets",
            tuple(tuple(offset) for offset in self.global_offsets),
        )
        object.__setattr__(
            self,
            "local_offsets",
            tuple(tuple(offset) for offset in self.local_offsets),
        )
        object.__setattr__(
            self,
            "local_sizes",
            tuple(tuple(size) for size in self.local_sizes),
        )
        self._validate_geometry()

    def _validate_geometry(self) -> None:
        shard_counts = {
            len(self.global_offsets),
            len(self.local_offsets),
            len(self.local_sizes),
        }
        if len(shard_counts) != 1:
            raise ValueError(
                "global_offsets, local_offsets, and local_sizes must describe "
                "the same number of shards"
            )

        ndim = len(self.global_shape)
        if any(size < 0 for size in self.global_shape):
            raise ValueError("global_shape values must be nonnegative")

        for shard_index, (global_offset, local_offset, local_size) in enumerate(
            zip(self.global_offsets, self.local_offsets, self.local_sizes)
        ):
            self._validate_shard_dimensions(
                shard_index, ndim, global_offset, local_offset, local_size
            )
            self._validate_shard_values(
                shard_index, global_offset, local_offset, local_size
            )

    def _validate_shard_dimensions(
        self,
        shard_index: int,
        ndim: int,
        global_offset: tuple[int, ...],
        local_offset: tuple[int, ...],
        local_size: tuple[int, ...],
    ) -> None:
        for field_name, coordinates in (
            ("global_offsets", global_offset),
            ("local_offsets", local_offset),
            ("local_sizes", local_size),
        ):
            if len(coordinates) != ndim:
                raise ValueError(
                    f"{field_name}[{shard_index}] must have {ndim} dimensions, "
                    f"got {len(coordinates)}"
                )

    def _validate_shard_values(
        self,
        shard_index: int,
        global_offset: tuple[int, ...],
        local_offset: tuple[int, ...],
        local_size: tuple[int, ...],
    ) -> None:
        if any(offset < 0 for offset in global_offset):
            raise ValueError(
                f"global_offsets[{shard_index}] values must be nonnegative"
            )
        if any(offset < 0 for offset in local_offset):
            raise ValueError(f"local_offsets[{shard_index}] values must be nonnegative")
        if any(size < 0 for size in local_size):
            raise ValueError(f"local_sizes[{shard_index}] values must be nonnegative")
        if any(
            offset + size > global_size
            for offset, size, global_size in zip(
                global_offset, local_size, self.global_shape
            )
        ):
            raise ValueError(
                f"Shard {shard_index} extends beyond the logical global tensor"
            )

    def _validate_local_bounds(self, local_shape: tuple[int, ...]) -> None:
        if len(local_shape) != len(self.global_shape):
            raise ValueError(
                "The local tensor and global_shape must have the same number of "
                "dimensions"
            )
        for shard_index, (local_offset, local_size) in enumerate(
            zip(self.local_offsets, self.local_sizes)
        ):
            if any(
                offset + size > tensor_size
                for offset, size, tensor_size in zip(
                    local_offset, local_size, local_shape
                )
            ):
                raise ValueError(f"Shard {shard_index} extends beyond the local tensor")

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor
    ) -> "CheckpointableTensorShardingMetadata":
        """Create metadata from a tensor implementing ``CheckpointableTensor``."""
        if not isinstance(tensor, torch.Tensor) or not isinstance(
            tensor, CheckpointableTensor
        ):
            raise TypeError(
                "CheckpointableTensorShardingMetadata.from_tensor requires a torch.Tensor "
                "implementing CheckpointableTensor"
            )

        metadata = cls(
            global_shape=tensor.global_shape,
            global_offsets=tensor.global_offsets,
            local_offsets=tensor.local_offsets,
            local_sizes=tensor.local_sizes,
            dtype=str(tensor.dtype),
        )
        metadata._validate_local_bounds(tuple(tensor.shape))
        return metadata

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "CheckpointableTensorShardingMetadata":
        """Derive metadata for this rank's local tensor of ``dtensor``.

        The placements may give the local tensor several shards: a
        ``_StridedShard`` placement can leave it holding blocks that are not
        adjacent in the global tensor. Each shard is a non-empty block of the
        local tensor whose global indices are consecutive along every dimension,
        and the shards together cover the local tensor. A rank outside the device
        mesh holds no shards.
        """
        for placement in dtensor.placements:
            if not isinstance(
                placement, (DTensorShard, DTensorStridedShard, DTensorReplicate)
            ):
                raise ValueError(
                    f"Cannot checkpoint a DTensor with placement {placement}: only "
                    "Shard, StridedShard and Replicate placements are supported"
                )
        mesh = dtensor.device_mesh
        local_shape, global_offsets, local_offsets, local_sizes = (
            compute_local_shape_and_chunks(dtensor.shape, mesh, dtensor.placements)
        )
        actual_shape = tuple(dtensor._local_tensor.shape)
        if mesh.get_coordinate() is not None and local_shape != actual_shape:
            raise ValueError(
                f"The placements {tuple(dtensor.placements)} of a DTensor of "
                f"shape {tuple(dtensor.shape)} give this rank a local tensor "
                f"of shape {local_shape}, but it has shape {actual_shape}"
            )
        # Upstream keeps empty chunks so that every rank gets the same number of
        # chunks, which keeps it traceable. Checkpointing needs no tracing, and
        # an empty chunk can share its global offset with a non-empty one, so
        # list only the non-empty chunks. Untraced, every value is an int.
        chunks = [
            tuple(tuple(map(int, field)) for field in chunk)
            for chunk in zip(global_offsets, local_offsets, local_sizes)
            if 0 not in chunk[2]
        ]
        return cls(
            global_shape=tuple(dtensor.shape),
            global_offsets=tuple(chunk[0] for chunk in chunks),
            local_offsets=tuple(chunk[1] for chunk in chunks),
            local_sizes=tuple(chunk[2] for chunk in chunks),
            dtype=str(dtensor.dtype),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the metadata to a JSON-compatible dictionary."""
        return {
            "global_shape": list(self.global_shape),
            "global_offsets": [list(offset) for offset in self.global_offsets],
            "local_offsets": [list(offset) for offset in self.local_offsets],
            "local_sizes": [list(size) for size in self.local_sizes],
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckpointableTensorShardingMetadata":
        """Create metadata from a serialized dictionary."""
        return cls(
            global_shape=d["global_shape"],
            global_offsets=d["global_offsets"],
            local_offsets=d["local_offsets"],
            local_sizes=d["local_sizes"],
            dtype=d["dtype"],
        )

    @property
    def equivalent_ranks(self) -> tuple[int, ...] | None:
        """Annotations are rank-specific, so this metadata cannot be compacted by rank."""
        return None
