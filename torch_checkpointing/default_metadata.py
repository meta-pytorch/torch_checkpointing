# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding metadata for tensors that expose checkpoint shard geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .checkpointable import CheckpointableTensor
from .distributed_metadata import ShardingMetadata


@dataclass(frozen=True)
class DefaultShardingMetadata(ShardingMetadata, type_name="default_tensor_v1"):
    """Shard geometry for a local tensor within a logical global tensor."""

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
    def from_tensor(cls, tensor: torch.Tensor) -> "DefaultShardingMetadata":
        """Create metadata from a tensor implementing ``CheckpointableTensor``."""
        if not isinstance(tensor, torch.Tensor) or not isinstance(
            tensor, CheckpointableTensor
        ):
            raise TypeError(
                "DefaultShardingMetadata.from_tensor requires a torch.Tensor "
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
    def from_dict(cls, d: dict[str, Any]) -> "DefaultShardingMetadata":
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
        """Default metadata is rank-specific and cannot be compacted by rank."""
        return None
