# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DTensor-specific metadata for distributed checkpointing.

This module provides dataclasses for describing the distributed nature of
DTensor objects being checkpointed. These are implementations of the abstract
ShardingMetadata interface defined in distributed_metadata.py.

The key components are:
- _PlacementSpec: Base class for DTensor placement specifications
- ShardSpec: Shard placement specification (shards a tensor dimension)
- StridedShardSpec: Right-to-left shard placement specification
- ReplicateSpec: Replicate placement specification (replicates across a mesh dimension)
- DeviceMeshSpec: Lightweight, immutable device mesh specification
- DTensorShardingMetadata: Complete DTensor distribution information
"""

import itertools
import math
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    _StridedShard as DTensorStridedShard,
    Replicate as DTensorReplicate,
    Shard as DTensorShard,
)

from .distributed_metadata import ShardingMetadata


class _PlacementSpec(ABC):
    """
    Base class for placement specifications. Describes how a Tensor is placed onto
    a DeviceMesh dimension. This is a local version of torch.distributed.tensor.Placement
    that allows for custom placement types specific to checkpointing.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the placement specification to a dictionary for serialization.

        Returns:
            A dictionary representation of the placement specification.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "_PlacementSpec":
        """
        Create a PlacementSpec object from a dictionary representation.

        Returns:
            A PlacementSpec object constructed from the dictionary.
        """
        pass


@dataclass(frozen=True)
class ShardSpec(_PlacementSpec):
    """
    Shard placement specification. Describes DTensor sharding on a tensor dimension
    over a corresponding DeviceMesh dimension.

    Args:
        dim (int): The tensor dimension to shard over.
    """

    dim: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardSpec):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        return hash(self.dim)

    def __repr__(self) -> str:
        return f"ShardSpec(dim={self.dim})"

    def __str__(self) -> str:
        return f"Shard({self.dim})"

    def to_dict(self) -> dict[str, Any]:
        # Keep "Shard" for backward compatibility with existing checkpoints
        return {"type": "Shard", "dim": self.dim}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_PlacementSpec":
        return ShardSpec(dim=d["dim"])


@dataclass(frozen=True)
class StridedShardSpec(_PlacementSpec):
    """Right-to-left DTensor sharding on a tensor dimension.

    ``split_factor`` records how many same-dimension shards are logically
    applied after this placement.
    """

    dim: int
    split_factor: int

    def __repr__(self) -> str:
        return f"StridedShardSpec(dim={self.dim}, split_factor={self.split_factor})"

    def __str__(self) -> str:
        return f"StridedShard({self.dim}, split_factor={self.split_factor})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "StridedShard",
            "dim": self.dim,
            "split_factor": self.split_factor,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_PlacementSpec":
        return StridedShardSpec(
            dim=d["dim"],
            split_factor=d["split_factor"],
        )


@dataclass(frozen=True)
class ReplicateSpec(_PlacementSpec):
    """
    Replicate placement specification. Describes DTensor replication on a
    corresponding DeviceMesh dimension.
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReplicateSpec)

    def __hash__(self) -> int:
        return -1  # All replicate placements are the same

    def __repr__(self) -> str:
        return "ReplicateSpec()"

    def __str__(self) -> str:
        return "Replicate"

    def to_dict(self) -> dict[str, Any]:
        # Keep "Replicate" for backward compatibility with existing checkpoints
        return {"type": "Replicate"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_PlacementSpec":
        return ReplicateSpec()


@dataclass(frozen=True)
class DeviceMeshSpec:
    """
    Lightweight, immutable device mesh specification for distributed tensors.

    Stores mesh topology as immutable Python tuples for O(1) hashing.
    Tensor representation is reconstructed lazily when needed.

    Note: Meshes can be non-contiguous (e.g., [0, 2] from slicing a 2D mesh),
    so we must store the actual rank IDs, not just the shape.

    Args:
        device_type: Type of device ("cuda", "cpu", etc.)
        mesh_shape: Shape of the mesh as tuple, e.g., (2, 4) for 8 GPUs
        mesh_data: Flattened mesh global rank IDs as tuple, e.g., (0, 2, 1, 3)
        mesh_dim_names: Optional names for mesh dimensions, e.g., ("dp", "tp")
    """

    device_type: str
    mesh_shape: tuple[int, ...]
    mesh_data: tuple[int, ...]
    mesh_dim_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.mesh_data) != math.prod(self.mesh_shape):
            raise ValueError(
                f"mesh_data has {len(self.mesh_data)} ranks but mesh_shape "
                f"{self.mesh_shape} needs {math.prod(self.mesh_shape)}"
            )
        if len(set(self.mesh_data)) != len(self.mesh_data):
            raise ValueError(f"mesh_data contains duplicate ranks: {self.mesh_data}")
        if self.mesh_dim_names is not None and len(self.mesh_dim_names) != len(
            self.mesh_shape
        ):
            raise ValueError(
                f"mesh_dim_names {self.mesh_dim_names} do not match mesh_shape "
                f"{self.mesh_shape}"
            )

    @cached_property
    def _hash(self) -> int:
        return hash(
            (self.device_type, self.mesh_shape, self.mesh_data, self.mesh_dim_names)
        )

    def __hash__(self) -> int:
        # Memoized: resharding looks each rank's coordinate up by the spec once
        # per path and source rank, and hashing mesh_data costs O(mesh size).
        # __reduce__ keeps the memo out of pickles.
        return self._hash

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle the fields only, and unpickle to the interned spec.

        A pickled ``_hash`` memo would be stale in a process whose string
        hashes differ, and the cached ``_dict`` is derived. Unpickling through
        ``get_device_mesh_spec`` shares one spec per mesh, as building one does.
        """
        return (
            get_device_mesh_spec,
            (self.device_type, self.mesh_shape, self.mesh_data, self.mesh_dim_names),
        )

    @classmethod
    def from_mesh(
        cls,
        device_type: str,
        mesh: torch.Tensor,
        mesh_dim_names: tuple[str, ...] | None = None,
    ) -> "DeviceMeshSpec":
        """Create DeviceMeshSpec from a mesh tensor (with caching for pickle optimization)."""
        return get_device_mesh_spec(
            device_type=device_type,
            mesh_shape=tuple(mesh.shape),
            mesh_data=tuple(mesh.flatten().tolist()),
            mesh_dim_names=mesh_dim_names,
        )

    @classmethod
    def from_device_mesh(cls, device_mesh: DeviceMesh) -> "DeviceMeshSpec":
        """Create DeviceMeshSpec from a DeviceMesh, building it once per mesh.

        Building a spec reads every rank in the mesh, so DTensors that share a
        DeviceMesh share one spec instead of each paying O(mesh size).
        """
        spec = _DEVICE_MESH_SPECS.get(device_mesh)
        if spec is None:
            spec = cls.from_mesh(
                device_type=device_mesh.device_type,
                mesh=device_mesh.mesh,
                mesh_dim_names=device_mesh.mesh_dim_names,
            )
            _DEVICE_MESH_SPECS[device_mesh] = spec
        return spec

    @property
    def mesh(self) -> torch.Tensor:
        """Reconstruct mesh tensor."""
        return torch.tensor(self.mesh_data, dtype=torch.int64).reshape(self.mesh_shape)

    @cached_property
    def _dict(self) -> dict[str, Any]:
        """Cached dictionary representation for pickle deduplication."""
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the device mesh specification to a dictionary for serialization.

        Returns the same cached dict instance to enable pickle's memo mechanism
        to deduplicate references during serialization.

        Returns:
            A dictionary representation of the device mesh specification.
        """
        return self._dict

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeviceMeshSpec":
        """
        Create a DeviceMeshSpec object from a dictionary representation.

        Returns:
            A DeviceMeshSpec object constructed from the dictionary.
        """
        return get_device_mesh_spec(**d)

    def get_coordinate(self, rank: int) -> tuple[int, ...] | None:
        """
        Get the coordinate of a rank in this mesh.

        Looks the rank up in a rank-to-coordinate map built once per spec, so
        resolving every rank of a mesh costs O(mesh size) in total rather than
        per rank. Works correctly for non-contiguous meshes like [0, 2, 1, 3].

        Args:
            rank: The rank to find the coordinate for.

        Returns:
            A tuple of indices representing the coordinate of the rank in the mesh,
            or None if the rank is not found in the mesh.

        Example:
            For a mesh tensor([[0, 1], [2, 3]]) and rank=2,
            returns (1, 0) since rank 2 is at row 1, column 0.
        """
        return _rank_coordinates(self).get(rank)


def get_device_mesh_spec(
    device_type: str,
    mesh_shape: tuple[int, ...],
    mesh_data: tuple[int, ...],
    mesh_dim_names: tuple[str, ...] | None = None,
) -> DeviceMeshSpec:
    """
    Get or create a DeviceMeshSpec with reference deduplication.

    Returns the same instance for identical parameters, however they are
    passed. This enables pickle's memo mechanism to automatically deduplicate
    DeviceMeshSpec references during serialization.

    Args:
        device_type: Type of device ("cuda", "cpu", etc.)
        mesh_shape: Shape of the mesh as tuple, e.g., (2, 4) for 8 GPUs
        mesh_data: Flattened mesh global rank IDs as tuple, e.g., (0, 2, 1, 3)
        mesh_dim_names: Optional names for mesh dimensions, e.g., ("dp", "tp")

    Returns:
        A cached DeviceMeshSpec instance for the given parameters.
    """
    # Positional, so that keyword and positional calls share cache entries.
    return _interned_device_mesh_spec(
        device_type, mesh_shape, mesh_data, mesh_dim_names
    )


@lru_cache(maxsize=None)
def _interned_device_mesh_spec(
    device_type: str,
    mesh_shape: tuple[int, ...],
    mesh_data: tuple[int, ...],
    mesh_dim_names: tuple[str, ...] | None,
) -> DeviceMeshSpec:
    """The one DeviceMeshSpec for these parameters (unbounded cache)."""
    return DeviceMeshSpec(
        device_type=device_type,
        mesh_shape=mesh_shape,
        mesh_data=mesh_data,
        mesh_dim_names=mesh_dim_names,
    )


# Kept outside the specs so they are never pickled along with them, and keyed
# weakly so a mesh or spec that is no longer used can be freed.
_DEVICE_MESH_SPECS: "weakref.WeakKeyDictionary[DeviceMesh, DeviceMeshSpec]" = (
    weakref.WeakKeyDictionary()
)
_MESH_COORDINATES: "weakref.WeakKeyDictionary[DeviceMeshSpec, dict[int, tuple[int, ...]]]" = weakref.WeakKeyDictionary()


def _rank_coordinates(spec: DeviceMeshSpec) -> Mapping[int, tuple[int, ...]]:
    """Map each rank in ``spec`` to its coordinate, computed once per spec."""
    coordinates = _MESH_COORDINATES.get(spec)
    if coordinates is None:
        positions = itertools.product(*(range(size) for size in spec.mesh_shape))
        coordinates = dict(zip(spec.mesh_data, positions, strict=True))
        _MESH_COORDINATES[spec] = coordinates
    return coordinates


@dataclass(frozen=True)
class DTensorShardingMetadata(ShardingMetadata):
    """
    Complete metadata for a DTensor (Distributed Tensor).

    Contains all information needed to reconstruct a DTensor from checkpoint data,
    including shape, dtype, stride, mesh specification, and placement information.
    This metadata is specifically designed for PyTorch DTensor objects.

    Args:
        global_shape: Global DTensor shape dimensions
        dtype: String representation of tensor data type
        stride: Tensor stride information
        mesh_spec: Device mesh specification
        placements: Tuple of placement specifications for each mesh dimension
    """

    global_shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    mesh_spec: DeviceMeshSpec
    placements: tuple[_PlacementSpec, ...]

    @classmethod
    def from_dtensor(
        cls,
        dtensor: DTensor,
    ) -> "DTensorShardingMetadata":
        """Create DTensorShardingMetadata from a DTensor."""
        # Convert DTensor placements to our custom placement types
        converted_placements = []
        for placement in dtensor.placements:
            if isinstance(placement, DTensorStridedShard):
                converted_placements.append(
                    StridedShardSpec(
                        dim=placement.dim,
                        split_factor=int(placement.split_factor),
                    )
                )
                continue
            elif isinstance(placement, DTensorShard):
                converted_placements.append(ShardSpec(dim=placement.dim))
                continue
            elif isinstance(placement, DTensorReplicate):
                converted_placements.append(ReplicateSpec())
                continue
            raise RuntimeError(
                f"Unsupported placement type {type(placement)} encountered for DTensor. "
                "During checkpointing, we only support Shard, StridedShard, and "
                "Replicate placements."
            )

        mesh_spec = DeviceMeshSpec.from_device_mesh(dtensor.device_mesh)

        return DTensorShardingMetadata(
            global_shape=tuple(dtensor.shape),
            dtype=str(dtensor.dtype),
            stride=tuple(dtensor.stride()),
            mesh_spec=mesh_spec,
            placements=tuple(converted_placements),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the tensor metadata to a dictionary for serialization.

        Returns:
            A dictionary representation of the tensor metadata.
        """
        return {
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "stride": list(self.stride),
            "mesh_spec": self.mesh_spec.to_dict(),
            "placements": [p.to_dict() for p in self.placements],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DTensorShardingMetadata":
        """
        Create a DTensorShardingMetadata object from a dictionary representation.

        Returns:
            A DTensorShardingMetadata object constructed from the dictionary.
        """
        placements: list[_PlacementSpec] = []
        for placement in d["placements"]:
            placement_type = placement["type"]
            if placement_type == "Shard":
                placements.append(ShardSpec.from_dict(placement))
            elif placement_type == "StridedShard":
                placements.append(StridedShardSpec.from_dict(placement))
            elif placement_type == "Replicate":
                placements.append(ReplicateSpec.from_dict(placement))
            else:
                raise ValueError(f"Unknown DTensor placement type: {placement_type}")

        return cls(
            global_shape=tuple(d["global_shape"]),
            dtype=d["dtype"],
            stride=tuple(d["stride"]),
            mesh_spec=DeviceMeshSpec.from_dict(d["mesh_spec"]),
            placements=tuple(placements),
        )

    @property
    def equivalent_ranks(self) -> tuple[int, ...] | None:
        """Return all ranks in the mesh - they all share identical metadata."""
        return self.mesh_spec.mesh_data
