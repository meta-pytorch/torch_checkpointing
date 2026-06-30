"""
DTensor-specific metadata for distributed checkpointing.

This module provides dataclasses for describing the distributed nature of
DTensor objects being checkpointed. These are implementations of the abstract
ShardingMetadata interface defined in distributed_metadata.py.

The key components are:
- _PlacementSpec: Base class for DTensor placement specifications
- ShardSpec: Shard placement specification (shards a tensor dimension)
- ReplicateSpec: Replicate placement specification (replicates across a mesh dimension)
- DeviceMeshSpec: Lightweight, immutable device mesh specification
- DTensorShardingMetadata: Complete DTensor distribution information
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
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

    @cached_property
    def _hash(self) -> int:
        return hash(
            (self.device_type, self.mesh_shape, self.mesh_data, self.mesh_dim_names)
        )

    def __hash__(self) -> int:
        return self._hash

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

        Uses torch.where() on the reconstructed mesh tensor for vectorized lookup.
        Works correctly for non-contiguous meshes like [0, 2, 1, 3].

        Args:
            rank: The rank to find the coordinate for.

        Returns:
            A tuple of indices representing the coordinate of the rank in the mesh,
            or None if the rank is not found in the mesh.

        Example:
            For a mesh tensor([[0, 1], [2, 3]]) and rank=2,
            returns (1, 0) since rank 2 is at row 1, column 0.
        """
        indices = torch.where(self.mesh == rank)
        if indices[0].numel() == 0:
            return None
        return tuple(int(idx[0].item()) for idx in indices)


@lru_cache(maxsize=None)
def get_device_mesh_spec(
    device_type: str,
    mesh_shape: tuple[int, ...],
    mesh_data: tuple[int, ...],
    mesh_dim_names: tuple[str, ...] | None = None,
) -> DeviceMeshSpec:
    """
    Get or create a DeviceMeshSpec with reference deduplication.

    Uses lru_cache (unbounded) to return the same instance for identical
    parameters. This enables pickle's memo mechanism to automatically
    deduplicate DeviceMeshSpec references during serialization.

    Args:
        device_type: Type of device ("cuda", "cpu", etc.)
        mesh_shape: Shape of the mesh as tuple, e.g., (2, 4) for 8 GPUs
        mesh_data: Flattened mesh global rank IDs as tuple, e.g., (0, 2, 1, 3)
        mesh_dim_names: Optional names for mesh dimensions, e.g., ("dp", "tp")

    Returns:
        A cached DeviceMeshSpec instance for the given parameters.
    """
    return DeviceMeshSpec(
        device_type=device_type,
        mesh_shape=mesh_shape,
        mesh_data=mesh_data,
        mesh_dim_names=mesh_dim_names,
    )


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

    @cached_property
    def _hash(self) -> int:
        return hash(
            (
                self.global_shape,
                self.dtype,
                self.stride,
                self.mesh_spec,
                self.placements,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    @classmethod
    def from_dtensor(
        cls,
        dtensor: DTensor,
    ) -> "DTensorShardingMetadata":
        """Create DTensorShardingMetadata from a DTensor."""
        # Convert DTensor placements to our custom placement types
        converted_placements = []
        for placement in dtensor.placements:
            if isinstance(placement, DTensorShard):
                converted_placements.append(ShardSpec(dim=placement.dim))
                continue
            elif isinstance(placement, DTensorReplicate):
                converted_placements.append(ReplicateSpec())
                continue
            raise RuntimeError(
                f"Unsupported placement type {type(placement)} encountered for DTensor. "
                "During checkpointing, we only support Shard and Replicate placements."
            )

        # Create DeviceMeshSpec from the DTensor's device mesh using factory method
        # No need to clone - from_mesh converts tensor to tuples once at construction
        mesh_spec = DeviceMeshSpec.from_mesh(
            device_type=dtensor.device_mesh.device_type,
            mesh=dtensor.device_mesh.mesh,
            mesh_dim_names=dtensor.device_mesh.mesh_dim_names,
        )

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
        return cls(
            global_shape=tuple(d["global_shape"]),
            dtype=d["dtype"],
            stride=tuple(d["stride"]),
            mesh_spec=DeviceMeshSpec.from_dict(d["mesh_spec"]),
            placements=tuple(
                (
                    ShardSpec.from_dict(p)
                    if p["type"] == "Shard"
                    else ReplicateSpec.from_dict(p)
                )
                for p in d["placements"]
            ),
        )

    @property
    def equivalent_ranks(self) -> tuple[int, ...] | None:
        """Return all ranks in the mesh - they all share identical metadata."""
        return self.mesh_spec.mesh_data
