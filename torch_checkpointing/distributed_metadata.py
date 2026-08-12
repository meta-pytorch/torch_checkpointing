# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed checkpoint metadata required to reshard distributed tensors.

This module provides dataclasses for describing the distributed nature of
objects being checkpointed. It defines core abstract base classes that can
be extended to support various sharding frameworks.

The key components are:
- ShardingMetadata: Abstract base class with registry for polymorphic serialization
- GlobalObjectMetadata: Aggregated per-path metadata with rank grouping
- DistributedItemMetadata: Per-item view of distributed metadata for loading
- DistributedMetadata: Top-level container for checkpoint metadata
- CheckpointMetadata: Combined view of distributed and local metadata
"""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from .checkpoint_layout import default_layout_info, LayoutInfo
from .logging_utils import EventLogger, EventType
from .storage.base_storage import Storage
from .types import CheckpointPath, NestedPath

logger = logging.getLogger(__name__)


METADATA_FILE_NAME: str = "metadata.pkl"

_CURRENT_VERSION: str = "2.0"
_SUPPORTED_VERSIONS: set[str] = {"1.0", "2.0"}  # Support loading both formats


class ShardingMetadata(ABC):
    """
    Abstract base class for sharding metadata that describes how objects in a
    STATE_DICT are sharded across ranks.

    This class provides a registry mechanism for subclasses to automatically
    register themselves, enabling polymorphic serialization and deserialization
    of different sharding metadata types.
    """

    _registry: ClassVar[dict[str, type["ShardingMetadata"]]] = {}
    _type_name: ClassVar[str]

    # auto-register subclasses, so we can use the type name at run time to call the
    # corresponding from_dict method
    def __init_subclass__(cls, /, type_name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = type_name or cls.__name__
        ShardingMetadata._registry[name] = cls
        cls._type_name = name  # convenience for dumps

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the sharding metadata to a dictionary for serialization.

        Returns:
            A dictionary representation of the sharding metadata.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShardingMetadata":
        """
        Create a ShardingMetadata object from a dictionary representation.

        Returns:
            A ShardingMetadata object constructed from the dictionary.
        """
        pass

    @property
    @abstractmethod
    def equivalent_ranks(self) -> tuple[int, ...] | None:
        """
        Return ranks that share identical metadata.

        Returns:
            Tuple of ranks with identical metadata, or None if not applicable.
        """
        ...

    def _pack(self) -> dict[str, Any]:
        """
        Pack the sharding metadata into a typed dictionary for serialization.

        Returns:
            A dictionary containing the type name and serialized data.
        """
        return {"type": self._type_name, "data": self.to_dict()}

    @classmethod
    def _unpack(cls, packed: dict[str, Any]) -> "ShardingMetadata":
        """
        Unpack a typed dictionary into a ShardingMetadata subclass instance.

        Args:
            packed: Dictionary containing 'type' and 'data' fields.

        Returns:
            A ShardingMetadata subclass instance constructed from the packed data.

        Raises:
            ValueError: If the type field is missing or unknown.
        """
        t = packed.get("type")
        if t is None:
            raise ValueError(
                "Invalid ShardingMetadata format: missing 'type' field in packed data"
            )
        if t not in ShardingMetadata._registry:
            available_types = ", ".join(sorted(ShardingMetadata._registry.keys()))
            raise ValueError(
                f"Unknown ShardingMetadata type '{t}'. "
                f"Available types: {available_types}"
            )
        metadata_cls = ShardingMetadata._registry[t]
        return metadata_cls.from_dict(packed["data"])


# Type alias for metadata of all sharded objects within one checkpoint item
# Maps nested path (within item) to the sharding metadata for that object
ItemMetadata = dict[NestedPath, ShardingMetadata]


def _serialize_nested_path(path: NestedPath) -> str:
    """Serialize a NestedPath to a JSON string for use as dict key."""
    return json.dumps(list(path), separators=(",", ":"))


def _deserialize_nested_path(data: str) -> NestedPath:
    """Deserialize a JSON string back to a NestedPath."""
    return tuple(json.loads(data))


@dataclass(frozen=True)
class GlobalObjectMetadata:
    """
    Aggregated metadata for a single CheckpointPath from a group of ranks.

    This class combines sharding metadata for one path with the ranks that share
    this exact metadata, enabling efficient deduplication based on ShardingMetadata hash.

    Attributes:
        sharding_metadata: The ShardingMetadata for this checkpoint path.
        ranks: Tuple of ranks that share this exact metadata configuration.
    """

    sharding_metadata: ShardingMetadata
    ranks: tuple[int, ...]

    @cached_property
    def _hash(self) -> int:
        return hash((self.sharding_metadata, self.ranks))

    def __hash__(self) -> int:
        return self._hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sharding_metadata": self.sharding_metadata._pack(),
            "ranks": self.ranks,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GlobalObjectMetadata":
        """Create from dictionary representation."""
        return cls(
            sharding_metadata=ShardingMetadata._unpack(d["sharding_metadata"]),
            ranks=tuple(d["ranks"]),
        )


@dataclass(frozen=True)
class DistributedItemMetadata:
    """
    Per-item metadata stored directly in DistributedMetadata.

    This class stores the distributed metadata for a single checkpoint item,
    organized by nested path within that item. It provides efficient O(1) lookup
    by nested path and includes layout information for file access.

    Attributes:
        nested_path_to_metadata: Dict mapping NestedPath (within item) to list of
            GlobalObjectMetadata, each containing the sharding metadata and ranks
            that share it.
        rank_to_layout_info: Mapping from source rank to LayoutInfo for this item.
    """

    nested_path_to_metadata: dict[NestedPath, list[GlobalObjectMetadata]]
    rank_to_layout_info: dict[int, LayoutInfo | None]

    def get_file_path(self, rank: int, checkpoint_path: Path, item_key: str) -> Path:
        """
        Construct source file path for a given rank.

        Args:
            rank: The source rank to get the file path for.
            checkpoint_path: Base path to the checkpoint directory.
            item_key: The checkpoint item key.

        Returns:
            Full path to the checkpoint file for this rank and item.
        """
        layout = self.rank_to_layout_info.get(rank)
        if layout is None:
            layout = default_layout_info(item_key, rank)
        return checkpoint_path / layout.file_path

    def get_metadata_for_path_and_rank(
        self, nested_path: NestedPath, rank: int
    ) -> ShardingMetadata | None:
        """Get sharding metadata for a specific nested path and source rank."""
        if nested_path not in self.nested_path_to_metadata:
            return None
        for group in self.nested_path_to_metadata[nested_path]:
            if rank in group.ranks:
                return group.sharding_metadata
        return None

    def get_all_ranks(self) -> set[int]:
        """Get all source ranks that have any path in this item."""
        ranks: set[int] = set()
        for groups in self.nested_path_to_metadata.values():
            for group in groups:
                ranks.update(group.ranks)
        return ranks

    def get_nested_paths(self) -> set[NestedPath]:
        """Get all nested paths in this item's metadata."""
        return set(self.nested_path_to_metadata.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nested_path_to_metadata": {
                _serialize_nested_path(nested_path): [
                    group.to_dict() for group in groups
                ]
                for nested_path, groups in self.nested_path_to_metadata.items()
            },
            "rank_to_layout_info": {
                rank: (layout_info.to_dict() if layout_info is not None else None)
                for rank, layout_info in self.rank_to_layout_info.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistributedItemMetadata":
        """Create from dictionary representation."""
        nested_path_to_metadata = {
            _deserialize_nested_path(nested_path_str): [
                GlobalObjectMetadata.from_dict(group) for group in groups
            ]
            for nested_path_str, groups in d["nested_path_to_metadata"].items()
        }
        rank_to_layout_info = {
            int(rank): (
                LayoutInfo.from_dict(layout_dict) if layout_dict is not None else None
            )
            for rank, layout_dict in d["rank_to_layout_info"].items()
        }
        return cls(
            nested_path_to_metadata=nested_path_to_metadata,
            rank_to_layout_info=rank_to_layout_info,
        )


@dataclass(frozen=True)
class DistributedMetadata:
    """
    Container for distributed checkpoint metadata grouped by item_key.

    Uses a format where each item_key maps to a DistributedItemMetadata containing
    the nested path metadata and layout info for that item. This provides intuitive
    organization matching how users think about checkpoint items, with direct O(1)
    access via `.metadata[item_key]`.

    Each DistributedItemMetadata contains per-nested-path sharding info with a `ranks`
    field listing which ranks share this exact metadata configuration.

    In the common case (FSDP/DP), each path has a single GlobalObjectMetadata entry
    with all ranks grouped together since they share identical sharding metadata.
    This enables efficient deduplication based on ShardingMetadata hash.

    This class enables:
    - Cross-topology checkpoint loading (different mesh shapes)
    - Efficient metadata serialization/deserialization using ShardingMetadata hash
    - Support for both tensor and custom object types
    - Version-compatible checkpoint format evolution

    Attributes:
        metadata: Maps item_key to DistributedItemMetadata containing the nested
            path metadata and layout info for that item.
        world_size: Total number of ranks that participated in writing the checkpoint
        version: Metadata format version for backward compatibility.

    Example:
        >>> # Create metadata for a simple model (common case - all ranks share metadata)
        >>> all_ranks = tuple(range(100000))  # 100k ranks
        >>> nested_path = ("encoder", "weight")
        >>> item_metadata = DistributedItemMetadata(
        ...     nested_path_to_metadata={
        ...         nested_path: [GlobalObjectMetadata(sharding_metadata=dtensor_meta, ranks=all_ranks)],
        ...     },
        ...     rank_to_layout_info={0: LayoutInfo(...), 1: LayoutInfo(...), ...},
        ... )
        >>> metadata = DistributedMetadata(
        ...     metadata={"model": item_metadata},
        ...     world_size=100000,
        ... )
    """

    # Maps item_key to DistributedItemMetadata
    metadata: dict[str, DistributedItemMetadata]

    world_size: int
    version: str = _CURRENT_VERSION

    def __post_init__(self) -> None:
        """Validate distributed metadata invariants."""
        # Validate all ranks are present in each item's layout info
        expected_ranks = set(range(self.world_size))

        for item_key, item_metadata in self.metadata.items():
            actual_ranks = set(item_metadata.rank_to_layout_info.keys())
            if actual_ranks != expected_ranks:
                missing = expected_ranks - actual_ranks
                extra = actual_ranks - expected_ranks
                raise RuntimeError(
                    f"Item '{item_key}' rank_to_layout_info must contain all ranks 0 to {self.world_size - 1}. "
                    f"Missing ranks: {sorted(missing)}, Extra ranks: {sorted(extra)}"
                )

        # Validate no two ranks write to same file path (across all items)
        used_paths: dict[str, tuple[str, int]] = {}  # file_path -> (item_key, rank)
        for item_key, item_metadata in self.metadata.items():
            for rank, layout_info in item_metadata.rank_to_layout_info.items():
                if layout_info is None:
                    continue
                file_path = layout_info.file_path
                if file_path in used_paths:
                    other_item, other_rank = used_paths[file_path]
                    raise RuntimeError(
                        f"Item '{other_item}' rank {other_rank} & item '{item_key}' rank {rank} "
                        f"both write to the same file path {file_path}"
                    )
                used_paths[file_path] = (item_key, rank)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the metadata to a dictionary for serialization.

        Serializes in v3.0 format with nested structure.

        Returns:
            A dictionary representation of the metadata.
        """
        event_logger = EventLogger()
        result: dict[str, Any] = {
            "metadata": {
                item_key: item_metadata.to_dict()
                for item_key, item_metadata in self.metadata.items()
            },
            "world_size": self.world_size,
            "version": self.version,
        }

        num_items = len(result["metadata"])
        logger.info(
            f"Completed to_dict() for DistributedMetadata with {num_items} items",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.to_dict.latency_ms",
            ),
        )

        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistributedMetadata":
        """
        Create a DistributedMetadata object from a dictionary representation.

        Supports loading v1.0 (per-path flat) and v2.0 (grouped by item_key) formats.

        Returns:
            A DistributedMetadata object constructed from the dictionary.
        """
        event_logger = EventLogger()
        version = d.get("version", "1.0")

        if version == "2.0":
            # v2.0 format - grouped by item_key with DistributedItemMetadata
            metadata = {
                item_key: DistributedItemMetadata.from_dict(item_dict)
                for item_key, item_dict in d["metadata"].items()
            }
        else:
            # v1.0 format - per-path flat, convert to v2.0 grouped format
            logger.info(
                "Converting v1.0 metadata to v2.0 format",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name="train.checkpoint_metadata.v1_to_v2_conversion",
                ),
            )
            metadata = cls._convert_v1_to_v2(
                d["metadata"], d["rank_to_layout_info_mappings"]
            )

        result = cls(
            metadata=metadata,
            world_size=d["world_size"],
            version=_CURRENT_VERSION,  # Always use current version
        )

        logger.info(
            f"Completed from_dict() for DistributedMetadata with {len(result.metadata)} items (loaded from v{version})",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.from_dict.latency_ms",
            ),
        )

        return result

    @classmethod
    def _convert_v1_to_v2(
        cls,
        v1_metadata: dict[str, list[dict[str, Any]]],
        raw_layout_info: dict[str | int, dict[str, Any]],
    ) -> dict[str, DistributedItemMetadata]:
        """
        Convert v1.0 per-path format to v2.0 grouped format.

        v1.0 format: dict[CheckpointPath_serialized, list[GlobalObjectMetadata_dict]]
        v2.0 format: dict[item_key, DistributedItemMetadata]
        """
        # Parse layout info
        rank_to_layout_info_mappings: dict[int, dict[str, LayoutInfo | None]] = {
            int(rank): {
                key: (
                    LayoutInfo.from_dict(layout_dict)
                    if layout_dict is not None
                    else None
                )
                for key, layout_dict in layout_mapping.items()
            }
            for rank, layout_mapping in raw_layout_info.items()
        }

        # Group metadata by item_key
        item_to_nested_paths: dict[
            str, dict[NestedPath, list[GlobalObjectMetadata]]
        ] = {}

        for path_str, groups in v1_metadata.items():
            path = CheckpointPath.deserialize(path_str)
            item_key = path.item_key
            nested_path = path.nested_path

            if item_key not in item_to_nested_paths:
                item_to_nested_paths[item_key] = {}

            item_to_nested_paths[item_key][nested_path] = [
                GlobalObjectMetadata.from_dict(group) for group in groups
            ]

        # Build DistributedItemMetadata for each item
        result: dict[str, DistributedItemMetadata] = {}
        for item_key, nested_path_to_metadata in item_to_nested_paths.items():
            # Build rank_to_layout_info for this item
            rank_to_layout_info: dict[int, LayoutInfo | None] = {
                rank: layout_mapping.get(item_key)
                for rank, layout_mapping in rank_to_layout_info_mappings.items()
            }
            result[item_key] = DistributedItemMetadata(
                nested_path_to_metadata=nested_path_to_metadata,
                rank_to_layout_info=rank_to_layout_info,
            )

        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DistributedMetadata):
            return False
        return (
            self.metadata == other.metadata
            and self.world_size == other.world_size
            and self.version == other.version
        )


@dataclass
class CheckpointMetadata:
    """
    Container for both distributed and local metadata extracted from a state dictionary.

    This dataclass combines two complementary views of checkpoint metadata:
    - distributed_metadata: Global view of how tensors/objects are distributed across all ranks
    - local_metadata: Local view of sharding metadata for objects present on the current rank

    Attributes:
        distributed_metadata: Complete metadata aggregated from all ranks.
        local_metadata: Mapping from item_key to nested path metadata for objects present
            on the current rank only. Structure: dict[item_key, dict[NestedPath, ShardingMetadata]].
    """

    distributed_metadata: DistributedMetadata
    local_metadata: dict[str, dict[NestedPath, ShardingMetadata]]


def load_distributed_metadata(
    checkpoint_dir: str | Path,
    storage: Storage,
) -> DistributedMetadata | None:
    """Load distributed metadata from a trusted checkpoint.

    .. warning::
        ``metadata.pkl`` is deserialized with pickle and may execute arbitrary
        code. Only load checkpoints from trusted sources that have not been
        tampered with.
    """
    metadata_path = Path(checkpoint_dir) / METADATA_FILE_NAME
    if not storage.exists(metadata_path):
        return None
    return DistributedMetadata.from_dict(pickle.loads(storage.read(metadata_path)))
