"""
Resharding APIs for checkpoint loading across different distributed configurations.

This module provides the abstract base class for customizing resharding logic
during checkpoint loading. Implementations control how data from source checkpoints
are mapped to target tensors when distributed configurations differ.
"""

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .distributed_metadata import (
    DistributedItemMetadata,
    ShardingMetadata,
)
from .storage.base_storage import Storage
from .types import NestedPath


@dataclass
class LoadPlan:
    """
    Describes how to fill a chunk of a target tensor with data from a source checkpoint.

    This dataclass contains the coordinates needed to read data from source checkpoints
    and copy it to the correct location in the target tensor.

    Attributes:
        offsets: Position of the chunk inside the target tensor.
        sizes: Shape of the chunk inside the target tensor.
        src_rank: Source rank identifying which checkpoint file to read from.
        src_fqn: Fully qualified name of the source tensor to read.
        src_offsets: Position of the chunk inside the source tensor.
        src_sizes: Shape of the chunk inside the source tensor.
        transpose_dims: Transpose to apply to source chunk to match target layout.
    """

    # Target chunk info
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]

    # Source chunk info
    src_rank: int
    src_fqn: str
    src_offsets: tuple[int, ...]
    src_sizes: tuple[int, ...]

    # Only used for transposition
    transpose_dims: tuple[int, ...] = ()

    # Source element size in bytes (e.g., 4 for float32, 2 for bfloat16).
    # 0 means not set.
    src_elem_size: int = 0


@dataclass
class ReshardingInfo:
    """
    Container for resharding information generated during checkpoint loading.

    Attributes:
        nested_path_to_load_plans: Dictionary mapping NestedPaths to lists of LoadPlan
            objects describing how to load and reshard from source to target.
        non_reshardable_paths: List of NestedPaths that could not be resharded.
    """

    nested_path_to_load_plans: dict[NestedPath, list[LoadPlan]]
    non_reshardable_paths: list[NestedPath]


class Resharder(abc.ABC):
    """
    Abstract base class for customizing resharding logic during checkpoint loading.

    This class defines the interface for handling resharding of checkpointed state
    (tensors, parameters) when loading across different sharding strategies or device
    meshes. Implementations control how data chunks from source checkpoints are mapped
    to target tensors.

    The API operates at item level (e.g., "model", "optimizer"), with path-level
    details being internal to resharder implementations.

    Typical use cases include:
      - Resharding tensors when changing parallelism strategies (e.g., data parallel
        to tensor parallel).
      - Loading checkpoints across different device mesh configurations.
      - Handling custom sharding annotations or non-standard tensor layouts.
      - Supporting advanced resharding scenarios where source and target layouts
        differ significantly.
    """

    @property
    def skip_resharding(self) -> bool:
        """If True, skip metadata loading and resharding checks during load.

        Use for job retry scenarios where mesh config is identical between
        save and load, to avoid expensive metadata loading overhead.

        Subclasses can override this property to control skip behavior.
        Default is False (perform resharding as normal).
        """
        return False

    @abc.abstractmethod
    def extract_sharding_metadata(
        self,
        item_key: str,
        item_value: Any,
    ) -> dict[NestedPath, ShardingMetadata]:
        """
        Extract sharding metadata for a checkpoint item.

        This method walks the item's value and extracts ShardingMetadata for each
        sharded object (e.g., DTensor) found within. The walking logic is handled
        internally by the implementation.

        Args:
            item_key: The checkpoint item key (e.g., "model", "optimizer")
            item_value: The item's value (e.g., state_dict, tensor, etc.)

        Returns:
            Dictionary mapping NestedPath (within item) to ShardingMetadata for each
            sharded object within the item. Empty dict if no sharded objects.
        """
        ...

    def should_reshard(
        self,
        source_metadata: DistributedItemMetadata | None,
        target_metadata: dict[NestedPath, ShardingMetadata] | None,
    ) -> bool:
        """
        Determine if resharding is needed for a specific checkpoint item.

        This method compares the source distributed metadata from the checkpoint
        with the target metadata to decide whether resharding is necessary.

        The default implementation returns True if metadata differs between source
        and target for any common path.

        Subclasses can override this method to implement custom logic, such as:
        - Checking specific metadata fields (e.g., only mesh topology)
        - Supporting partial compatibility (e.g., allowing certain layout differences)
        - Adding performance-based heuristics

        Args:
            source_metadata: Distributed metadata from the checkpoint being loaded.
                None if the checkpoint doesn't contain metadata.
            target_metadata: This rank's target sharding from extract_sharding_metadata().
                None if not provided by the user.

        Returns:
            True if resharding is needed, False otherwise.
        """
        # If either metadata is missing, we cannot reshard
        if source_metadata is None or target_metadata is None:
            return False

        # Compare target metadata for each nested path against source rank groups
        for nested_path, target_sharding in target_metadata.items():
            if nested_path not in source_metadata.nested_path_to_metadata:
                continue  # Path not in source - will be handled as missing

            # Check if any source rank group has matching metadata for this path
            found_match = False
            for group in source_metadata.nested_path_to_metadata[nested_path]:
                if group.sharding_metadata == target_sharding:
                    found_match = True
                    break

            if not found_match:
                return True  # This path needs resharding

        return False

    @abc.abstractmethod
    def load(
        self,
        source_path: Path,
        item_key: str,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
        target: Any,
        storage: Storage,
    ) -> list[NestedPath]:
        """
        Load and reshard checkpoint data into target.

        This method combines load plan generation and execution into a single call,
        allowing flexible resharding strategies. Implementations can generate load
        plans internally or implement custom loading logic.

        Args:
            source_path: Base path to the source checkpoint directory
            item_key: The checkpoint item key being loaded (e.g., "model")
            target_metadata: This rank's target sharding (from extract_sharding_metadata)
            source_metadata: Source checkpoint's distributed metadata for this item
            target: Target object to load data into (modified in-place)
            storage: Storage backend for reading checkpoint files

        Returns:
            List of NestedPaths that could not be resharded.
        """
        ...
