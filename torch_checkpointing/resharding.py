# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        src_dtype: Canonical source tensor dtype, such as ``torch.float16``.
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

    # Empty for legacy plans; loaders resolve it from checkpoint metadata.
    src_dtype: str = ""


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
        """If True, load this item without checking whether it needs resharding.

        A resharding load is expensive: the reader loads the checkpoint's
        distributed metadata, then calls ``should_reshard`` to compare each item's
        saved sharding with the target's. On a job retry whose mesh matches the
        one that saved the checkpoint, both are wasted work, so a resharder can
        opt out with this property.

        How the reader loads::

            load(path) or load(path, metadata_format=...)
                                            |
                                            v
            +-----------------------------------------------------------+
            | Every item has no resharder, or a resharder that skips,   |-- no --+
            | and metadata_format is unset or rank addressable?         |        |
            +-----------------------------------------------------------+        |
                    | yes                                                         v
                    v                                        +-------------------------------------------+
            +-------------------------------------------+   | Load metadata once, for every item. With  |
            | Read each item's file for this rank, in   |   | metadata_format: only that format, and    |
            | full, without loading metadata. Layout:   |   | raise if the checkpoint lacks it.         |
            | 1. The item's configured layout, if set.  |   | Without: the reader's own formats.        |
            | 2. The default <item_key>_<rank>.pt.      |   +-------------------------------------------+
            | A missing file raises, naming the item.   |                         | for each item, including
            +-------------------------------------------+                         | items with no resharder
                                                                                  v
                                                            +-------------------------------------------+
                                                            | Resharder that does not skip, and         |-yes-> Reshard
                                                            | should_reshard?                           |
                                                            +-------------------------------------------+
                                                                                  | no
                                                                                  v
                                                            +-------------------------------------------+
                                                            | metadata_format not rank addressable and  |-yes-> Raise:
                                                            | its metadata describes this item?         |       needs a
                                                            +-------------------------------------------+       resharder
                                                                                  | no
                                                                                  v
                                                            +-------------------------------------------+
                                                            | Read the item's file for this rank, in    |
                                                            | full. Layout:                             |
                                                            | 1. The layout the metadata records for    |
                                                            |    this rank, if any.                     |
                                                            | 2. The item's configured layout, if set.  |
                                                            | 3. The default <item_key>_<rank>.pt.      |
                                                            | A missing file raises, naming the item.   |
                                                            +-------------------------------------------+

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
