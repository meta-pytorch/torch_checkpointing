# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base classes for checkpointing.

This module defines the fundamental data structures used in checkpointing:
- CheckpointItem: Represents a single item in a checkpoint
- CheckpointBase: Abstract base class for checkpoint objects
- CheckpointInfo: Encapsulates checkpoint data with guaranteed consistency
"""

import abc
import re
from collections.abc import KeysView
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from typing_extensions import TypeAlias

from .checkpoint_layout import LayoutInfo
from .distributed_metadata import CheckpointMetadata
from .resharding import Resharder
from .types import STATE_DICT

# Type alias for state_dict values - descriptive name without generic complexity
TValue: TypeAlias = Any


@dataclass
class CheckpointItem:
    """Represents a single item in a checkpoint.

    Attributes:
    ----------
    value:
        The value of the item. E.g. a torch.nn.Module or a torch.optim.Optimizer.

        For write operations: The actual data to save.

        For read operations: The template structure that controls filtering:
        - If None, loads all data from the checkpoint file without filtering.
        - If a dict/list structure, only loads keys/indices present in this structure.
        - Tensors are copied in-place to preserve references.
    requires_copy:
        Whether the value requires a copy during async checkpointing, as otherwise
        it can get changed by the caller, causing a race condition.
    layout:
        Provides instructions on where to save the item. All items without a layout
        will be saved into a file with the same name as the key + rank id
        using `torch.save()`.
    resharder:
        Optional resharder for redistributing checkpoint data across different
        parallelization strategies during loading. Used when the checkpoint was
        saved with a different sharding configuration than the current one.
    """

    value: Any = None
    requires_copy: bool = True
    layout: LayoutInfo | None = None
    resharder: Resharder | None = None


class CheckpointBase(abc.ABC):
    """
    Abstract base class for representing a checkpoint object to be saved.
    """

    @abc.abstractmethod
    def get_items(self) -> dict[str, CheckpointItem]:
        """
        Override this method in your checkpoint objects such that it returns a dict of CheckpointItem objects representing the checkpoint, keyed by the item's key. By default, all items are saved into separate files with the same name as the key + rank id using `torch.save()`.

        The dict keys identify each checkpoint item and must be valid filename components
        (alphanumeric characters, hyphens, and underscores only). Keys must be unique.

        Note that you can specialize this method on different ranks. E.g.:
            * For custom file names and serialization (aka layout), you can specify file name based on the rank.
            * For rank 0, you may want to include some global metadata that is the
              same across all ranks.
        """
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict: STATE_DICT) -> None:
        """
        Override this method to load the state from a loaded state dictionary into this checkpoint object.

        This is called after loading a checkpoint from storage. The checkpoint object
        should already have its model, optimizer, and other components initialized,
        and this method should load the saved state into them.

        Args:
            state_dict: The loaded state dictionary containing saved state
        """
        pass


@dataclass
class CheckpointInfo:
    """
    Base class encapsulating checkpoint data with guaranteed consistency between
    state_dict and layout_info_mappings keys.

    This class ensures by design that state_dict and layout_info_mappings
    always have the same set of keys, preventing divergence issues.

    Args:
        checkpoint_items: Dict of CheckpointItem objects keyed by the item's key.
              Keys must be valid filename components (alphanumeric, hyphens, underscores).
    """

    checkpoint_items: dict[str, CheckpointItem]

    def __post_init__(self) -> None:
        """Initialize internal mappings from checkpoint items and validate keys."""
        # Validate all keys are valid filename components
        invalid_keys = []
        for key in self.checkpoint_items.keys():
            if not re.match(r"^[a-zA-Z0-9_-]+$", key):
                invalid_keys.append(key)

        if invalid_keys:
            raise ValueError(
                f"Invalid checkpoint keys {invalid_keys}: keys must contain only "
                "alphanumeric characters, hyphens, and underscores (no path "
                "separators, whitespace, special characters, or extensions)"
            )

        # Build internal mappings - iterate over dict.items() to get both key and item
        self._value_and_layout_info_mappings: dict[
            str, tuple[TValue, LayoutInfo | None]
        ] = {
            key: (item.value, item.layout)
            for key, item in self.checkpoint_items.items()
        }

    @cached_property
    def state_dict(self) -> dict[str, TValue]:
        """Returns the state dictionary with values only."""
        return {
            key: value
            for key, (value, _) in self._value_and_layout_info_mappings.items()
        }

    @cached_property
    def layout_info_mappings(self) -> dict[str, LayoutInfo | None]:
        """Returns the layout info mappings only."""
        return {
            key: layout
            for key, (_, layout) in self._value_and_layout_info_mappings.items()
        }

    @property
    def keys(self) -> KeysView[str]:
        """Returns the keys (same for both state_dict and layout_info_mappings)."""
        return self._value_and_layout_info_mappings.keys()

    def for_writes(
        self, serialized_distributed_metadata: bytes | None = None
    ) -> "CheckpointWriteInfo":
        """
        Create a CheckpointWriteInfo from this CheckpointInfo for write operations.

        Args:
            serialized_distributed_metadata: Optional pre-serialized bytes of
                distributed metadata to be written directly to storage.

        Returns:
            CheckpointWriteInfo ready for writing to storage.
        """
        return CheckpointWriteInfo(
            checkpoint_items=self.checkpoint_items,
            serialized_distributed_metadata=serialized_distributed_metadata,
        )

    def for_reads(
        self, checkpoint_metadata: CheckpointMetadata | None = None
    ) -> "CheckpointReadInfo":
        """
        Create a CheckpointReadInfo from this CheckpointInfo for read operations.

        Args:
            checkpoint_metadata: Optional checkpoint metadata needed for resharding
                during read operations.

        Returns:
            CheckpointReadInfo ready for reading from storage.
        """
        return CheckpointReadInfo(
            checkpoint_items=self.checkpoint_items,
            checkpoint_metadata=checkpoint_metadata,
        )


@dataclass
class CheckpointWriteInfo(CheckpointInfo):
    """
    Encapsulates checkpoint data for writing to storage.

    Extends CheckpointInfo with serialized distributed metadata that can be
    written directly to storage without further serialization.
    """

    serialized_distributed_metadata: bytes | None = field(default=None)


@dataclass
class CheckpointReadInfo(CheckpointInfo):
    """
    Encapsulates checkpoint data for reading from storage.

    Extends CheckpointInfo with checkpoint metadata needed for resharding
    during the read operation.
    """

    checkpoint_metadata: CheckpointMetadata | None = field(default=None)
