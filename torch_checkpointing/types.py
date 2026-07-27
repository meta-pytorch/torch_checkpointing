# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Type definitions for distributed training and checkpointing.

This module provides type definitions and classes for managing rank information
in distributed training environments, which is essential for proper checkpoint
saving and loading.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping

from typing_extensions import TypeAlias

# Type alias for state dictionaries used in checkpointing
STATE_DICT: TypeAlias = Mapping[str, Any]

# Type alias for a top-level checkpoint item key.
ItemKey: TypeAlias = str

# Type alias for nested paths within a checkpoint item
# Components can be strings (dict keys) or ints (sequence indices)
NestedPath = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class CheckpointPath:
    """
    Strongly-typed path to an object within CheckpointInfo.

    This provides unambiguous identification of checkpointed objects by explicitly
    separating the top-level checkpoint item key from any nested path within that
    item's value.

    Attributes:
        item_key: The top-level key in CheckpointInfo.checkpoint_items.
                  Must be a valid filename component (alphanumeric, hyphens, underscores).
        nested_path: Path components within the item's value.
                    - (): Empty tuple refers to the item's value itself (leaf value like scalar/tensor)
                    - ("key",): Single-level dict key access
                    - ("key", 0, "field"): Mixed dict/sequence access
                    - (0, 1): Nested sequence access

    Examples:
        >>> CheckpointPath("step")  # Leaf: checkpoint_items["step"].value directly
        >>> CheckpointPath("model", ("encoder.weight",))  # Dict: state_dict["encoder.weight"]
        >>> CheckpointPath("optimizer", ("state", 0, "exp_avg"))  # optimizer.state[0]["exp_avg"]
        >>> CheckpointPath("data", (0,))  # Sequence: data[0]
    """

    item_key: str
    nested_path: NestedPath = ()

    def __str__(self) -> str:
        """Human-readable string representation '{item_key}::{nested_path}'."""
        if len(self.nested_path) > 0:
            path_str = ".".join(str(c) for c in self.nested_path)
            return f"{self.item_key}::{path_str}"
        return self.item_key

    def serialize(self) -> str:
        """Serialize to a compact JSON string.

        Examples:
            >>> CheckpointPath("model").serialize()
            '["model"]'
            >>> CheckpointPath("model", ("encoder", 0, "weight")).serialize()
            '["model","encoder",0,"weight"]'
        """
        return json.dumps([self.item_key, *self.nested_path], separators=(",", ":"))

    @classmethod
    def deserialize(cls, data: str) -> "CheckpointPath":
        """Deserialize from a JSON string.

        Args:
            data: A JSON-encoded string representing [item_key, *nested_path].

        Examples:
            >>> CheckpointPath.deserialize('["model"]')
            CheckpointPath("model")
            >>> CheckpointPath.deserialize('["model","encoder",0]')
            CheckpointPath("model", ("encoder", 0))
        """

        parsed = json.loads(data)
        if len(parsed) == 0:
            raise ValueError("CheckpointPath list cannot be empty")
        return cls(
            item_key=parsed[0],
            nested_path=tuple(parsed[1:]),
        )


@dataclass
class RankInfo:
    """
    Information about the current rank in a distributed training environment.

    Attributes:
        global_rank: Global rank of the current process.
        global_world_size: Total number of processes in the distributed environment.
        role_rank: Rank within the current role.
        role_world_size: Total number of processes in the current role.
    """

    global_rank: int
    global_world_size: int
    role_rank: int
    role_world_size: int
