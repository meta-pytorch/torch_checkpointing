# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Walk utilities for traversing nested checkpoint structures.

This module provides functions for walking through nested data structures (dicts, lists,
tuples, etc.) commonly found in PyTorch checkpoints, applying callbacks to leaf values,
and tracking missing keys.
"""

from collections.abc import Mapping, MutableSequence, Sequence, Set
from typing import Any, Callable

import torch
from torch.distributed.tensor import DTensor

from .types import CheckpointPath

# String-like types are Sequences but should be treated as leaves
_STR_LIKE_TYPES = (str, bytes, bytearray)


def _check_type_compatibility(
    src: Any,
    tgt: Any,
    nested_path: tuple[str | int, ...],
) -> None:
    """Check that source and target types are compatible.

    Args:
        src: Source value.
        tgt: Target value.
        nested_path: Current path in the structure for error messages.

    Raises:
        RuntimeError: If target type is incompatible with source type.
    """
    if tgt is None:
        return

    # Get source type, handling FakeTensor
    src_type = type(src)
    if hasattr(src, "_tensor") or "FakeTensor" in str(type(src)):
        src_type = torch.Tensor

    # For Mappings, treat dict and OrderedDict as compatible
    if isinstance(tgt, Mapping) and isinstance(src, Mapping):
        return
    # For Sequences (excluding str-like), check both are sequences
    if (
        isinstance(tgt, Sequence)
        and not isinstance(tgt, _STR_LIKE_TYPES)
        and isinstance(src, Sequence)
        and not isinstance(src, _STR_LIKE_TYPES)
    ):
        return
    # For Sets, check both are sets
    if isinstance(tgt, Set) and isinstance(src, Set):
        return
    # For other types, check exact type match
    if not isinstance(tgt, src_type):
        raise RuntimeError(
            f"Target value at {nested_path=} is set to {type(tgt)}, "
            f"but source value is {type(src)}"
        )


def _default_leaf_fn(src: Any, tgt: Any) -> Any:
    """Default leaf function: copy tensors in-place, return source for others.

    Args:
        src: Source value.
        tgt: Target value.

    Returns:
        The target tensor (after in-place copy) if both are tensors, else source.

    Raises:
        RuntimeError: If one tensor is DTensor and the other is not.
    """
    if isinstance(src, torch.Tensor) and isinstance(tgt, torch.Tensor):
        src_is_dtensor = isinstance(src, DTensor)
        tgt_is_dtensor = isinstance(tgt, DTensor)
        if src_is_dtensor != tgt_is_dtensor:
            raise RuntimeError(
                f"Cannot copy between DTensor and regular Tensor: "
                f"source is {'DTensor' if src_is_dtensor else 'Tensor'}, "
                f"target is {'DTensor' if tgt_is_dtensor else 'Tensor'}"
            )
        # Extract local tensors for DTensors, use directly for regular Tensors
        if isinstance(src, DTensor):
            src_tensor = src._local_tensor
        else:
            src_tensor = src
        if isinstance(tgt, DTensor):
            tgt_tensor = tgt._local_tensor
        else:
            tgt_tensor = tgt
        tgt_tensor.copy_(src_tensor)
        return tgt
    return src


def _call_leaf(
    *,
    item_key: str,
    nested_path: tuple[str | int, ...],
    src: Any,
    tgt: Any,
    leaf_fn: Callable[[CheckpointPath, Any, Any], Any] | None,
) -> Any:
    """Call the leaf function, constructing CheckpointPath only when needed.

    Args:
        item_key: Top-level item key for CheckpointPath construction.
        nested_path: Current nested path.
        src: Source value.
        tgt: Target value.
        leaf_fn: Optional custom leaf function.

    Returns:
        Result of calling the leaf function.
    """
    if leaf_fn is not None:
        path = CheckpointPath(item_key=item_key, nested_path=nested_path)
        return leaf_fn(path, src, tgt)
    else:
        # No custom leaf_fn: use default behavior without path allocation
        return _default_leaf_fn(src, tgt)


def _walk_mapping(
    *,
    src: Mapping,
    tgt: Mapping | None,
    current_path_list: list[str | int],
    item_key: str,
    leaf_fn: Callable[[CheckpointPath, Any, Any], Any] | None,
    missing_keys: list[list[str | int]],
) -> dict:
    """Walk through a Mapping (dict, OrderedDict, etc.).

    Modifies the target dict in-place when provided.

    Args:
        src: Source mapping to traverse.
        tgt: Target mapping to merge into. If None, creates new dict.
        current_path_list: Current path as mutable list.
        item_key: Top-level item key for CheckpointPath.
        leaf_fn: Optional custom leaf function.
        missing_keys: List to collect missing key paths.

    Returns:
        The merged dict (same object as tgt if provided).
    """
    if tgt is None:
        tgt = dict.fromkeys(src.keys())

    for key in list(tgt.keys()):
        child_path = current_path_list + [key]
        if key in src:
            tgt[key] = _walk(
                src=src[key],
                tgt=tgt[key],
                current_path_list=child_path,
                item_key=item_key,
                leaf_fn=leaf_fn,
                missing_keys=missing_keys,
            )
        else:
            missing_keys.append(child_path.copy())
    return tgt


def _convert_to_original_type(result_list: list, original_type: type) -> Any:
    """Convert result list to the original container type.

    Args:
        result_list: The list of results to convert.
        original_type: The original container type (list, tuple, deque, etc.).

    Returns:
        Container of the original type, or list if conversion fails.
    """
    if original_type is list:
        return result_list
    elif original_type is tuple:
        return tuple(result_list)
    else:
        try:
            return original_type(result_list)  # type: ignore[abstract]
        except (TypeError, ValueError):
            return result_list


def _walk_sequence(
    *,
    src: Sequence,
    tgt: Sequence | None,
    current_path_list: list[str | int],
    item_key: str,
    leaf_fn: Callable[[CheckpointPath, Any, Any], Any] | None,
    missing_keys: list[list[str | int]],
) -> Sequence:
    """Walk through a Sequence (list, tuple, deque, etc.).

    Modifies mutable sequences (list, deque) in-place when provided.
    Creates new containers for immutable sequences (tuple).

    Args:
        src: Source sequence to traverse.
        tgt: Target sequence to merge into. If None, iterates over source.
        current_path_list: Current path as mutable list.
        item_key: Top-level item key for CheckpointPath.
        leaf_fn: Optional custom leaf function.
        missing_keys: List to collect missing key paths.

    Returns:
        The merged sequence (same object as tgt for mutable, new for immutable).
    """
    original_type = type(src)

    # If target is an empty sequence, return source as-is.
    # An empty template means "load everything", not "filter to nothing".
    # This handles cases like checkpoint restore where the current state has empty
    # bins but the saved checkpoint has populated bins that need to be restored.
    if tgt is not None and len(tgt) == 0:
        return _convert_to_original_type(list(src), original_type)

    if tgt is None:
        # Iterate over all source elements
        result_list = [
            _walk(
                src=src_item,
                tgt=None,
                current_path_list=current_path_list + [i],
                item_key=item_key,
                leaf_fn=leaf_fn,
                missing_keys=missing_keys,
            )
            for i, src_item in enumerate(src)
        ]
        return _convert_to_original_type(result_list, original_type)

    if isinstance(tgt, MutableSequence):
        # Modify mutable sequence in-place
        for i in range(len(tgt)):
            child_path = current_path_list + [i]
            if i < len(src):
                tgt[i] = _walk(
                    src=src[i],
                    tgt=tgt[i],
                    current_path_list=child_path,
                    item_key=item_key,
                    leaf_fn=leaf_fn,
                    missing_keys=missing_keys,
                )
            else:
                missing_keys.append(child_path.copy())
        return tgt

    # Immutable sequence (tuple, etc.) - build new container
    result_list = []
    for i in range(len(tgt)):
        child_path = current_path_list + [i]
        if i < len(src):
            result_list.append(
                _walk(
                    src=src[i],
                    tgt=tgt[i],
                    current_path_list=child_path,
                    item_key=item_key,
                    leaf_fn=leaf_fn,
                    missing_keys=missing_keys,
                )
            )
        else:
            missing_keys.append(child_path.copy())
            result_list.append(tgt[i])

    return _convert_to_original_type(result_list, original_type)


def _walk(
    *,
    src: Any,
    tgt: Any | None,
    current_path_list: list[str | int],
    item_key: str,
    leaf_fn: Callable[[CheckpointPath, Any, Any], Any] | None,
    missing_keys: list[list[str | int]],
) -> Any:
    """Recursively walk through the structure.

    Args:
        src: Source value to traverse.
        tgt: Target value to merge into.
        current_path_list: Current path as mutable list.
        item_key: Top-level item key for CheckpointPath.
        leaf_fn: Optional custom leaf function.
        missing_keys: List to collect missing key paths as lists.

    Returns:
        The merged/transformed result.
    """
    nested_path = tuple(current_path_list)

    _check_type_compatibility(src, tgt, nested_path)

    # String-like types and Sets are leaves
    if isinstance(src, (_STR_LIKE_TYPES, Set)):
        return _call_leaf(
            item_key=item_key,
            nested_path=nested_path,
            src=src,
            tgt=tgt,
            leaf_fn=leaf_fn,
        )

    # Handle Mapping types (dict, OrderedDict, etc.)
    if isinstance(src, Mapping):
        return _walk_mapping(
            src=src,
            tgt=tgt,
            current_path_list=current_path_list,
            item_key=item_key,
            leaf_fn=leaf_fn,
            missing_keys=missing_keys,
        )

    # Handle Sequence types (list, tuple, deque, etc.)
    if isinstance(src, Sequence):
        return _walk_sequence(
            src=src,
            tgt=tgt,
            current_path_list=current_path_list,
            item_key=item_key,
            leaf_fn=leaf_fn,
            missing_keys=missing_keys,
        )

    # Everything else is a leaf
    return _call_leaf(
        item_key=item_key,
        nested_path=nested_path,
        src=src,
        tgt=tgt,
        leaf_fn=leaf_fn,
    )


def walk_checkpoint_structure(
    *,
    item_key: str,
    source: Any,
    target: Any | None = None,
    leaf_fn: Callable[[CheckpointPath, Any, Any], Any] | None = None,
) -> tuple[Any, list[CheckpointPath]]:
    """
    Walk through nested structures, merging source with target and optionally
    applying a callback function to leaf values.

    Container type support (via collections.abc):
    - Mapping: dict, OrderedDict, and any Mapping subclass
    - Sequence: list, tuple, deque (excludes str/bytes which are leaves)
    - Set: treated as leaf values (no meaningful order for filtering)

    In-place modification behavior:
        When a target is provided, the following are modified IN-PLACE:
        - Mutable containers: dict, list, deque - values are updated in the
          existing container objects
        - Tensors: target tensors receive data via copy_() from source tensors,
          preserving the target tensor's identity (same object, updated data)

        The following are NOT modified in-place (new objects are created):
        - Immutable containers: tuple, frozenset - a new container is returned
          with merged values
        - Non-tensor leaf values: the source value replaces the target value

        When target is None:
        - A new structure is created based on source, no in-place modification

    Args:
        item_key: Key for constructing CheckpointPath (top-level item key)
        source: Source value to traverse (loaded data)
        target: Optional target structure to merge into. If None, structure is
                derived from source.
        leaf_fn: Optional callback for leaf values. Signature:
                 (CheckpointPath, source_value, target_value) -> result_value
                 If None, uses default behavior (tensor copy, else return source).

    Returns:
        Tuple of (merged_result, list[CheckpointPath]) where missing_paths contains
        paths that exist in target but not in source.

    Raises:
        RuntimeError: If target type is incompatible with source type at any level.
    """
    missing_keys: list[list[str | int]] = []

    result = _walk(
        src=source,
        tgt=target,
        current_path_list=[],
        item_key=item_key,
        leaf_fn=leaf_fn,
        missing_keys=missing_keys,
    )

    # Convert raw paths to CheckpointPath objects
    missing_paths: list[CheckpointPath] = [
        CheckpointPath(item_key=item_key, nested_path=tuple(path))
        for path in missing_keys
    ]

    return result, missing_paths
