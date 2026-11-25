# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint layout functionality for controlling the on-disk format of checkpoints.

This module provides components for defining custom checkpoint layouts that control
how different parts of a state dictionary are saved to and loaded from storage.

## Default behavior

When an item has no explicit layout, the checkpointing system writes one
Torch-serialized file for that item and rank:
- File name: `<item_key>_<rank>.pt`
- Serialization: The item is saved using `torch.save()`
- Location: The file is saved directly in the checkpoint directory

For example, items named `model` and `optimizer` on rank 0 produce
`model_0.pt` and `optimizer_0.pt`.

## Custom Layouts (checkpoint_layout=callable)

For more control, provide a layout function that splits data across multiple files:
- Different components (model, optimizer, metadata) can be split into separate files.
- Only top-level keys in the state_dict are supported
- Each state_dict key maps to its own unique file (no file sharing between keys)
- Use different serialization formats (Torch tensors vs JSON) for different data
- Control file naming and organization within the checkpoint directory
- Use a fixed shared path only when the caller arranges for one rank to write it

Example usage:
    def my_layout(rank: int) -> dict[str, LayoutInfo]:
        return {
            'model': LayoutInfo(f'model_{rank}.pt', TorchSerialization()),
            'optimizer': LayoutInfo(f'optimizer_{rank}.pt', TorchSerialization()),
            'metadata': LayoutInfo(f'config_{rank}.json', JsonSerialization(dict)),
        }

    writer = CheckpointWriter(config, rank_info, checkpoint_layout=my_layout)
    # Creates one file per configured item and rank.
"""

import abc
import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, final

logger = logging.getLogger(__name__)


class SerializationFormat(abc.ABC):
    """Base class for defining how data should be serialized to storage."""

    @final
    @staticmethod
    def supported_types() -> tuple[type["SerializationFormat"], ...]:
        """Return concrete payload formats built into this module."""
        return (
            TorchSerialization,
            JsonSerialization,
            RawSerialization,
            SafetensorsSerialization,
        )

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the serialization format to a dictionary."""
        ...

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "SerializationFormat":
        """Deserialize a serialization format from a dictionary."""
        ...


@dataclass(frozen=True)
class TorchSerialization(SerializationFormat):
    """Pytorch serialization format using torch.save/torch.load"""

    def __repr__(self) -> str:
        return "TorchSerialization()"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "TorchSerialization"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TorchSerialization":
        return cls()


@dataclass(frozen=True)
class JsonSerialization(SerializationFormat):
    """Serialization format for JSON-compatible data.

    Uses json.dump/json.load for serialization. Suitable for:
    - Configuration data
    - Metadata like epoch numbers, training steps
    - Any JSON-serializable Python objects

    Args:
        cls: The Python type to deserialize JSON data into. When ``None``
            (the default), no target type is recorded and the reader returns
            the raw JSON-decoded value (dict/list/scalar) as-is instead of
            reconstructing a typed object.
    """

    cls: type | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "JsonSerialization",
            "cls": (
                f"{self.cls.__module__}.{self.cls.__name__}"
                if self.cls is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JsonSerialization":
        cls_path = d.get("cls")
        if cls_path is None:
            return cls()
        module_name, class_name = cls_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        target_cls = getattr(module, class_name)
        return cls(cls=target_cls)


@dataclass(frozen=True)
class RawSerialization(SerializationFormat):
    """Raw serialization format that writes data as-is without any encoding.

    This is useful for data that is already in the desired format. For example,
    pre-serialized JSON strings can be encoded to bytes and written directly.

    Use cases:
    - Pre-formatted JSON strings encoded as bytes
    - Raw binary data
    """

    def __repr__(self) -> str:
        return "RawSerialization()"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "RawSerialization"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawSerialization":
        return cls()


@dataclass(frozen=True, eq=False)
class SafetensorsSerialization(SerializationFormat):
    """Safetensors format for tensor-only data (model state dicts).

    Non-tensor values are not supported and will raise an error during write.
    Nested dicts are flattened to dot-separated keys on save; on read the
    checkpoint reader re-nests them to match the caller's target structure
    (when a target is provided).
    DTensors are automatically unwrapped to their local tensor on save.

    Args:
        metadata: Optional user metadata embedded in the safetensors file. Both
            keys and values must be ``str`` (this matches the safetensors library's
            contract). Equality / hashing are order-independent over the items.

    Note:
        ``frozen=True`` prevents reassigning the ``metadata`` field but does NOT
        prevent mutating its contents. Treat the dict as read-only after
        construction; if it's mutated, the hash will silently drift.
    """

    metadata: dict[str, str] | None = None

    def __hash__(self) -> int:
        # Custom hash because dict isn't hashable. Sort items so two dicts with
        # equal contents hash equally regardless of insertion order — consistent
        # with how __eq__ compares dicts (order-independent).
        if self.metadata is None:
            return hash(("SafetensorsSerialization", None))
        return hash(("SafetensorsSerialization", tuple(sorted(self.metadata.items()))))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SafetensorsSerialization):
            return NotImplemented
        return self.metadata == other.metadata

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "SafetensorsSerialization"}
        if self.metadata is not None:
            # Defensive copy so the on-disk representation is decoupled from any
            # post-hoc mutation of self.metadata.
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SafetensorsSerialization":
        raw = d.get("metadata")
        return cls(metadata=dict(raw)) if raw is not None else cls()

    @staticmethod
    def prepare_tensors_for_save(data: dict[str, Any]) -> dict[str, Any]:
        """Recursively flatten a nested state_dict to dot-separated keys with tensor values.

        Supported container shapes (composable to any depth):
        - dict: key segments joined with ``.``
        - list / tuple: stringified index segments joined with ``.``
        - Tensor leaf: unwrapped via ``DTensor._local_tensor`` if applicable, forced
          contiguous, stored at the joined path
        - Anything else: raises (safetensors only stores tensors)

        Mirrors ``unflatten_to_target`` so a value written here round-trips back into
        its original shape on read.

        Args:
            data: A nested state_dict whose leaves must be tensors.

        Returns:
            A flat ``dict[str, Tensor]`` suitable for ``safetensors.torch.save()``.

        Raises:
            ValueError: If any leaf value is not a ``torch.Tensor``.
        """
        import torch
        from torch.distributed.tensor import DTensor

        flat: dict[str, Any] = {}

        def _walk(value: Any, full_key: str) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    _walk(v, f"{full_key}.{k}" if full_key else str(k))
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    _walk(v, f"{full_key}.{i}" if full_key else str(i))
            elif isinstance(value, torch.Tensor):
                tensor = value
                if isinstance(tensor, DTensor):
                    tensor = tensor._local_tensor
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                flat[full_key] = tensor
            else:
                raise ValueError(
                    f"SafetensorsSerialization only supports tensor values, but key "
                    f"'{full_key}' has type '{type(value).__name__}'. Use "
                    f"TorchSerialization or JsonSerialization for non-tensor data."
                )

        _walk(data, "")
        return flat

    @staticmethod
    def unflatten_to_target(
        flat: dict[str, Any],
        target: Any,
    ) -> Any:
        """Reshape a flat dotted-key dict (from ``safetensors_load``) to match target's nesting.

        Walks the target's structure; for each leaf path, joins the path with ``.`` and
        looks the joined key up in ``flat``. Leaves missing from ``flat`` are dropped
        from dict parents — the downstream ``walk_checkpoint_structure`` then correctly
        reports them as missing keys instead of silently aligning a flat source with a
        nested target. For list/tuple parents, missing leaves come back as ``None``
        (positional containers can't have holes).
        """

        def _walk(t: Any, prefix: str) -> Any:
            if isinstance(t, dict):
                out: dict[Any, Any] = {}
                for k, v in t.items():
                    sub = f"{prefix}.{k}" if prefix else str(k)
                    if isinstance(v, (dict, list, tuple)):
                        out[k] = _walk(v, sub)
                    elif sub in flat:
                        out[k] = flat[sub]
                    # else: leaf missing; drop so walk_checkpoint_structure reports it
                return out
            if isinstance(t, (list, tuple)):
                return type(t)(
                    _walk(v, f"{prefix}.{i}" if prefix else str(i))
                    for i, v in enumerate(t)
                )
            # Leaf-only target (recursion landed on a non-container element).
            return flat.get(prefix)

        return _walk(target, "")


@dataclass(frozen=True)
class LayoutInfo:
    """Information about how a specific state dict key should be stored.

    Args:
        file_path: Exact path to the file relative to the checkpoint directory.
                   It takes precedence over the default per-rank layout:

                   Examples:
                   - "model_{rank}.pt" -> a file per rank after materialization
                   - "rank_0/model.pt" -> saves to checkpoint_dir/rank_0/model.pt
                   - f"model_rank_{rank}.pt" -> per-rank files

                   A fixed path such as "model.pt" requires the caller to
                   arrange for only one rank to write it.

        serialization_format: How the data should be serialized
        num_parts: Number of files the item is split into, so that
                   ``file_write_max_threads`` threads can write one large item
                   in parallel. Only ``TorchSerialization`` items that are
                   dicts can be split; see ``split_into_parts``.

                   - 1 -> the item is written to ``file_path``
                   - 3 with "optimizer_0.pt" -> "optimizer_0.part0.pt",
                     "optimizer_0.part1.pt" and "optimizer_0.part2.pt"
    """

    file_path: str
    serialization_format: SerializationFormat
    num_parts: int = 1

    def __post_init__(self) -> None:
        if self.num_parts < 1:
            raise ValueError(f"num_parts must be at least 1, got {self.num_parts}")
        if self.num_parts > 1 and not isinstance(
            self.serialization_format, TorchSerialization
        ):
            raise ValueError(
                "num_parts > 1 requires TorchSerialization, got "
                f"{self.serialization_format!r} for {self.file_path!r}"
            )

    def part_file_paths(self) -> list[str]:
        """Paths of the files that hold this item, relative to the checkpoint."""
        if self.num_parts == 1:
            return [self.file_path]
        path = PurePosixPath(self.file_path)
        return [
            str(path.with_name(f"{path.stem}.part{part_index}{path.suffix}"))
            for part_index in range(self.num_parts)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the layout info to a dictionary."""
        d: dict[str, Any] = {
            "file_path": self.file_path,
            "serialization_format": self.serialization_format.to_dict(),
        }
        # Omitted for one part, so unsplit layouts serialize as before.
        if self.num_parts > 1:
            d["num_parts"] = self.num_parts
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayoutInfo":
        """Deserialize a layout info from a dictionary."""
        return cls(
            file_path=d["file_path"],
            serialization_format=serialization_format_from_dict(
                d["serialization_format"]
            ),
            num_parts=d.get("num_parts", 1),
        )


def split_into_parts(value: Any, num_parts: int) -> list[dict[Any, Any]]:
    """Split a nested dict into ``num_parts`` dicts of about equal tensor bytes.

    A leaf is any value that is not a non-empty dict. Largest storage first, each
    storage's leaves go to the part with the fewest bytes so far. Leaves without
    tensor data (non-tensors, empty or meta tensors) count as 0 bytes.

    Example:

        split_into_parts(
            {"exp_avg": {"w": tensor_8mb, "b": tensor_1mb}, "step": 3}, num_parts=2
        )
        # -> [{"exp_avg": {"w": tensor_8mb}}, {"exp_avg": {"b": tensor_1mb}, "step": 3}]

        # Views of one storage stay in one part: torch.save writes the whole storage.
        split_into_parts({"a": buffer[:4], "b": buffer[4:]}, num_parts=2)
        # -> [{"a": buffer[:4], "b": buffer[4:]}, {}]

    ``merge_parts`` reverses it.
    """
    import torch
    from torch.distributed.tensor import DTensor

    from ._tensor_utils import has_storage_data

    if not isinstance(value, Mapping):
        raise ValueError(
            f"Only a dict can be split into parts, got {type(value).__name__}"
        )

    leaves: list[tuple[tuple[Any, ...], Any]] = []

    def collect(prefix: tuple[Any, ...], mapping: Mapping[Any, Any]) -> None:
        for key, child in mapping.items():
            if isinstance(child, Mapping) and child:
                collect((*prefix, key), child)
            else:
                leaves.append(((*prefix, key), child))

    collect((), value)

    # Group the leaves by storage, weighed by the storage's bytes. Views of one
    # storage in two parts would write it twice. Other leaves are groups of one.
    leaves_of_group: dict[Any, list[int]] = {}
    group_bytes: dict[Any, int] = {}
    for leaf_index, (_, leaf) in enumerate(leaves):
        tensor = leaf._local_tensor if isinstance(leaf, DTensor) else leaf
        if isinstance(tensor, torch.Tensor) and has_storage_data(tensor):
            storage = tensor.untyped_storage()
            group = (tensor.device, storage.data_ptr())
            group_bytes[group] = storage.nbytes()
        else:
            group = ("leaf", leaf_index)
            group_bytes[group] = 0
        leaves_of_group.setdefault(group, []).append(leaf_index)

    # Place the largest groups first, each on the part with the fewest bytes.
    part_bytes = [0] * num_parts
    part_of_leaf: list[int] = [0] * len(leaves)
    for group in sorted(leaves_of_group, key=group_bytes.__getitem__, reverse=True):
        part_index = min(range(num_parts), key=part_bytes.__getitem__)
        for leaf_index in leaves_of_group[group]:
            part_of_leaf[leaf_index] = part_index
        part_bytes[part_index] += group_bytes[group]

    # Rebuild each part in the original key order.
    parts: list[dict[Any, Any]] = [{} for _ in range(num_parts)]
    for (path, leaf), part_index in zip(leaves, part_of_leaf, strict=True):
        node = parts[part_index]
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = leaf
    return parts


def merge_parts(parts: list[Any]) -> dict[Any, Any]:
    """Deep-merge the dicts ``split_into_parts`` returned back into one dict."""
    merged: dict[Any, Any] = {}

    def merge_into(target: dict[Any, Any], source: Mapping[Any, Any]) -> None:
        for key, child in source.items():
            if isinstance(child, Mapping) and isinstance(target.get(key), dict):
                merge_into(target[key], child)
            else:
                target[key] = child

    for part in parts:
        merge_into(merged, part)
    return merged


def default_torch_layout_info(key: str, rank: int) -> LayoutInfo:
    """Default per-rank Torch serialization layout for a checkpoint item.

    Args:
        key: The checkpoint item key.
        rank: The rank whose file is being named.

    Returns:
        The ``<key>_<rank>.pt`` layout used by the native Torch writer.
    """
    return LayoutInfo(
        f"{key}_{rank}.pt",
        TorchSerialization(),
    )


def serialization_format_from_dict(d: dict[str, Any]) -> SerializationFormat:
    """Factory function to deserialize a SerializationFormat from a dictionary."""
    type_name = d.get("type")
    format_type = next(
        (
            format_type
            for format_type in SerializationFormat.supported_types()
            if format_type.__name__ == type_name
        ),
        None,
    )
    if format_type is None:
        raise ValueError(f"Unknown SerializationFormat type: {type_name}")
    return format_type.from_dict(d)
