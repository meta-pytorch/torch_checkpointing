# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint layout functionality for controlling the on-disk format of checkpoints.

This module provides components for defining custom checkpoint layouts that control
how different parts of a state dictionary are saved to and loaded from storage.

## Default Behavior (checkpoint_layout=None)

When no layout is specified, the checkpointing system uses a simple single-file schema:
- File name: `checkpoint_{rank}.pt` (where {rank} is the global rank)
- Serialization: The entire state_dict is saved using `torch.save()`
- Location: Saved directly in the checkpoint directory

Example:
    writer = CheckpointWriter(config, rank_info)  # checkpoint_layout=None (default)
    # Creates: /checkpoint_path/checkpoint_0.pt, /checkpoint_path/checkpoint_1.pt, etc.

## Custom Layouts (checkpoint_layout=callable)

For more control, provide a layout function that splits data across multiple files:
- Different components (model, optimizer, metadata) can be split into separate files.
- Only top-level keys in the state_dict are supported
- Each state_dict key maps to its own unique file (no file sharing between keys)
- Use different serialization formats (Torch tensors vs JSON) for different data
- Control file naming and organization within the checkpoint directory
- Support both per-rank and global files in distributed settings

Example usage:
    def my_layout(rank: int) -> dict[str, LayoutInfo]:
        return {
            'model': LayoutInfo('model.pt', TorchSerialization()),
            'optimizer': LayoutInfo(f'optimizer_{rank}.pt', TorchSerialization()),
            'metadata': LayoutInfo('config.json', JsonSerialization(dict)),
        }

    writer = CheckpointWriter(config, rank_info, checkpoint_layout=my_layout)
    # Creates: /checkpoint_path/model.pt, /checkpoint_path/optimizer_rank_0.pt, etc.
"""

import abc
import importlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class SerializationFormat(abc.ABC):
    """Base class for defining how data should be serialized to storage."""

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
        file_path: Path to the file relative to the checkpoint directory.
                   This gives you full control over file naming and organization:

                   Examples:
                   - "model.pt" -> saves to checkpoint_dir/model.pt
                   - "rank_0/model.pt" -> saves to checkpoint_dir/rank_0/model.pt
                   - f"model_rank_{rank}.pt" -> per-rank files
                   - "shared/config.json" -> global file in subdirectory

        serialization_format: How the data should be serialized
    """

    file_path: str
    serialization_format: SerializationFormat

    def to_dict(self) -> dict[str, Any]:
        """Serialize the layout info to a dictionary."""
        return {
            "file_path": self.file_path,
            "serialization_format": self.serialization_format.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayoutInfo":
        """Deserialize a layout info from a dictionary."""
        return cls(
            file_path=d["file_path"],
            serialization_format=serialization_format_from_dict(
                d["serialization_format"]
            ),
        )


def default_layout_info(key: str, rank: int) -> LayoutInfo:
    """
    Default layout info for a key and rank.

    Args:
        key (str): The key to use in the layout.
        rank (int): The rank to use in the layout.

    Returns:
        LayoutInfo: The layout info for the key and rank.
    """
    return LayoutInfo(
        f"{key}_{rank}.pt",
        TorchSerialization(),
    )


def serialization_format_from_dict(d: dict[str, Any]) -> SerializationFormat:
    """Factory function to deserialize a SerializationFormat from a dictionary."""
    type_name = d.get("type")
    if type_name == "TorchSerialization":
        return TorchSerialization.from_dict(d)
    elif type_name == "JsonSerialization":
        return JsonSerialization.from_dict(d)
    elif type_name == "RawSerialization":
        return RawSerialization.from_dict(d)
    elif type_name == "SafetensorsSerialization":
        return SafetensorsSerialization.from_dict(d)
    else:
        raise ValueError(f"Unknown SerializationFormat type: {type_name}")
