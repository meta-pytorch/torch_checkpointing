"""
State Transformations for Checkpointing.

This module provides transform functions for optimizer state dictionaries.
These transforms enable format conversions during save and load operations,
converting between integer keys (PyTorch format) and string keys (readable,
reshardable format).

Example:
    from torch_checkpointing.state_transformations import (
        optimizer_transform_pre,
        optimizer_transform_post,
    )

    # Before checkpointing sees the data: convert int keys to string keys
    transformed_state = optimizer_transform_pre(optimizer_state_dict)

    # After checkpointing is done: convert string keys back to int keys
    restored_state = optimizer_transform_post(loaded_state_dict)
"""

from dataclasses import dataclass, field, fields
from typing import Any, cast

# Format version for new optimizer state format
_CURRENT_FORMAT_VERSION = 1


@dataclass
class ParamGroup:
    """A single optimizer param group with required fields for resharding.

    Attributes:
        param_names: List of parameter names corresponding to each param.
            Must be populated by the optimizer wrapper for resharding support.
        extra: Additional optimizer-specific fields (lr, betas, weight_decay, etc.)
            May contain 'params' if loaded from a v0 format checkpoint.
    """

    param_names: list[str]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParamGroup":
        """Create a ParamGroup from a dictionary."""
        known_fields = {f.name for f in fields(cls) if f.name != "extra"}
        return cls(
            **{name: d[name] for name in known_fields},
            extra={k: v for k, v in d.items() if k not in known_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert this ParamGroup to a dictionary."""
        known_fields = {f.name for f in fields(self) if f.name != "extra"}
        return {name: getattr(self, name) for name in known_fields} | self.extra


@dataclass
class OptimizerStateDict:
    """Typed wrapper for PyTorch optimizer state_dict.

    Attributes:
        state: Mapping from param index/key to per-parameter optimizer state.
        param_groups: List of parameter groups with structured access.
        _optimizer_state_format_version: Format version (0 = legacy int keys, 1 = string keys).
    """

    state: dict[int | str, dict[str, Any]]
    param_groups: list[ParamGroup]
    _optimizer_state_format_version: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizerStateDict":
        """Create an OptimizerStateDict from a raw dictionary."""
        expected_keys = {f.name for f in fields(cls)}

        unexpected = set(d.keys()) - expected_keys
        if unexpected:
            raise ValueError(f"Unexpected keys in optimizer state dict: {unexpected}")

        return cls(
            state=d["state"],
            param_groups=[ParamGroup.from_dict(pg) for pg in d["param_groups"]],
            _optimizer_state_format_version=d.get("_optimizer_state_format_version", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert this OptimizerStateDict to a raw dictionary.

        Note: The format version key is only included for v1+ formats.
        V0 format (PyTorch-native) does not have this key.
        """
        result: dict[str, Any] = {
            "state": self.state,
            "param_groups": [pg.to_dict() for pg in self.param_groups],
        }
        # Only include format version for v1+ (v0 is PyTorch-native, no version key)
        if self._optimizer_state_format_version > 0:
            result["_optimizer_state_format_version"] = (
                self._optimizer_state_format_version
            )
        return result


def optimizer_transform_pre(optimizer_state: dict[str, Any]) -> dict[str, Any]:
    """Pre-checkpoint transform for optimizer state_dict.

    Applied before the checkpointing layer processes the optimizer state.
    Converts integer keys to string keys using param_name to enable
    checkpoint resharding and debuggability.

    The keys are fully qualified names (FQNs) from model.named_parameters().

    Example:
        >>> state = {
        ...     "state": {
        ...         0: {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
        ...         1: {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
        ...     },
        ...     "param_groups": [{"params": [0, 1], "param_names": ["layers.0.weight", "layers.0.bias"], ...}],
        ... }
        >>> optimizer_transform_pre(state)
        {
            "state": {
                "layers.0.weight": {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
                "layers.0.bias": {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
            },
            "param_groups": [{"param_names": ["layers.0.weight", "layers.0.bias"], ...}],
            "_optimizer_state_format_version": 1,
        }

    Args:
        optimizer_state: The original optimizer state_dict with integer keys.

    Returns:
        A new state_dict with string keys and format version marker.
        The 'params' field is excluded from param_groups.
    """
    # Check if already in v1 format - return unchanged (idempotent)
    format_version = optimizer_state.get("_optimizer_state_format_version", 0)
    if format_version == _CURRENT_FORMAT_VERSION:
        return optimizer_state

    # Parse input using typed wrapper
    opt_state = OptimizerStateDict.from_dict(optimizer_state)

    # Build idx → param_name mapping from param_groups
    # Compute global indices on the fly instead of relying on params field
    idx_to_key: dict[int, str] = {}
    seen_keys: set[str] = set()
    current_idx = 0

    for pg in opt_state.param_groups:
        for name in pg.param_names:
            if name in seen_keys:
                raise ValueError(f"Duplicate param name detected: {name}")
            seen_keys.add(name)
            idx_to_key[current_idx] = name
            current_idx += 1

    # Build new state with string keys
    new_state: dict[int | str, dict[str, Any]] = {}
    for idx, state_val in opt_state.state.items():
        if idx not in idx_to_key:
            raise ValueError(
                f"State key {idx} not found in param_groups mapping. "
                f"Available keys: {sorted(idx_to_key.keys())}"
            )
        new_state[idx_to_key[cast(int, idx)]] = state_val

    # Remove 'params' from each param_group (it flows through extra)
    for pg in opt_state.param_groups:
        pg.extra.pop("params", None)

    return OptimizerStateDict(
        state=new_state,
        param_groups=opt_state.param_groups,
        _optimizer_state_format_version=_CURRENT_FORMAT_VERSION,
    ).to_dict()


def optimizer_transform_post(optimizer_state: dict[str, Any]) -> dict[str, Any]:
    """Post-checkpoint transform for optimizer state_dict.

    Applied after the checkpointing layer has processed the optimizer state.
    Converts string keys back to integer keys for PyTorch optimizer compatibility.

    The string keys are fully qualified names (FQNs) from model.named_parameters().

    Example:
        >>> state = {
        ...     "state": {
        ...         "layers.0.weight": {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
        ...         "layers.0.bias": {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
        ...     },
        ...     "param_groups": [{"param_names": ["layers.0.weight", "layers.0.bias"], ...}],
        ...     "_optimizer_state_format_version": 1,
        ... }
        >>> optimizer_transform_post(state)
        {
            "state": {
                0: {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
                1: {"exp_avg": tensor(...), "exp_avg_sq": tensor(...)},
            },
            "param_groups": [{"params": [0, 1], "param_names": [...], ...}],
        }

    Args:
        optimizer_state: The state_dict loaded from checkpoint.

    Returns:
        A new state_dict with integer keys suitable for PyTorch optimizer.
    """
    # Check format version (default to 0 for old format)
    format_version = optimizer_state.get("_optimizer_state_format_version", 0)

    if format_version == 0:
        # Old format (int keys) - return as-is
        return optimizer_state

    # Parse input using typed wrapper
    opt_state = OptimizerStateDict.from_dict(optimizer_state)

    # Build param_name → idx mapping from param_groups
    # The indices are sequential starting from 0, matching the order in param_names
    key_to_idx: dict[str, int] = {}
    seen_keys: set[str] = set()
    current_idx = 0

    for pg in opt_state.param_groups:
        for name in pg.param_names:
            if name in seen_keys:
                raise ValueError(f"Duplicate param name detected: {name}")
            seen_keys.add(name)
            key_to_idx[name] = current_idx
            current_idx += 1

    # Build new state with integer keys
    new_state: dict[int | str, dict[str, Any]] = {}
    for key, state_val in opt_state.state.items():
        if key not in key_to_idx:
            raise ValueError(
                f"State key '{key}' not found in param_groups. "
                f"Available keys: {sorted(key_to_idx.keys())}"
            )
        new_state[key_to_idx[cast(str, key)]] = state_val

    # Regenerate params for each param_group with sequential indices
    start_idx = 0
    for pg in opt_state.param_groups:
        num_params = len(pg.param_names)
        pg.extra["params"] = list(range(start_idx, start_idx + num_params))
        start_idx += num_params

    return OptimizerStateDict(
        state=new_state,
        param_groups=opt_state.param_groups,
    ).to_dict()
