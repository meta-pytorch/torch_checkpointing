# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for state transformations module."""

import pytest
import torch
import torch.nn as nn
from torch_checkpointing.state_transformations import (
    _CURRENT_FORMAT_VERSION,
    optimizer_transform_post,
    optimizer_transform_pre,
    OptimizerStateDict,
    ParamGroup,
)


def _create_optimizer_state_with_param_names(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict:
    """Create optimizer state_dict with param_names field added.

    PyTorch's standard optimizer state_dict doesn't include param_names.
    This helper adds param_names as a custom optimizer wrapper would.

    Args:
        model: The model whose parameters the optimizer is managing.
        optimizer: The optimizer to get state_dict from.

    Returns:
        Optimizer state_dict with param_names field added to each param group.
    """
    state_dict = optimizer.state_dict()

    # Build mapping from param id to param name
    param_id_to_name = {id(p): name for name, p in model.named_parameters()}

    # Add param_names to each param group based on the original param objects
    for pg_state, pg_orig in zip(
        state_dict["param_groups"], optimizer.param_groups, strict=True
    ):
        pg_state["param_names"] = [param_id_to_name[id(p)] for p in pg_orig["params"]]

    return state_dict


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_optimizer_state() -> dict:
    """Creates a simple optimizer state dict using a real PyTorch optimizer.

    Returns:
        Optimizer state_dict from a real Adam optimizer with param_names added,
        containing two parameters (weight, bias) in a single param group.
    """
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Do a forward/backward pass to populate optimizer state
    loss = model(torch.randn(1, 4)).sum()
    loss.backward()
    optimizer.step()

    return _create_optimizer_state_with_param_names(model, optimizer)


@pytest.fixture
def multi_group_optimizer_state() -> dict:
    """Creates an optimizer state dict with two param groups using real PyTorch optimizer.

    Returns:
        Optimizer state_dict from a real Adam optimizer with param_names added,
        containing two param groups: layer0 (2 params) and layer1 (2 params).
    """
    model = nn.Sequential(
        nn.Linear(4, 4),  # layer 0
        nn.Linear(4, 2),  # layer 1
    )

    # Create optimizer with separate param groups for each layer
    optimizer = torch.optim.Adam(
        [
            {"params": model[0].parameters(), "lr": 0.01},
            {"params": model[1].parameters(), "lr": 0.001},
        ]
    )

    # Do a forward/backward pass to populate optimizer state
    loss = model(torch.randn(1, 4)).sum()
    loss.backward()
    optimizer.step()

    return _create_optimizer_state_with_param_names(model, optimizer)


# =============================================================================
# ParamGroup Tests
# =============================================================================


def test_param_group_from_dict():
    """Tests that ParamGroup.from_dict correctly parses a dict."""
    d = {
        "params": [0, 1],
        "param_names": ["weight", "bias"],
        "lr": 0.01,
        "weight_decay": 0.001,
    }

    pg = ParamGroup.from_dict(d)

    assert pg.param_names == ["weight", "bias"]
    # params is stored in extra (not a separate field)
    assert pg.extra == {"params": [0, 1], "lr": 0.01, "weight_decay": 0.001}


def test_param_group_to_dict():
    """Tests that ParamGroup.to_dict correctly produces a dict."""
    pg = ParamGroup(
        param_names=["weight", "bias"],
        extra={"lr": 0.01, "betas": (0.9, 0.999)},
    )

    d = pg.to_dict()

    # params is NOT regenerated in to_dict - it flows through extra
    # The transform functions handle params regeneration
    assert d == {
        "param_names": ["weight", "bias"],
        "lr": 0.01,
        "betas": (0.9, 0.999),
    }


def test_param_group_to_dict_with_params_in_extra():
    """Tests that ParamGroup.to_dict includes params when in extra."""
    pg = ParamGroup(
        param_names=["weight", "bias"],
        extra={"params": [5, 6], "lr": 0.01},
    )

    d = pg.to_dict()

    # params flows through extra
    assert d["params"] == [5, 6]
    assert d["param_names"] == ["weight", "bias"]
    assert d["lr"] == 0.01


def test_param_group_round_trip():
    """Tests that ParamGroup round-trips correctly through dict."""
    original = {
        "params": [0, 1, 2],
        "param_names": ["a", "b", "c"],
        "lr": 0.001,
        "eps": 1e-8,
    }

    pg = ParamGroup.from_dict(original)
    result = pg.to_dict()

    # params flows through extra, so round-trip preserves it
    assert result == original


def test_param_group_from_dict_without_params():
    """Tests that ParamGroup.from_dict works without params (v1 format)."""
    # In v1 format, params is not saved to disk
    d = {
        "param_names": ["weight", "bias"],
        "lr": 0.01,
    }

    pg = ParamGroup.from_dict(d)

    assert pg.param_names == ["weight", "bias"]
    assert pg.extra == {"lr": 0.01}
    # params is NOT in extra since input didn't have it
    # params is regenerated by optimizer_transform_post, not by to_dict
    assert "params" not in pg.to_dict()


# =============================================================================
# OptimizerStateDict Tests
# =============================================================================


def test_optimizer_state_dict_from_dict(simple_optimizer_state):
    """Tests that OptimizerStateDict.from_dict correctly parses a dict."""
    # Add format version key (required by from_dict)
    simple_optimizer_state["_optimizer_state_format_version"] = 1
    opt = OptimizerStateDict.from_dict(simple_optimizer_state)

    assert len(opt.param_groups) == 1
    assert opt.param_groups[0].param_names == ["weight", "bias"]
    assert opt.param_groups[0].extra["lr"] == 0.001
    assert opt._optimizer_state_format_version == 1


def test_optimizer_state_dict_from_dict_unexpected_key():
    """Tests that OptimizerStateDict.from_dict raises on unexpected keys."""
    state = {
        "state": {},
        "param_groups": [],
        "unexpected_key": "value",
    }

    with pytest.raises(ValueError, match="Unexpected keys"):
        OptimizerStateDict.from_dict(state)


def test_optimizer_state_dict_to_dict(simple_optimizer_state):
    """Tests that OptimizerStateDict.to_dict correctly produces a dict."""
    # Add format version key (required by from_dict)
    simple_optimizer_state["_optimizer_state_format_version"] = 0
    opt = OptimizerStateDict.from_dict(simple_optimizer_state)
    result = opt.to_dict()

    assert result["state"] == simple_optimizer_state["state"]
    assert result["param_groups"] == simple_optimizer_state["param_groups"]
    # V0 format should not include format version key (PyTorch-native format)
    assert "_optimizer_state_format_version" not in result


def test_optimizer_state_dict_v1_includes_version():
    """Tests that OptimizerStateDict with v1 format includes version key."""
    opt = OptimizerStateDict(
        state={"0.weight": {"step": 1}},
        param_groups=[ParamGroup(param_names=["weight"])],
        _optimizer_state_format_version=1,
    )

    result = opt.to_dict()

    assert "_optimizer_state_format_version" in result
    assert result["_optimizer_state_format_version"] == 1


def test_optimizer_state_dict_to_dict_preserves_params_from_extra():
    """Tests that to_dict preserves params from extra if present."""
    opt = OptimizerStateDict(
        state={"weight": {"step": 1}},
        param_groups=[
            ParamGroup(
                param_names=["weight", "bias"], extra={"params": [0, 1], "lr": 0.01}
            ),
            ParamGroup(
                param_names=["layer2.weight"], extra={"params": [2], "lr": 0.001}
            ),
        ],
        _optimizer_state_format_version=1,
    )

    result = opt.to_dict()

    # params flows through extra
    assert result["param_groups"][0]["params"] == [0, 1]
    assert result["param_groups"][1]["params"] == [2]


# =============================================================================
# optimizer_transform_pre Tests
# =============================================================================


def test_optimizer_transform_pre_adds_format_version(simple_optimizer_state):
    """Tests that optimizer_transform_pre includes format version marker."""
    result = optimizer_transform_pre(simple_optimizer_state)

    assert "_optimizer_state_format_version" in result
    assert result["_optimizer_state_format_version"] == _CURRENT_FORMAT_VERSION


def test_optimizer_transform_pre_uses_string_keys(simple_optimizer_state):
    """Tests that optimizer_transform_pre converts int keys to string keys."""
    result = optimizer_transform_pre(simple_optimizer_state)

    # All state keys should be strings
    for key in result["state"].keys():
        assert isinstance(key, str), f"Expected string key, got {type(key)}: {key}"

    # Keys should be param names directly
    expected_keys = {"weight", "bias"}
    assert set(result["state"].keys()) == expected_keys


def test_optimizer_transform_pre_excludes_params(simple_optimizer_state):
    """Tests that optimizer_transform_pre does not save params to disk."""
    result = optimizer_transform_pre(simple_optimizer_state)

    # params should NOT be in the output param_groups (not saved to disk)
    for pg in result["param_groups"]:
        assert "params" not in pg, "params should not be saved to disk in v1 format"
        # but param_names should still be present
        assert "param_names" in pg


def test_optimizer_transform_pre_multi_param_groups(multi_group_optimizer_state):
    """Tests that multi-param-group optimizers use param names as keys."""
    result = optimizer_transform_pre(multi_group_optimizer_state)

    # Keys should be param names (e.g., "0.weight", "0.bias", "1.weight", "1.bias")
    # These are the param names from nn.Sequential with two Linear layers
    state_keys = set(result["state"].keys())
    assert len(state_keys) == 4
    for key in state_keys:
        assert isinstance(key, str)


def test_optimizer_transform_pre_missing_param_names_raises_error():
    """Tests that missing param_names raises KeyError."""
    state = {
        "state": {0: {"step": 1}},
        "param_groups": [{"params": [0]}],  # No param_names
    }

    with pytest.raises(KeyError, match="param_names"):
        optimizer_transform_pre(state)


def test_optimizer_transform_pre_state_key_not_found_raises():
    """Tests that state key not found in param_groups raises error."""
    state = {
        "state": {0: {"step": 1}, 1: {"step": 1}, 2: {"step": 1}},  # 3 state entries
        "param_groups": [
            {
                "params": [0, 1],
                "param_names": ["weight", "bias"],  # Only 2 names
            }
        ],
    }

    with pytest.raises(ValueError, match="State key 2 not found"):
        optimizer_transform_pre(state)


def test_optimizer_transform_pre_invalid_state_dict_raises_error():
    """Tests that invalid optimizer state dict raises KeyError."""
    # Missing 'state' key
    state1 = {"param_groups": []}
    with pytest.raises(KeyError):
        optimizer_transform_pre(state1)

    # Missing 'param_groups' key
    state2 = {"state": {}}
    with pytest.raises(KeyError):
        optimizer_transform_pre(state2)


def test_optimizer_transform_pre_preserves_state_values(simple_optimizer_state):
    """Tests that state values are preserved through transformation."""
    original_state_0 = simple_optimizer_state["state"][0]
    original_state_1 = simple_optimizer_state["state"][1]

    result = optimizer_transform_pre(simple_optimizer_state)

    # Values should be the same objects (not copies)
    assert result["state"]["weight"] is original_state_0
    assert result["state"]["bias"] is original_state_1


# =============================================================================
# optimizer_transform_post Tests
# =============================================================================


def test_optimizer_transform_post_v0_format_unchanged(simple_optimizer_state):
    """Tests that old format (v0, int keys) is returned unchanged."""
    # V0 format has no format version key, so it defaults to 0
    result = optimizer_transform_post(simple_optimizer_state)

    # State should be unchanged (int keys preserved)
    assert set(result["state"].keys()) == {0, 1}


def test_optimizer_transform_post_v1_format_converts_to_int_keys(
    simple_optimizer_state,
):
    """Tests that new format (v1, string keys) is converted to int keys."""
    # Create v1 format state dict
    v1_state = optimizer_transform_pre(simple_optimizer_state)
    assert "_optimizer_state_format_version" in v1_state

    result = optimizer_transform_post(v1_state)

    # State should have int keys
    assert all(isinstance(k, int) for k in result["state"].keys())
    assert set(result["state"].keys()) == {0, 1}
    # Format version key should be removed (PyTorch-native format)
    assert "_optimizer_state_format_version" not in result


def test_optimizer_transform_post_regenerates_params():
    """Tests that optimizer_transform_post regenerates params on the fly."""
    # Simulate v1 format loaded from disk (no params field)
    v1_state = {
        "state": {
            "layer.weight": {"step": 1},
            "layer.bias": {"step": 2},
        },
        "param_groups": [
            {
                "param_names": ["layer.weight", "layer.bias"],
                "lr": 0.001,
            }
        ],
        "_optimizer_state_format_version": 1,
    }

    result = optimizer_transform_post(v1_state)

    # params should be regenerated as [0, 1]
    assert result["param_groups"][0]["params"] == [0, 1]
    # State should have int keys
    assert set(result["state"].keys()) == {0, 1}


def test_optimizer_transform_post_regenerates_params_multi_group():
    """Tests that optimizer_transform_post regenerates params for multi-group optimizer."""
    # Simulate v1 format with multiple param groups (no params field)
    v1_state = {
        "state": {
            "0.weight": {"step": 1},
            "0.bias": {"step": 2},
            "1.weight": {"step": 3},
            "1.bias": {"step": 4},
        },
        "param_groups": [
            {
                "param_names": ["0.weight", "0.bias"],
                "lr": 0.01,
            },
            {
                "param_names": ["1.weight", "1.bias"],
                "lr": 0.001,
            },
        ],
        "_optimizer_state_format_version": 1,
    }

    result = optimizer_transform_post(v1_state)

    # params should be regenerated sequentially across groups
    assert result["param_groups"][0]["params"] == [0, 1]
    assert result["param_groups"][1]["params"] == [2, 3]
    # State should have int keys
    assert set(result["state"].keys()) == {0, 1, 2, 3}


def test_optimizer_transform_post_invalid_state_dict_v0_format_unchanged():
    """Tests that v0 format state dict is returned unchanged even if incomplete."""
    # V1 format (no format version key) - return as-is without validation
    state = {"state": {}, "extra_key": "ignored"}
    result = optimizer_transform_post(state)
    assert result is state


# =============================================================================
# Round-trip Tests
# =============================================================================


def test_round_trip_single_param_group(simple_optimizer_state):
    """Tests that single param group optimizer can round-trip correctly."""
    # pre_save → post_load should return equivalent state
    saved = optimizer_transform_pre(simple_optimizer_state)
    loaded = optimizer_transform_post(saved)

    # Compare state contents
    assert set(loaded["state"].keys()) == set(simple_optimizer_state["state"].keys())
    for key in simple_optimizer_state["state"]:
        assert loaded["state"][key] == simple_optimizer_state["state"][key]


def test_round_trip_multi_param_group(multi_group_optimizer_state):
    """Tests that multi-param-group optimizer can round-trip correctly."""
    saved = optimizer_transform_pre(multi_group_optimizer_state)
    loaded = optimizer_transform_post(saved)

    # Compare state contents
    assert set(loaded["state"].keys()) == set(
        multi_group_optimizer_state["state"].keys()
    )
    for key in multi_group_optimizer_state["state"]:
        assert loaded["state"][key] == multi_group_optimizer_state["state"][key]


def test_round_trip_preserves_param_groups(simple_optimizer_state):
    """Tests that param_groups metadata are preserved through round-trip."""
    simple_optimizer_state["param_groups"][0]["lr"] = 0.01
    simple_optimizer_state["param_groups"][0]["weight_decay"] = 0.001

    saved = optimizer_transform_pre(simple_optimizer_state)
    loaded = optimizer_transform_post(saved)

    # params is regenerated on load, so compare all fields
    original_pg = simple_optimizer_state["param_groups"][0]
    loaded_pg = loaded["param_groups"][0]

    # params should be regenerated correctly
    assert loaded_pg["params"] == original_pg["params"]
    # param_names should be preserved
    assert loaded_pg["param_names"] == original_pg["param_names"]
    # Other metadata should be preserved
    assert loaded_pg["lr"] == original_pg["lr"]
    assert loaded_pg["weight_decay"] == original_pg["weight_decay"]
