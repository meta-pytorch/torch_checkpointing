# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch_checkpointing.types import CheckpointPath, RankInfo, STATE_DICT


def test_rank_info_initialization():
    """Test that RankInfo initializes correctly with all parameters."""
    rank_info = RankInfo(
        global_rank=0,
        global_world_size=4,
        role_rank=0,
        role_world_size=4,
    )

    assert rank_info.global_rank == 0
    assert rank_info.global_world_size == 4


def test_rank_info_default_initialization():
    """Test that RankInfo initializes correctly with default parameters."""
    rank_info = RankInfo(
        global_rank=0,
        global_world_size=1,
        role_rank=0,
        role_world_size=1,
    )

    assert rank_info.global_rank == 0
    assert rank_info.global_world_size == 1


def test_state_dict_type_alias() -> None:
    """Test that STATE_DICT type alias works correctly."""
    state_dict = {"model": {"weight": [1, 2, 3]}, "optimizer": {"lr": 0.01}}

    state_dict_var: STATE_DICT = state_dict
    assert state_dict_var == state_dict


def test_checkpoint_path_leaf():
    """Test CheckpointPath for leaf values (no nested path)."""
    path = CheckpointPath("step")

    assert path.item_key == "step"
    assert path.nested_path == ()
    assert str(path) == "step"


def test_checkpoint_path_nested():
    """Test CheckpointPath for nested values."""
    path = CheckpointPath("model", ("encoder", "layer1", "weight"))

    assert path.item_key == "model"
    assert path.nested_path == ("encoder", "layer1", "weight")
    assert str(path) == "model::encoder.layer1.weight"


def test_checkpoint_path_mixed_nested():
    """Test CheckpointPath with mixed string/int nested path."""
    path = CheckpointPath("optimizer", ("state", 0, "exp_avg"))

    assert path.item_key == "optimizer"
    assert path.nested_path == ("state", 0, "exp_avg")
    assert str(path) == "optimizer::state.0.exp_avg"


def test_checkpoint_path_serialization():
    """Test CheckpointPath serialize and deserialize."""
    # Test leaf path
    leaf_path = CheckpointPath("step")
    leaf_str = leaf_path.serialize()
    assert leaf_str == '["step"]'
    assert CheckpointPath.deserialize(leaf_str) == leaf_path

    # Test nested path
    nested_path = CheckpointPath("model", ("encoder", "weight"))
    nested_str = nested_path.serialize()
    assert nested_str == '["model","encoder","weight"]'
    assert CheckpointPath.deserialize(nested_str) == nested_path

    # Test mixed nested path with int
    mixed_path = CheckpointPath("optimizer", ("state", 0, "exp_avg"))
    mixed_str = mixed_path.serialize()
    assert mixed_str == '["optimizer","state",0,"exp_avg"]'
    assert CheckpointPath.deserialize(mixed_str) == mixed_path


def test_checkpoint_path_post_init_validation():
    """Test CheckpointPath valid nested_path values."""
    # Valid cases - all should work without raising
    CheckpointPath("key")  # Default empty tuple is valid
    CheckpointPath("key", ())  # Explicit empty tuple is valid (leaf value)
    CheckpointPath("key", ("a",))  # Non-empty tuple is valid
    CheckpointPath("key", ("a", "b", 0))  # Multi-element tuple is valid
