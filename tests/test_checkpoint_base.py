# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch_checkpointing.checkpoint_base import CheckpointInfo, CheckpointItem


@pytest.fixture
def valid_keys():
    """All types of valid checkpoint keys."""
    return [
        # Basic alphanumeric
        "model",
        "optimizer",
        "scheduler",
        # With hyphens
        "model-state",
        "optimizer-state",
        # With underscores
        "model_state",
        "optimizer_state",
        # Mixed
        "model-v2_state",
        # Single character
        "a",
        "Z",
        "0",
        "_",
        "-",
        # Long keys
        "very_long_model_checkpoint_state_v2_final",
        # Starting/ending with special
        "_model",
        "model_",
        "-optimizer",
        "optimizer-",
    ]


@pytest.fixture
def invalid_keys():
    """All types of invalid checkpoint keys."""
    return [
        # Path separators
        "model/state",
        "model.state",
        # Whitespace
        "model space",
        " model",
        "model ",
        # Special characters
        "model@state",
        "model#state",
        "model$state",
        "model%state",
        "model&state",
        "model*state",
        # File extensions
        "model.pt",
        "checkpoint.pth",
        "metadata.json",
        # Empty string
        "",
    ]


def test_multiple_valid_keys(valid_keys):
    """Test that multiple valid keys are accepted."""
    checkpoint_items = {key: CheckpointItem(value=None) for key in valid_keys}

    checkpoint_info = CheckpointInfo(checkpoint_items=checkpoint_items)

    assert set(checkpoint_info.keys) == set(valid_keys)


def test_multiple_invalid_keys(invalid_keys):
    """Test that multiple invalid keys are rejected and all reported in error message."""
    checkpoint_items = {key: CheckpointItem(value=None) for key in invalid_keys}

    with pytest.raises(ValueError) as exc_info:
        CheckpointInfo(checkpoint_items=checkpoint_items)

    error_message = str(exc_info.value)
    # Verify that all invalid keys are mentioned in the error message
    for key in invalid_keys:
        if key:  # Skip empty string check
            assert key in error_message


def test_mixed_valid_and_invalid_keys(valid_keys, invalid_keys):
    """Test that with mixed keys, only invalid keys are rejected and reported."""
    # Use subset of keys for clearer test
    test_valid_keys = valid_keys[:3]
    test_invalid_keys = invalid_keys[:3]
    all_keys = {
        **{k: CheckpointItem(value=None) for k in test_valid_keys},
        **{k: CheckpointItem(value=None) for k in test_invalid_keys},
    }

    with pytest.raises(ValueError) as exc_info:
        CheckpointInfo(checkpoint_items=all_keys)

    error_message = str(exc_info.value)
    # Verify that invalid keys are mentioned
    for key in test_invalid_keys:
        assert key in error_message
