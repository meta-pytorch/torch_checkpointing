# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for CheckpointLoader class.
"""

import os
import tempfile
from unittest import mock

import pytest
import torch
from torch_checkpointing import (
    CheckpointLoader,
    CheckpointReader,
    make_async_checkpoint_saver,
    make_sync_checkpoint_saver,
)
from torch_checkpointing.checkpoint_base import (
    CheckpointBase,
    CheckpointInfo,
    CheckpointItem,
)
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.metadata_manager import DefaultMetadataManager
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo


class SimpleCheckpoint(CheckpointBase):
    """Simple checkpoint wrapper for testing that wraps a state dict."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def get_items(self):
        return {
            key: CheckpointItem(value=value) for key, value in self._state_dict.items()
        }

    def load_state_dict(self, state_dict):
        self._state_dict.update(state_dict)


class ReshardableCheckpoint(SimpleCheckpoint):
    """SimpleCheckpoint whose items all declare a DefaultResharder."""

    def get_items(self):
        return {
            key: CheckpointItem(value=value, resharder=DefaultResharder())
            for key, value in self._state_dict.items()
        }


@pytest.fixture
def rank_info():
    return RankInfo(
        global_world_size=1,
        global_rank=0,
        role_rank=0,
        role_world_size=1,
    )


@pytest.fixture
def sample_state_dict():
    return {
        "model": {"weight": torch.randn(10, 10), "bias": torch.randn(10)},
        "optimizer": {"step": torch.tensor(100)},
    }


def _make_loader(rank_info, metadata_manager=None):
    """Helper to construct a CheckpointLoader with a CheckpointReader."""
    reader = CheckpointReader(
        rank_info=rank_info,
        storage_config=LocalFileSystemStorageConfig(),
    )
    return CheckpointLoader(reader=reader, metadata_manager=metadata_manager)


def test_create_checkpoint_loader(rank_info):
    """Test that CheckpointLoader can be constructed directly."""
    loader = _make_loader(rank_info)
    assert isinstance(loader, CheckpointLoader)
    loader.close()


def test_create_checkpoint_loader_with_metadata_manager(rank_info):
    """Test that CheckpointLoader accepts metadata_manager."""
    metadata_manager = DefaultMetadataManager(rank_info=rank_info)
    loader = _make_loader(rank_info, metadata_manager=metadata_manager)
    assert isinstance(loader, CheckpointLoader)
    loader.close()


def test_load_returns_none(rank_info, sample_state_dict):
    """Test that load() returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint")

        # Save using sync checkpointer
        save_checkpoint = SimpleCheckpoint(sample_state_dict)
        checkpointer = make_sync_checkpoint_saver(rank_info=rank_info)
        checkpointer.save(checkpoint_path, save_checkpoint)

        # Load using loader
        loader = _make_loader(rank_info)
        load_checkpoint = SimpleCheckpoint(
            {
                "model": {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)},
                "optimizer": {"step": torch.tensor(0)},
            }
        )
        result = loader.load(checkpoint_path, load_checkpoint)

        assert result is None
        loader.close()
        checkpointer.close()


def test_load_basic(rank_info, sample_state_dict):
    """Test basic load() without metadata manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint")

        # Save
        save_checkpoint = SimpleCheckpoint(sample_state_dict)
        checkpointer = make_sync_checkpoint_saver(rank_info=rank_info)
        checkpointer.save(checkpoint_path, save_checkpoint)

        # Load without metadata manager
        loader = _make_loader(rank_info)
        load_checkpoint = SimpleCheckpoint(
            {
                "model": {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)},
                "optimizer": {"step": torch.tensor(0)},
            }
        )
        loader.load(checkpoint_path, load_checkpoint)

        # Verify data was loaded correctly
        assert torch.allclose(
            load_checkpoint._state_dict["model"]["weight"],
            sample_state_dict["model"]["weight"],
        )
        loader.close()
        checkpointer.close()


def test_every_load_extracts_local_target_metadata_without_aggregating(
    rank_info, sample_state_dict
):
    """Each load passes freshly extracted local metadata to the reader and never
    runs the (collective) global aggregation or touches the save-side cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint")
        checkpointer = make_sync_checkpoint_saver(rank_info=rank_info)
        checkpointer.save(checkpoint_path, ReshardableCheckpoint(sample_state_dict))
        checkpointer.close()

        metadata_manager = DefaultMetadataManager(rank_info=rank_info)
        loader = _make_loader(rank_info, metadata_manager=metadata_manager)
        targets = []
        with (
            mock.patch.object(
                metadata_manager,
                "compute_distributed_metadata",
                side_effect=AssertionError("load must not aggregate metadata"),
            ),
            mock.patch.object(
                loader._reader, "read", wraps=loader._reader.read
            ) as read,
        ):
            for _ in range(2):
                target = ReshardableCheckpoint(
                    {
                        "model": {
                            "weight": torch.zeros(10, 10),
                            "bias": torch.zeros(10),
                        },
                        "optimizer": {"step": torch.tensor(0)},
                    }
                )
                loader.load(checkpoint_path, target)
                targets.append(target)

        assert read.call_count == 2
        for call, target in zip(read.call_args_list, targets):
            target_local_metadata = call.kwargs["checkpoint_info"].target_local_metadata
            expected = metadata_manager.extract_object_metadata(
                CheckpointInfo(checkpoint_items=target.get_items())
            )
            assert set(expected) == {"model", "optimizer"}
            assert target_local_metadata == expected
            assert torch.equal(
                target._state_dict["model"]["weight"],
                sample_state_dict["model"]["weight"],
            )
        assert not metadata_manager.has_computed_metadata()

        loader.close()
        metadata_manager.close()


@pytest.mark.gpus_needed_1
@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="requires an accelerator (async saver staging)",
)
def test_shared_metadata_manager_load_then_save(rank_info, sample_state_dict):
    """Test that a metadata_manager can be shared between a loader and a saver."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path1 = os.path.join(tmpdir, "checkpoint1")
        checkpoint_path2 = os.path.join(tmpdir, "checkpoint2")

        # First, save a checkpoint
        save_checkpoint = SimpleCheckpoint(sample_state_dict)
        checkpointer1 = make_async_checkpoint_saver(rank_info=rank_info)
        checkpointer1.save(checkpoint_path1, save_checkpoint)
        checkpointer1.close()

        # Create shared metadata_manager for loader and saver
        metadata_manager = DefaultMetadataManager(rank_info=rank_info)

        loader = _make_loader(rank_info, metadata_manager=metadata_manager)
        load_checkpoint = SimpleCheckpoint(
            {
                "model": {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)},
                "optimizer": {"step": torch.tensor(0)},
            }
        )
        loader.load(checkpoint_path1, load_checkpoint)
        loader.close()

        checkpointer2 = make_async_checkpoint_saver(
            rank_info=rank_info,
            checkpoint_metadata_manager=metadata_manager,
        )
        checkpointer2.save(checkpoint_path2, load_checkpoint)
        checkpointer2.close()
        metadata_manager.close()

        # Verify the checkpoint was saved
        assert os.path.exists(checkpoint_path2)
