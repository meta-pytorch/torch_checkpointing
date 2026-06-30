"""
Tests for CheckpointLoader class.
"""

import os
import tempfile

import pytest
import torch
from torch_checkpointing import (
    CheckpointLoader,
    CheckpointReader,
    make_async_checkpoint_saver,
    make_sync_checkpoint_saver,
)
from torch_checkpointing.checkpoint_base import CheckpointBase
from torch_checkpointing.metadata_manager import DefaultMetadataManager
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo


class SimpleCheckpoint(CheckpointBase):
    """Simple checkpoint wrapper for testing that wraps a state dict."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def get_items(self):
        from torch_checkpointing.checkpoint_base import CheckpointItem

        return {
            key: CheckpointItem(value=value) for key, value in self._state_dict.items()
        }

    def load_state_dict(self, state_dict):
        self._state_dict.update(state_dict)


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


def test_load_with_metadata_manager(rank_info, sample_state_dict):
    """Test load() with metadata manager triggers metadata computation and serialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint")

        # Save
        save_checkpoint = SimpleCheckpoint(sample_state_dict)
        checkpointer = make_sync_checkpoint_saver(rank_info=rank_info)
        checkpointer.save(checkpoint_path, save_checkpoint)

        # Load with metadata manager
        metadata_manager = DefaultMetadataManager(rank_info=rank_info)
        loader = _make_loader(rank_info, metadata_manager=metadata_manager)
        load_checkpoint = SimpleCheckpoint(
            {
                "model": {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)},
                "optimizer": {"step": torch.tensor(0)},
            }
        )
        loader.load(checkpoint_path, load_checkpoint)

        # Metadata should be serialized in the metadata_manager
        # (computed asynchronously for later use by a saver with the same metadata_manager)
        serialized_bytes = metadata_manager.get_serialized_metadata()
        assert serialized_bytes is not None
        assert isinstance(serialized_bytes, bytes)
        assert len(serialized_bytes) > 0

        loader.close()
        metadata_manager.close()
        checkpointer.close()


@pytest.mark.gpus_needed_1
def test_shared_metadata_manager_load_then_save(rank_info, sample_state_dict):
    """Test that shared metadata_manager enables metadata reuse between load and save."""
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

        # Load using CheckpointLoader
        # Metadata is computed and serialized in the shared metadata_manager
        loader = _make_loader(rank_info, metadata_manager=metadata_manager)
        load_checkpoint = SimpleCheckpoint(
            {
                "model": {"weight": torch.zeros(10, 10), "bias": torch.zeros(10)},
                "optimizer": {"step": torch.tensor(0)},
            }
        )
        loader.load(checkpoint_path1, load_checkpoint)
        loader.close()

        # Create checkpointer with the SAME metadata_manager
        # It will reuse the metadata computed during load
        checkpointer2 = make_async_checkpoint_saver(
            rank_info=rank_info,
            checkpoint_metadata_manager=metadata_manager,
        )

        # Save should work and reuse the metadata from the shared manager
        checkpointer2.save(checkpoint_path2, load_checkpoint)
        checkpointer2.close()
        metadata_manager.close()

        # Verify the checkpoint was saved
        assert os.path.exists(checkpoint_path2)
