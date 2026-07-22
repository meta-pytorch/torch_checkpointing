# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import shutil
import tempfile
from collections.abc import Generator, Iterator
from concurrent.futures import Future
from typing import Any, Mapping
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch_checkpointing.checkpoint_base import CheckpointBase, CheckpointItem
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization
from torch_checkpointing.checkpoint_loader import CheckpointLoader
from torch_checkpointing.checkpoint_process import (
    CheckpointProcess,
    CheckpointProcessConfig,
)
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.checkpoint_saver import (
    AsyncCheckpointSaver,
    CheckpointSaver,
    SyncCheckpointSaver,
)
from torch_checkpointing.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterArgs,
    CheckpointWriterConfig,
)
from torch_checkpointing.metadata_manager import DefaultMetadataManager
from torch_checkpointing.staging import CheckpointStagerConfig, DefaultStager
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo
from torch_checkpointing.utils import ensure_future

from .resharding_test_utils import SimpleResharder


class SimpleCheckpoint(CheckpointBase):
    """Simple checkpoint wrapper for testing.

    This class demonstrates the proper pattern for checkpoint objects:
    - Each checkpoint component is an explicit required field with proper types
    - Model is torch.nn.Module, optimizer is torch.optim.Optimizer
    - Layout information is hardcoded for each component
    - Copy requirements are hardcoded based on component characteristics
    - get_items() extracts state_dicts and constructs the list of items

    This serves as a clear example of how to structure checkpoint objects.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int | None = None,
        step: int | None = None,
        rank: int = 0,
        use_layout: bool = True,
        needs_resharder=False,
    ):
        """Initialize checkpoint with explicit required fields.

        Args:
            model: PyTorch model (required)
            optimizer: PyTorch optimizer (required)
            epoch: Training epoch (required)
            step: Training step (required)
            rank: Rank for distributed training
            use_layout: Whether to use custom layout (vs default)
        """
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self._rank = rank
        self._use_layout = use_layout
        self._needs_resharder = needs_resharder

    def get_items(self) -> dict[str, CheckpointItem]:
        """Return a dict of CheckpointItem objects representing the checkpoint.

        This explicitly adds each component one by one, demonstrating:
        - Model and optimizer require staging/copying (large tensors that might be modified)
        - Epoch and step don't require staging (small scalars, read-only)
        - Model and optimizer use custom layout when enabled
        - Epoch and step use default layout (no layout specified) to test default behavior
        - Model and optimizer are serialized to state_dicts for storage
        """
        items = {}

        # Model state - requires copy, uses torch serialization with custom layout
        # Extract state_dict from the model for serialization
        items["model"] = CheckpointItem(
            value=self.model.state_dict(),
            requires_copy=True,  # Model state is large and might be modified
            layout=(
                LayoutInfo(
                    file_path=f"model_rank_{self._rank}.pt",
                    serialization_format=TorchSerialization(),
                )
                if self._use_layout
                else None
            ),
            resharder=SimpleResharder(),
        )

        # Optimizer state - requires copy, uses torch serialization with custom layout
        # Extract state_dict from the optimizer for serialization
        if self.optimizer is not None:
            items["optimizer"] = CheckpointItem(
                value=self.optimizer.state_dict(),
                requires_copy=True,  # Optimizer state is large and might be modified
                layout=(
                    LayoutInfo(
                        file_path=f"optimizer_rank_{self._rank}.pt",
                        serialization_format=TorchSerialization(),
                    )
                    if self._use_layout
                    else None
                ),
                resharder=SimpleResharder(),
            )

        # Epoch metadata - no copy needed, uses default layout (tests default behavior)
        if self.epoch is not None:
            items["epoch"] = CheckpointItem(
                value=self.epoch,
                requires_copy=False,  # Epoch is a small scalar, read-only
                layout=None,  # Use default layout
            )
        if self.step is not None:
            # Step metadata - no copy needed, uses default layout (tests default behavior)
            items["step"] = CheckpointItem(
                value=self.step,
                requires_copy=False,  # Step is a small scalar, read-only
                layout=None,  # Use default layout
            )

        return items

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load the state from a loaded state dictionary into this checkpoint object.

        This demonstrates the proper pattern: use the existing model and optimizer
        instances, and load the saved state into them using PyTorch APIs.

        Args:
            state_dict: The loaded state dictionary containing saved state
        """
        self.model.load_state_dict(state_dict["model"], strict=False)
        if "optimizer" in state_dict:
            assert self.optimizer is not None
            self.optimizer.load_state_dict(state_dict["optimizer"])

        # Update epoch and step
        if "epoch" in state_dict:
            self.epoch = state_dict["epoch"]
        if "step" in state_dict:
            self.step = state_dict["step"]


def subprocess_init_fn(*args) -> None:
    """Initialize the subprocess for async checkpointer tests."""
    if len(args) != 2:
        raise ValueError(f"Expected 2 arguments but got {len(args)}")

    name, parent_pid = args
    expected_names = [
        "test-async-checkpointer",
        "test-layout-checkpointer",
        "test-partial-layout-checkpointer",
    ]
    assert name in expected_names, (
        f"Unexpected subprocess name: {name}. Expected one of: {expected_names}"
    )
    assert os.getpid() != parent_pid, "This was supposed to run in a different process"
    assert os.getppid() == parent_pid, (
        "This was supposed to run as a child to main process"
    )


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def rank_info() -> RankInfo:
    """Create rank info for testing."""
    return RankInfo(
        global_world_size=1,
        global_rank=0,
        role_rank=0,
        role_world_size=1,
    )


@pytest.fixture
def writer_config() -> CheckpointWriterConfig:
    """Create writer config for testing."""
    return CheckpointWriterConfig()


@pytest.fixture
def reader(
    rank_info: RankInfo, storage_config: LocalFileSystemStorageConfig
) -> CheckpointReader:
    """Create reader for testing."""
    return CheckpointReader(rank_info=rank_info, storage_config=storage_config)


@pytest.fixture
def metadata_manager(rank_info: RankInfo) -> DefaultMetadataManager:
    """Create metadata manager for testing."""
    return DefaultMetadataManager(
        rank_info=rank_info,
    )


@pytest.fixture
def checkpoint() -> SimpleCheckpoint:
    """Create test state dictionary."""
    return create_full_checkpoint()


@pytest.fixture
def storage_config() -> LocalFileSystemStorageConfig:
    """Create storage config for testing."""
    return LocalFileSystemStorageConfig()


@pytest.fixture(params=["sync", "async"])
def checkpointer(
    request: pytest.FixtureRequest,
    temp_dir: str,
    rank_info: RankInfo,
    writer_config: CheckpointWriterConfig,
    metadata_manager: DefaultMetadataManager,
    storage_config: LocalFileSystemStorageConfig,
) -> Generator[CheckpointSaver, None, None]:
    """Parametrized fixture that provides different checkpointer implementations."""
    if request.param == "sync":
        ckpt = create_sync_checkpointer(
            rank_info, writer_config, metadata_manager, storage_config
        )
    else:
        ckpt = create_async_checkpointer(
            rank_info, writer_config, metadata_manager, storage_config
        )
    yield ckpt
    ckpt.close()


@pytest.fixture()
def async_checkpointer(
    rank_info: RankInfo,
    writer_config: CheckpointWriterConfig,
    metadata_manager: DefaultMetadataManager,
    storage_config: LocalFileSystemStorageConfig,
) -> Iterator[CheckpointSaver]:
    """Create checkpointer (sync or async) for testing."""
    ckpt = create_async_checkpointer(
        rank_info, writer_config, metadata_manager, storage_config
    )
    yield ckpt
    ckpt.close()


@pytest.fixture()
def loader(
    reader: CheckpointReader,
    metadata_manager: DefaultMetadataManager,
) -> Iterator[CheckpointLoader]:
    """Create CheckpointLoader for testing load operations."""
    ldr = CheckpointLoader(reader=reader, metadata_manager=metadata_manager)
    yield ldr
    ldr.close()


def load_checkpoint(
    loader: CheckpointLoader,
    checkpoint_path: str,
    checkpoint: SimpleCheckpoint,
    default_map_location: Any = None,
    strict: bool = False,
) -> None:
    """Load a checkpoint using CheckpointLoader.

    Args:
        loader: The CheckpointLoader to use for loading
        checkpoint_path: Path to load the checkpoint from
        checkpoint: The checkpoint object to load into
        default_map_location: Device mapping for relocating tensors
        strict: If True, raises an error when there are missing keys
    """
    result = loader.load(
        checkpoint_path,
        checkpoint,
        default_map_location=default_map_location,
        strict=strict,
    )
    if strict and result.missing_keys:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is missing keys: {result.missing_keys}"
        )


def save_sync(
    checkpointer: CheckpointSaver,
    checkpoint_path: str,
    checkpoint: SimpleCheckpoint,
    clear_cache: bool = False,
) -> None:
    """Save the checkpoint synchronously.

    Args:
        checkpointer: The checkpointer to use for saving
        checkpoint_path: Path where checkpoint should be saved
        checkpoint: The checkpoint to save
        clear_cache: If True, clears metadata cache after save. Use this when the same
                     checkpointer will be used for a subsequent load with a different
                     state dict structure (e.g., partial loads in tests).
    """
    save_result = checkpointer.save(checkpoint_path, checkpoint)
    if save_result is not None:
        stage_future, write_future = save_result
        stage_future.result()
        write_future.result()

    # Clear metadata cache if requested (useful for tests with save+load of different structures)
    if clear_cache:
        if isinstance(checkpointer, (SyncCheckpointSaver, AsyncCheckpointSaver)):
            metadata_mgr = checkpointer._metadata_manager
            if isinstance(metadata_mgr, DefaultMetadataManager):
                # Access private variables to clear cache
                metadata_mgr._cached_local_metadata = None


@pytest.mark.parametrize("use_map_location", [False, True])
def test_save_and_load_basic(
    checkpointer: CheckpointSaver,
    temp_dir: str,
    checkpoint,
    loader: CheckpointLoader,
    use_map_location: bool,
) -> None:
    """Test basic save and load functionality with and without map_location."""
    checkpoint_path = os.path.join(
        temp_dir, "checkpoint_map" if use_map_location else "checkpoint"
    )

    save_sync(checkpointer, checkpoint_path, checkpoint)

    # Load the checkpoint using the loader
    loaded = create_full_checkpoint(bias=1.0, epoch=10, step=2000)
    load_checkpoint(
        loader,
        checkpoint_path,
        default_map_location="cpu" if use_map_location else None,
        checkpoint=loaded,
    )

    # Verify that we match originally saved checkpoint
    assert loaded.model.bias.max() == 0.0  # type: ignore
    assert loaded.epoch == checkpoint.epoch
    assert loaded.step == checkpoint.step


def test_partial_load(
    checkpointer: CheckpointSaver,
    temp_dir: str,
    loader: CheckpointLoader,
) -> None:
    """Test partial loading."""
    checkpoint_path = os.path.join(temp_dir, "checkpoint_partial")

    original_checkpoint = create_full_checkpoint()
    save_sync(
        checkpointer=checkpointer,
        checkpoint_path=checkpoint_path,
        checkpoint=original_checkpoint,
        clear_cache=True,  # Clear cache since load has different state dict structure
    )

    # Load only the keys in partial_state_dict
    loaded_checkpoint = create_partial_checkpoint(bias=1.0, epoch=20)
    load_checkpoint(loader, checkpoint_path, checkpoint=loaded_checkpoint)

    # Verify partial loading worked
    assert torch.equal(loaded_checkpoint.model.bias, original_checkpoint.model.bias)  # type: ignore
    assert loaded_checkpoint.epoch == original_checkpoint.epoch
    assert loaded_checkpoint.step is None


def create_sync_checkpointer(
    rank_info: RankInfo,
    writer_config: CheckpointWriterConfig,
    metadata_manager: DefaultMetadataManager,
    storage_config: LocalFileSystemStorageConfig,
) -> SyncCheckpointSaver:
    """Create a synchronous checkpoint saver."""
    args = CheckpointWriterArgs(
        config=writer_config,
        rank_info=rank_info,
        storage_config=storage_config,
    )
    writer = CheckpointWriter(args=args)
    return SyncCheckpointSaver(writer, metadata_manager)


def create_async_checkpointer(
    rank_info: RankInfo,
    writer_config: CheckpointWriterConfig,
    metadata_manager: DefaultMetadataManager,
    storage_config: LocalFileSystemStorageConfig,
) -> AsyncCheckpointSaver:
    """Create an asynchronous checkpoint saver."""
    stager_config = CheckpointStagerConfig(
        use_async_staging=True,
        use_pinned_memory=False,
        use_shared_memory=True,
        use_non_blocking_copy=False,
    )

    process_config = CheckpointProcessConfig(
        subprocess_init_timeout_secs=60,
        subprocess_shutdown_timeout_secs=120,
    )

    checkpoint_stager = DefaultStager(stager_config)

    checkpoint_writer_args = CheckpointWriterArgs(
        config=writer_config,
        rank_info=rank_info,
        storage_config=storage_config,
    )

    checkpoint_process = CheckpointProcess(
        rank_info=rank_info,
        config=process_config,
        subprocess_init_fn=subprocess_init_fn,
        subprocess_init_args=(
            "test-async-checkpointer",
            os.getpid(),
        ),
        checkpoint_writer_args=checkpoint_writer_args,
    )

    return AsyncCheckpointSaver(
        checkpoint_stager=checkpoint_stager,
        checkpoint_process=checkpoint_process,
        metadata_manager=metadata_manager,
    )


class NestedModel(nn.Module):
    def __init__(
        self,
        in_features: int = 16,
        hidden: int = 32,
        out_features: int = 10,
        bias: float = 0.0,
        use_block_2: bool = True,
    ):
        super().__init__()
        # This test uses bias to test that model is different
        self.bias = torch.nn.Parameter(
            torch.ones(in_features) * bias, requires_grad=False
        )

        class Block(nn.Module):
            def __init__(self, in_f: int, out_f: int):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Linear(in_f, out_f),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(out_f),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.seq(x)

        class Outer(nn.Module):
            def __init__(self, in_f: int, hidden_f: int, out_f: int, use_block_2: bool):
                super().__init__()
                self.block1 = Block(in_f, hidden_f)
                self.block2 = Block(hidden_f, hidden_f) if use_block_2 else None

                self.head = nn.Sequential(
                    nn.Linear(hidden_f, out_f),
                    nn.LogSoftmax(dim=1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.block1(x)
                x = self.block2(x) if self.block2 is not None else x
                x = self.head(x)
                return x

        self.outer = Outer(
            in_features, hidden, out_features, use_block_2=use_block_2
        )  # level 2: Nested inside NestedModel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(x)


def create_full_checkpoint(
    bias: float = 0.0, epoch: int = 5, step: int = 1000
) -> SimpleCheckpoint:
    # Create model and optimizer
    model = NestedModel(bias=bias)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Perform a training step to populate optimizer state (momentum buffers)
    x = torch.randn(2, 16)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    return SimpleCheckpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
    )


def create_partial_checkpoint(bias: float = 0.0, epoch: int = 10) -> SimpleCheckpoint:
    return SimpleCheckpoint(
        model=NestedModel(bias=bias),
        epoch=epoch,
    )


@pytest.mark.parametrize("use_layout", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_load_full_model_from_partial_checkpoint(
    checkpointer: CheckpointSaver,
    loader: CheckpointLoader,
    use_layout: bool,
    temp_dir: str,
    strict: bool,
) -> None:
    """Test loading nested dictionaries partially."""
    model_full = NestedModel(use_block_2=True)
    model_partial = NestedModel(use_block_2=False)

    def _create_checkpoint(
        model: nn.Module, epoch: int, step: int | None
    ) -> SimpleCheckpoint:
        # hack - using model_full to allow optimizer reloading, this is fine for this test
        # But in general, if you are removing some parameters from the model, you'd need some
        # surgery on optimizer state to load it properly
        return SimpleCheckpoint(
            model=model,
            optimizer=torch.optim.SGD(model_full.parameters(), lr=0.01, momentum=0.9),
            epoch=epoch,
            step=step,
            use_layout=use_layout,
        )

    save_sync(
        checkpointer,
        temp_dir,
        _create_checkpoint(model_partial, epoch=0, step=3),
        clear_cache=True,  # Clear cache since load has different state dict structure
    )

    loaded_checkpoint = _create_checkpoint(model_full, epoch=1, step=2)
    if strict:
        with pytest.raises(RuntimeError, match="missing keys.*block2"):
            load_checkpoint(loader, temp_dir, checkpoint=loaded_checkpoint, strict=True)
        return
    else:
        load_checkpoint(loader, temp_dir, checkpoint=loaded_checkpoint)

    # Block shouldn't be loaded, as it is None in the load call
    assert loaded_checkpoint.model.outer.block2 is not None  # type: ignore
    # Original checkpoint value should be loaded for epoch and new checkpoint value for step
    assert loaded_checkpoint.epoch == 0
    assert loaded_checkpoint.step == 3


@pytest.mark.parametrize("use_layout", [False, True])
def test_nested_dict_partial_load(
    checkpointer: CheckpointSaver,
    loader: CheckpointLoader,
    use_layout: bool,
    temp_dir: str,
) -> None:
    """Test loading nested dictionaries partially."""
    model_full = NestedModel(use_block_2=True)
    model_partial = NestedModel(use_block_2=False)

    def _create_checkpoint(
        model: nn.Module, epoch: int, step: int | None
    ) -> SimpleCheckpoint:
        # hack - using model_full to allow optimizer reloading, this is fine for this test
        # But in general, if you are removing some parameters from the model, you'd need some
        # surgery on optimizer state to load it properly
        return SimpleCheckpoint(
            model=model,
            optimizer=torch.optim.SGD(model_full.parameters(), lr=0.01, momentum=0.9),
            epoch=epoch,
            step=step,
            use_layout=use_layout,
        )

    save_sync(
        checkpointer,
        temp_dir,
        _create_checkpoint(model_full, epoch=0, step=3),
        clear_cache=True,  # Clear cache since load has different state dict structure
    )

    partial_checkpoint = _create_checkpoint(model_partial, epoch=1, step=2)
    load_checkpoint(loader, temp_dir, checkpoint=partial_checkpoint)

    # Block shouldn't be loaded, as it is None in the load call
    assert partial_checkpoint.model.outer.block2 is None  # type: ignore
    # Original checkpoint value should be loaded for epoch and new checkpoint value for step
    assert partial_checkpoint.epoch == 0
    assert partial_checkpoint.step == 3


def test_metadata_file_written(
    checkpointer: CheckpointSaver,
    temp_dir: str,
) -> None:
    """Test that metadata.pkl file is written."""
    checkpoint_path = os.path.join(temp_dir, "checkpoint")
    save_sync(checkpointer, checkpoint_path, create_full_checkpoint())

    metadata_file = os.path.join(checkpoint_path, "metadata.pkl")
    assert os.path.exists(metadata_file), "metadata.pkl not found"

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    assert isinstance(metadata, dict)


def test_async_sequential_saves(
    temp_dir: str,
    async_checkpointer: AsyncCheckpointSaver,
    loader: CheckpointLoader,
) -> None:
    """Test that sequential async saves wait for previous operations."""

    mutable_checkpoint = create_full_checkpoint(bias=0.0, epoch=0, step=0)
    write_futures = []
    N = 10

    def _checkpoint_path(i):
        return os.path.join(temp_dir, f"checkpoint_seq_{i}")

    for i in range(N):
        stage_future, write_future = async_checkpointer.save(
            _checkpoint_path(i), mutable_checkpoint
        )
        stage_future.result()
        assert mutable_checkpoint.step is not None
        mutable_checkpoint.step = mutable_checkpoint.step + 1
        mutable_checkpoint.model.bias += 1  # type: ignore
        write_futures.append(write_future)

    for write_future in write_futures:
        write_future.result()

    for i in range(N):
        loaded = create_full_checkpoint(bias=3.0)
        load_checkpoint(loader, _checkpoint_path(i), loaded)

        assert loaded.model.bias.max() == i  # type: ignore
        assert loaded.step == i
        assert loaded.epoch == 0


def test_async_error_handling() -> None:
    """Test error handling in async operations."""
    mock_stager = Mock()
    mock_process = Mock()

    mock_staging_future = Future()
    mock_staging_future.set_result({"staged": "data"})
    mock_stager.stage.return_value = mock_staging_future

    mock_write_future = Future()
    mock_write_future.set_exception(RuntimeError("Write failed"))
    mock_process.write.return_value = mock_write_future

    checkpointer = AsyncCheckpointSaver(
        checkpoint_stager=mock_stager,
        checkpoint_process=mock_process,
    )

    try:
        stage_future, write_future = checkpointer.save(
            "/tmp/test", create_full_checkpoint()
        )

        with pytest.raises(RuntimeError, match="Write failed"):
            write_future.result()

    finally:
        checkpointer.close()


def test_save_selectively_stages_only_copy_required_items() -> None:
    """AsyncCheckpointSaver.save must stage only items whose CheckpointItem has
    requires_copy=True (replacing the value with a distinct CPU copy) and pass
    items with requires_copy=False straight through with identity preserved.

    This locks the selective-staging behavior at the save boundary, independent
    of where the requires_copy filtering physically lives (saver vs stager).
    """
    copied_tensor = torch.randn(4, 4)
    not_copied_tensor = torch.randn(4, 4)

    class _SelectiveCheckpoint(CheckpointBase):
        def get_items(self) -> dict[str, CheckpointItem]:
            return {
                "copied": CheckpointItem(value=copied_tensor, requires_copy=True),
                "not_copied": CheckpointItem(
                    value=not_copied_tensor, requires_copy=False
                ),
            }

        def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
            pass

    captured: dict[str, Any] = {}

    def _capture_write(checkpoint_info_fut: Any, path: str) -> Future:
        captured["write_info"] = ensure_future(checkpoint_info_fut).result()
        return Future()

    mock_process = Mock()
    mock_process.write.side_effect = _capture_write

    stager = DefaultStager(
        CheckpointStagerConfig(
            use_async_staging=False,
            use_pinned_memory=False,
            use_shared_memory=False,
            use_non_blocking_copy=False,
        )
    )
    saver = AsyncCheckpointSaver(
        checkpoint_stager=stager,
        checkpoint_process=mock_process,
    )
    try:
        stage_future, _ = saver.save("/tmp/unused", _SelectiveCheckpoint())
        stage_future.result()

        staged = captured["write_info"].state_dict

        # Copy-required item: staged into a distinct CPU tensor with equal values.
        assert staged["copied"] is not copied_tensor
        assert staged["copied"].data_ptr() != copied_tensor.data_ptr()
        torch.testing.assert_close(staged["copied"], copied_tensor)

        # Non-copy item: passed through unchanged (same object, no copy).
        assert staged["not_copied"] is not_copied_tensor
    finally:
        stager.close()


def test_metadata_caching_and_serialization(
    async_checkpointer: AsyncCheckpointSaver,
) -> None:
    """Test metadata caching and async serialization behavior.

    This test validates the complete lifecycle:
    1. First compute_metadata returns CheckpointMetadata and kicks off serialization
    2. Cached metadata exists in the manager after first call
    3. Second compute_metadata returns None (cache is valid)
    4. Serialization future is created on first call and reused (not recreated)
    5. Serialized metadata can be retrieved and is cached
    6. After _metadata_sent=True, no more metadata is returned
    """
    checkpoint = create_full_checkpoint()

    # Get metadata manager and verify type
    metadata_manager = async_checkpointer._metadata_manager
    assert metadata_manager is not None
    assert isinstance(metadata_manager, DefaultMetadataManager), (
        "Expected DefaultMetadataManager"
    )

    # Phase 1: First compute_metadata call
    result1 = async_checkpointer._compute_metadata_once(checkpoint)
    assert result1 is not None, "First compute should return CheckpointMetadata"
    assert result1.distributed_metadata is not None

    # Verify serialization future was created in the metadata_manager
    assert metadata_manager._serialization_future is not None

    # Verify cached metadata exists in manager
    assert metadata_manager._cached_local_metadata is not None

    # Phase 2: Second compute_metadata call (cache hit)
    old_future = metadata_manager._serialization_future
    result2 = async_checkpointer._compute_metadata_once(checkpoint)
    assert result2 is None, "Second compute should return None (cache valid)"
    assert metadata_manager._serialization_future is old_future, (
        "Should not recreate serialization future"
    )

    # Phase 3: Get serialized metadata (waits for background serialization)
    serialized1 = async_checkpointer._get_serialized_metadata_if_needed()
    assert serialized1 is not None
    assert isinstance(serialized1, bytes)
    assert len(serialized1) > 0

    # Verify serialization future was consumed and result cached in metadata_manager
    assert metadata_manager._serialization_future is None
    assert metadata_manager._cached_serialized_metadata is not None

    # Phase 4: After metadata is marked as sent, no more metadata returned
    async_checkpointer._metadata_sent = True
    serialized2 = async_checkpointer._get_serialized_metadata_if_needed()
    assert serialized2 is None, "Should not return metadata after _metadata_sent=True"


def test_metadata_sent_once_to_subprocess(
    temp_dir: str,
    async_checkpointer: AsyncCheckpointSaver,
) -> None:
    """Test that metadata is only sent to subprocess once across multiple saves."""
    checkpoint_path1 = os.path.join(temp_dir, "checkpoint1")
    checkpoint_path2 = os.path.join(temp_dir, "checkpoint2")

    checkpoint = create_full_checkpoint()

    # First save - metadata should be sent
    assert async_checkpointer._metadata_sent is False
    save_sync(async_checkpointer, checkpoint_path1, checkpoint)
    assert async_checkpointer._metadata_sent is True

    # Second save - metadata should NOT be sent again
    save_sync(async_checkpointer, checkpoint_path2, checkpoint)

    # Verify both checkpoints have metadata.pkl (written by subprocess)
    assert os.path.exists(os.path.join(checkpoint_path1, "metadata.pkl"))
    assert os.path.exists(os.path.join(checkpoint_path2, "metadata.pkl"))

    # Verify metadata content is the same (since same state dict structure)
    with open(os.path.join(checkpoint_path1, "metadata.pkl"), "rb") as f:
        metadata1 = pickle.load(f)
    with open(os.path.join(checkpoint_path2, "metadata.pkl"), "rb") as f:
        metadata2 = pickle.load(f)

    # Both should have same structure (world_size, version, metadata keys)
    assert metadata1["world_size"] == metadata2["world_size"]
    assert metadata1["version"] == metadata2["version"]
    assert set(metadata1["metadata"].keys()) == set(metadata2["metadata"].keys())


def test_load_before_save_prepares_metadata(
    temp_dir: str,
    async_checkpointer: AsyncCheckpointSaver,
) -> None:
    """Test that loading before saving prepares metadata for subsequent saves."""
    checkpoint_path_save = os.path.join(temp_dir, "checkpoint_save")
    checkpoint_path_load = os.path.join(temp_dir, "checkpoint_load")

    # First, create a checkpoint to load from
    original_checkpoint = create_full_checkpoint(bias=1.0, epoch=10, step=100)
    save_sync(async_checkpointer, checkpoint_path_load, original_checkpoint)

    # Create a new checkpointer to test the load-then-save flow
    rank_info = RankInfo(
        global_world_size=1,
        global_rank=0,
        role_rank=0,
        role_world_size=1,
    )
    storage_config = LocalFileSystemStorageConfig()
    metadata_manager = DefaultMetadataManager(rank_info=rank_info)

    # Create a loader for the load operation
    loader = CheckpointLoader(
        reader=CheckpointReader(
            rank_info=rank_info,
            storage_config=storage_config,
        ),
        metadata_manager=metadata_manager,
    )

    # Create a checkpointer for the save operation (shares metadata_manager with loader)
    new_checkpointer = create_async_checkpointer(
        rank_info=rank_info,
        writer_config=CheckpointWriterConfig(),
        metadata_manager=metadata_manager,
        storage_config=storage_config,
    )

    try:
        # Load checkpoint - this triggers metadata computation in the shared metadata_manager
        loaded_checkpoint = create_full_checkpoint(bias=0.0, epoch=0, step=0)
        loader.load(checkpoint_path_load, loaded_checkpoint)

        # Now save - metadata is reused via the shared metadata_manager
        save_sync(
            new_checkpointer,
            checkpoint_path_save,
            loaded_checkpoint,
        )

        # Verify metadata was sent (only once)
        assert new_checkpointer._metadata_sent is True

        # Verify metadata file exists in saved checkpoint
        assert os.path.exists(os.path.join(checkpoint_path_save, "metadata.pkl"))

    finally:
        loader.close()
        new_checkpointer.close()
        metadata_manager.close()


def test_multiple_saves_reuse_cached_metadata(
    temp_dir: str,
    async_checkpointer: AsyncCheckpointSaver,
) -> None:
    """Test that multiple saves achieve 0-overhead by skipping metadata computation after first save."""
    checkpoint = create_full_checkpoint(bias=0.0, epoch=0, step=0)

    metadata_manager = async_checkpointer._metadata_manager
    assert metadata_manager is not None

    # Track how many times compute_metadata is called
    original_compute = metadata_manager.compute_metadata
    compute_count = {"count": 0}

    def tracked_compute(*args, **kwargs):
        compute_count["count"] += 1
        return original_compute(*args, **kwargs)

    metadata_manager.compute_metadata = tracked_compute

    # Perform 5 saves with the same checkpoint structure
    N = 5
    for i in range(N):
        checkpoint_path = os.path.join(temp_dir, f"checkpoint_{i}")
        save_sync(async_checkpointer, checkpoint_path, checkpoint)

        # Modify checkpoint values (but not structure)
        assert checkpoint.step is not None
        assert checkpoint.epoch is not None
        checkpoint.step = checkpoint.step + 1
        checkpoint.epoch = checkpoint.epoch + 1

    # compute_metadata should only be called once (first save)
    # subsequent saves skip metadata computation when validate_state_dict=False
    assert compute_count["count"] == 1, (
        f"Expected 1 compute_metadata call, got {compute_count['count']}"
    )

    # Verify all checkpoints were created
    for i in range(N):
        checkpoint_path = os.path.join(temp_dir, f"checkpoint_{i}")
        assert os.path.exists(os.path.join(checkpoint_path, "metadata.pkl"))


def test_metadata_cache_invalidation_on_structure_change(
    temp_dir: str,
    async_checkpointer: AsyncCheckpointSaver,
) -> None:
    """Test that changing checkpoint structure with validate_state_dict=True raises error.

    To detect structure changes, validate_state_dict=True
    must be used.
    """
    checkpoint1 = create_full_checkpoint(bias=0.0, epoch=0, step=0)

    # First save with full checkpoint
    checkpoint_path1 = os.path.join(temp_dir, "checkpoint1")
    save_sync(async_checkpointer, checkpoint_path1, checkpoint1)

    # Verify metadata was cached (cast to access private attributes)
    metadata_manager = async_checkpointer._metadata_manager
    assert metadata_manager is not None
    assert isinstance(metadata_manager, DefaultMetadataManager), (
        "Expected DefaultMetadataManager"
    )

    # Try to save with a different checkpoint structure (partial checkpoint)
    # With validate_state_dict=True, metadata validation runs and should detect the change
    checkpoint2 = create_partial_checkpoint(bias=1.0, epoch=5)
    checkpoint_path2 = os.path.join(temp_dir, "checkpoint2")

    # This should raise an error because the state dict structure changed
    # Use a helper to wrap all the operations that may raise
    def save_and_wait():
        stage_future, write_future = async_checkpointer.save(
            checkpoint_path2, checkpoint2, validate_state_dict=True
        )
        stage_future.result()
        write_future.result()

    with pytest.raises(RuntimeError, match="State dictionary has changed"):
        save_and_wait()
