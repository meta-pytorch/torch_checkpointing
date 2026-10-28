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
from typing import Any, Callable, Mapping
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from torch_checkpointing.checkpoint_base import (
    CheckpointBase,
    CheckpointInfo,
    CheckpointItem,
    CheckpointWriteInfo,
)
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
from torch_checkpointing.utils import ensure_future, fut_then

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
) -> None:
    """Save the checkpoint synchronously.

    Args:
        checkpointer: The checkpointer to use for saving
        checkpoint_path: Path where checkpoint should be saved
        checkpoint: The checkpoint to save
    """
    save_result = checkpointer.save(checkpoint_path, checkpoint)
    if save_result is not None:
        stage_future, write_future = save_result
        stage_future.result()
        write_future.result()


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

    checkpoint_path = os.path.join(temp_dir, "checkpoint")
    save_sync(
        checkpointer,
        checkpoint_path,
        _create_checkpoint(model_partial, epoch=0, step=3),
    )

    loaded_checkpoint = _create_checkpoint(model_full, epoch=1, step=2)
    if strict:
        with pytest.raises(RuntimeError, match="missing keys.*block2"):
            load_checkpoint(
                loader, checkpoint_path, checkpoint=loaded_checkpoint, strict=True
            )
        return
    else:
        load_checkpoint(loader, checkpoint_path, checkpoint=loaded_checkpoint)

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

    checkpoint_path = os.path.join(temp_dir, "checkpoint")
    save_sync(
        checkpointer,
        checkpoint_path,
        _create_checkpoint(model_full, epoch=0, step=3),
    )

    partial_checkpoint = _create_checkpoint(model_partial, epoch=1, step=2)
    load_checkpoint(loader, checkpoint_path, checkpoint=partial_checkpoint)

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


def _metadata_item_keys(checkpoint_path: str) -> set[str]:
    with open(os.path.join(checkpoint_path, "metadata.pkl"), "rb") as f:
        return set(pickle.load(f)["metadata"])


def test_async_saver_sends_metadata_to_subprocess_only_when_it_changes(
    temp_dir: str,
    async_checkpointer: AsyncCheckpointSaver,
    metadata_manager: DefaultMetadataManager,
) -> None:
    """The write subprocess gets the metadata once per computation, and reuses
    it for every write until it is recomputed."""
    checkpoint = create_full_checkpoint()
    process = async_checkpointer._checkpoint_process
    sent: list[bytes | None] = []

    def capture_write(checkpoint_info_fut: Future, path: str) -> Future:
        write_info = ensure_future(checkpoint_info_fut).result()
        sent.append(write_info.serialized_distributed_metadata)
        return CheckpointProcess.write(process, checkpoint_info_fut, path)

    with patch.object(process, "write", side_effect=capture_write):
        for i in range(2):
            save_sync(
                async_checkpointer, os.path.join(temp_dir, f"ckpt_{i}"), checkpoint
            )
        metadata_manager.invalidate_computed_metadata()
        save_sync(
            async_checkpointer,
            os.path.join(temp_dir, "ckpt_2"),
            create_partial_checkpoint(),
        )

    assert sent[0] is not None
    assert sent[1] is None
    assert sent[2] is not None and sent[2] != sent[0]
    for i, expected in enumerate([sent[0], sent[0], sent[2]]):
        with open(os.path.join(temp_dir, f"ckpt_{i}", "metadata.pkl"), "rb") as f:
            assert f.read() == expected


def test_inflight_save_keeps_metadata_fetched_at_save_time(
    rank_info: RankInfo,
) -> None:
    """Invalidating and recomputing while a save is still staging must not change
    the metadata that save writes."""
    metadata_manager = DefaultMetadataManager(rank_info=rank_info)
    staging_future: Future = Future()
    stager = Mock()
    stager.stage.return_value = staging_future
    written: list[CheckpointWriteInfo] = []
    process = Mock()
    process.write.side_effect = lambda info_fut, _path: fut_then(
        info_fut, written.append
    )
    saver = AsyncCheckpointSaver(
        checkpoint_stager=stager,
        checkpoint_process=process,
        metadata_manager=metadata_manager,
    )
    try:
        checkpoint = create_full_checkpoint()
        saver.save("/tmp/unused", checkpoint)
        expected = metadata_manager.get_serialized_computed_metadata()

        metadata_manager.invalidate_computed_metadata()
        metadata_manager.compute_distributed_metadata(
            CheckpointInfo(create_partial_checkpoint().get_items())
        )
        assert metadata_manager.get_serialized_computed_metadata() != expected

        staging_future.set_result(
            {key: item.value for key, item in checkpoint.get_items().items()}
        )
        assert written[0].serialized_distributed_metadata is expected
    finally:
        saver.close()


def test_failed_save_launch_resends_metadata(rank_info: RankInfo) -> None:
    """If a save fails before its write is submitted, the next save sends the
    metadata again: the write subprocess may never have received it."""
    metadata_manager = DefaultMetadataManager(rank_info=rank_info)
    checkpoint = create_full_checkpoint()
    stager = Mock()
    stager.stage.side_effect = [
        RuntimeError("staging failed"),
        {key: item.value for key, item in checkpoint.get_items().items()},
    ]
    written: list[CheckpointWriteInfo] = []
    process = Mock()
    process.write.side_effect = lambda info_fut, _path: fut_then(
        info_fut, written.append
    )
    saver = AsyncCheckpointSaver(
        checkpoint_stager=stager,
        checkpoint_process=process,
        metadata_manager=metadata_manager,
    )
    try:
        with pytest.raises(RuntimeError, match="staging failed"):
            saver.save("/tmp/unused", checkpoint)
        saver.save("/tmp/unused", checkpoint)

        assert (
            written[0].serialized_distributed_metadata
            == metadata_manager.get_serialized_computed_metadata()
        )
    finally:
        saver.close()


def test_load_then_save_with_shared_metadata_manager(
    temp_dir: str,
    async_checkpointer: AsyncCheckpointSaver,
) -> None:
    """A save after a load through a shared manager computes and writes the
    metadata; the load itself computes nothing."""
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
        loaded_checkpoint = create_full_checkpoint(bias=0.0, epoch=0, step=0)
        loader.load(checkpoint_path_load, loaded_checkpoint)
        assert not metadata_manager.has_computed_metadata()

        save_sync(
            new_checkpointer,
            checkpoint_path_save,
            loaded_checkpoint,
        )

        assert metadata_manager.has_computed_metadata()
        assert _metadata_item_keys(checkpoint_path_save) == {"model", "optimizer"}

    finally:
        loader.close()
        new_checkpointer.close()
        metadata_manager.close()


def test_every_save_writes_metadata_computed_once(
    temp_dir: str,
    checkpointer: CheckpointSaver,
    metadata_manager: DefaultMetadataManager,
) -> None:
    """Every save writes metadata.pkl, while only the first one computes it."""
    checkpoint = create_full_checkpoint(bias=0.0, epoch=0, step=0)

    N = 5
    with patch.object(
        metadata_manager,
        "compute_distributed_metadata",
        wraps=metadata_manager.compute_distributed_metadata,
    ) as compute:
        for i in range(N):
            checkpoint_path = os.path.join(temp_dir, f"checkpoint_{i}")
            save_sync(checkpointer, checkpoint_path, checkpoint)

            # Modify checkpoint values (but not structure)
            assert checkpoint.step is not None
            assert checkpoint.epoch is not None
            checkpoint.step = checkpoint.step + 1
            checkpoint.epoch = checkpoint.epoch + 1

    assert compute.call_count == 1

    expected = metadata_manager.get_serialized_computed_metadata()
    for i in range(N):
        with open(os.path.join(temp_dir, f"checkpoint_{i}", "metadata.pkl"), "rb") as f:
            assert f.read() == expected


def test_save_after_invalidation_writes_the_new_structure(
    temp_dir: str,
    checkpointer: CheckpointSaver,
    metadata_manager: DefaultMetadataManager,
) -> None:
    """State that appears after the first save (here, the optimizer's momentum
    buffers) is in the metadata once the cached metadata is invalidated."""
    model = NestedModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    checkpoint = SimpleCheckpoint(model=model, optimizer=optimizer)

    save_sync(checkpointer, os.path.join(temp_dir, "before_step"), checkpoint)
    model(torch.randn(2, 16)).sum().backward()
    optimizer.step()
    metadata_manager.invalidate_computed_metadata()
    save_sync(checkpointer, os.path.join(temp_dir, "after_step"), checkpoint)

    assert _metadata_item_keys(os.path.join(temp_dir, "before_step")) == {"model"}
    assert _metadata_item_keys(os.path.join(temp_dir, "after_step")) == {
        "model",
        "optimizer",
    }


def _without_layouts(checkpoint: SimpleCheckpoint) -> SimpleCheckpoint:
    return SimpleCheckpoint(
        model=checkpoint.model,
        optimizer=checkpoint.optimizer,
        epoch=checkpoint.epoch,
        step=checkpoint.step,
        use_layout=False,
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda _: create_partial_checkpoint(bias=1.0, epoch=5),
        _without_layouts,
    ],
    ids=["structure", "layout"],
)
def test_save_raises_on_change_without_invalidation(
    temp_dir: str,
    checkpointer: CheckpointSaver,
    change: Callable[[SimpleCheckpoint], SimpleCheckpoint],
) -> None:
    """The sync saver always, and the async saver with validate_state_dict=True,
    raise instead of writing stale metadata for changed items or layouts."""
    checkpoint = create_full_checkpoint(bias=0.0, epoch=0, step=0)
    save_sync(checkpointer, os.path.join(temp_dir, "checkpoint1"), checkpoint)

    path = os.path.join(temp_dir, "checkpoint2")
    with pytest.raises(RuntimeError, match="invalidate_computed_metadata"):
        if isinstance(checkpointer, AsyncCheckpointSaver):
            checkpointer.save(path, change(checkpoint), validate_state_dict=True)
        else:
            checkpointer.save(path, change(checkpoint))
