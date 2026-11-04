# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
from torch_checkpointing.checkpoint_base import CheckpointBase, CheckpointItem
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.checkpoint_writer import CallbackArgs
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    CheckpointLoaderConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.default_load_plan_executor import DefaultLoadPlanExecutor
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.load_plan_executor import (
    LoadExecutionContext,
    LoadPlanExecutor,
)
from torch_checkpointing.metadata_serialization import (
    load_distributed_metadata,
    TorchDistributedMetadataFormat,
)
from torch_checkpointing.resharding import ReshardingInfo
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import STATE_DICT


def _payload() -> dict[str, Any]:
    return {
        "model": torch.arange(4),
        "step": 10,
    }


def _async_config_without_accelerator() -> CheckpointManager.Config:
    return CheckpointManager.Config(
        save=AsyncCheckpointSaverConfig(
            staging_config=CheckpointStagerConfig(
                use_pinned_memory=False,
                use_non_blocking_copy=False,
            )
        )
    )


class FakeStager:
    def __init__(self) -> None:
        self.future: Future[STATE_DICT] = Future()
        self.stage_calls = 0
        self.items: dict[str, CheckpointItem] | None = None

    def stage(self, items: dict[str, CheckpointItem]) -> Future[STATE_DICT]:
        self.stage_calls += 1
        self.items = items
        return self.future


class FakeSaver:
    def __init__(self, stager: FakeStager) -> None:
        self._stager = stager
        self.closed = False

    @property
    def stager(self) -> FakeStager:
        return self._stager

    def save(
        self,
        path: str,
        checkpoint: CheckpointBase,
    ) -> tuple[Future[Any], Future[Any]]:
        staging_future: Future[Any] = Future()
        write_future: Future[Any] = Future()
        staging_future.set_result(None)
        write_future.set_result(None)
        return staging_future, write_future

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("can_pin_memory_value", [False, True])
def test_with_async_save_defaults_to_pinned_memory_if_available(
    can_pin_memory_value: bool,
) -> None:
    # When the host can pin memory, async_save() enables pinning (and, with it,
    # non-blocking copies) by default.
    # Note, we need to mock `torch.cuda` here, mocking `can_pin_memory` is not sufficient.
    with (
        mock.patch("torch.cuda.is_available", return_value=can_pin_memory_value),
        mock.patch(
            "torch.cuda.cudart", return_value=object() if can_pin_memory_value else None
        ),
    ):
        config = CheckpointManager.Config.with_async_save()

    assert isinstance(config.save, AsyncCheckpointSaverConfig)
    assert config.save.wait_timeout_secs == 600
    assert config.save.staging_config.use_pinned_memory == can_pin_memory_value
    assert config.save.staging_config.use_shared_memory
    assert config.save.staging_config.use_async_staging
    assert config.save.staging_config.use_non_blocking_copy == can_pin_memory_value


def test_async_saver_config_can_disable_accelerator_features() -> None:
    config = _async_config_without_accelerator()

    assert isinstance(config.save, AsyncCheckpointSaverConfig)
    assert not config.save.staging_config.use_pinned_memory
    assert config.save.staging_config.use_shared_memory
    assert config.save.staging_config.use_async_staging
    assert not config.save.staging_config.use_non_blocking_copy


def test_with_sync_save_uses_sync_saver_config() -> None:
    config = CheckpointManager.Config.with_sync_save()

    assert isinstance(config.save, SyncCheckpointSaverConfig)


def test_saver_config_validates_wait_timeout() -> None:
    assert AsyncCheckpointSaverConfig(wait_timeout_secs=None).wait_timeout_secs is None

    with pytest.raises(ValueError, match="non-negative or None"):
        AsyncCheckpointSaverConfig(wait_timeout_secs=-1)


def test_config_has_default_load_config() -> None:
    config = CheckpointManager.Config()

    assert isinstance(config.load, CheckpointLoaderConfig)
    assert config.load.use_mmap is True
    assert config.load.load_plan_executor is None


def test_sync_save_returns_none_and_writes_checkpoint(tmp_path: Path) -> None:
    config = CheckpointManager.Config.with_sync_save()
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "sync_checkpoint")
    try:
        assert manager.save(checkpoint_path, _payload()) is None

        assert os.path.exists(os.path.join(checkpoint_path, "model_0.pt"))
        assert os.path.exists(os.path.join(checkpoint_path, "step_0.pt"))
    finally:
        manager.close()


def test_sync_save_runs_finalize_callback_after_checkpoint_write(
    tmp_path: Path,
) -> None:
    config = CheckpointManager.Config.with_sync_save()
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    checkpoint_path = os.path.join(tmp_path, "sync_checkpoint")
    finalize_observed_checkpoint: list[bool] = []

    def finalize_callback(args: CallbackArgs) -> None:
        assert args.path == checkpoint_path
        finalize_observed_checkpoint.append(
            os.path.exists(os.path.join(checkpoint_path, "model_0.pt"))
        )

    config.finalize_callback = finalize_callback
    manager = config.build()

    try:
        assert manager.save(checkpoint_path, _payload()) is None

        assert finalize_observed_checkpoint == [True]
    finally:
        manager.close()


def test_async_save_prewarms_staging_and_writes_checkpoint(tmp_path: Path) -> None:
    config = _async_config_without_accelerator()
    assert isinstance(config.save, AsyncCheckpointSaverConfig)
    config.save.wait_timeout_secs = 30
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "async_checkpoint")
    try:
        payload = _payload()
        manager.prewarm_staging(payload)
        write_future = manager.save(checkpoint_path, payload)
        assert write_future is not None
        write_future.result(timeout=30)

        assert os.path.exists(os.path.join(checkpoint_path, "model_0.pt"))
        assert os.path.exists(os.path.join(checkpoint_path, "step_0.pt"))
    finally:
        manager.close()


def test_async_save_config_forwards_finalize_callback_to_builder() -> None:
    stager = FakeStager()
    config = _async_config_without_accelerator()
    fake_saver = FakeSaver(stager)

    def finalize_callback(args: CallbackArgs) -> None:
        assert args.path == "checkpoint"

    config.finalize_callback = finalize_callback
    with mock.patch(
        "torch_checkpointing.checkpoint_manager.make_async_checkpoint_saver",
        return_value=fake_saver,
    ) as make_saver:
        manager = config.build()

    assert make_saver.call_args is not None
    assert make_saver.call_args.kwargs["finalize_callback"] is finalize_callback

    try:
        write_future = manager.save("checkpoint", _payload())
        assert write_future is not None
        write_future.result(timeout=1)
    finally:
        manager.close()


def test_prewarm_staging_stages_only_requires_copy_items() -> None:
    stager = FakeStager()
    config = _async_config_without_accelerator()
    config.save.wait_timeout_secs = 0
    # model uses the permissive default (requires_copy=True); step opts out.
    config.items = {"step": ItemSpec(requires_copy=False)}
    with mock.patch(
        "torch_checkpointing.checkpoint_manager.make_async_checkpoint_saver",
        return_value=FakeSaver(stager),
    ):
        manager = config.build()

    try:
        manager.prewarm_staging(_payload())

        assert stager.stage_calls == 1
        assert stager.items is not None
        assert set(stager.items.keys()) == {"model"}
        assert not stager.future.done()
    finally:
        if not stager.future.done():
            stager.future.set_result({})
        manager.close()


def test_sync_save_then_load_round_trip(tmp_path: Path) -> None:
    config = CheckpointManager.Config.with_sync_save()
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "sync_round_trip")
    try:
        assert manager.save(checkpoint_path, _payload()) is None

        loaded = manager.load(
            checkpoint_path,
            into={"model": torch.zeros(4, dtype=torch.long), "step": 0},
        )

        assert torch.equal(loaded["model"], torch.arange(4))
        assert loaded["step"] == 10
    finally:
        manager.close()


def test_async_save_then_load_round_trip(tmp_path: Path) -> None:
    config = _async_config_without_accelerator()
    config.save.wait_timeout_secs = 30
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "async_round_trip")
    try:
        write_future = manager.save(checkpoint_path, _payload())
        assert write_future is not None
        write_future.result(timeout=30)

        loaded = manager.load(
            checkpoint_path,
            into={"model": torch.zeros(4, dtype=torch.long), "step": 0},
        )

        assert torch.equal(loaded["model"], torch.arange(4))
        assert loaded["step"] == 10
    finally:
        manager.close()


@pytest.mark.parametrize("async_save", [False, True], ids=["sync", "async"])
def test_load_subset_then_save_all_items_writes_metadata_for_all(
    tmp_path: Path, async_save: bool
) -> None:
    # Fine-tuning from a weights-only checkpoint: load the model, then save the
    # model with a fresh optimizer through the same manager.
    def build() -> CheckpointManager:
        config = (
            _async_config_without_accelerator()
            if async_save
            else CheckpointManager.Config.with_sync_save()
        )
        config.items = {
            "model": ItemSpec(resharder=DefaultResharder()),
            "optim": ItemSpec(resharder=DefaultResharder(), required=False),
        }
        config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
        return config.build()

    def save(manager: CheckpointManager, path: str, payload: dict[str, Any]) -> None:
        write_future = manager.save(path, payload)
        if write_future is not None:
            write_future.result(timeout=30)

    weights_path = os.path.join(tmp_path, "weights")
    full_path = os.path.join(tmp_path, "full")
    weights = {"w": torch.arange(4, dtype=torch.float32)}
    optim = {"exp_avg": torch.ones(4)}

    manager = build()
    try:
        save(manager, weights_path, {"model": weights})
    finally:
        manager.close()

    manager = build()
    try:
        model = {"w": torch.zeros(4)}
        manager.load(weights_path, into={"model": model}, strict=True)
        save(manager, full_path, {"model": model, "optim": optim})
    finally:
        manager.close()

    metadata = load_distributed_metadata(
        full_path,
        LocalFileSystemStorageConfig(use_direct_io=False).create_storage(),
        formats=(TorchDistributedMetadataFormat,),
    )
    assert metadata is not None
    assert set(metadata.metadata) == {"model", "optim"}

    manager = build()
    try:
        loaded = manager.load(
            full_path,
            into={"model": {"w": torch.zeros(4)}, "optim": {"exp_avg": torch.zeros(4)}},
            strict=True,
        )
    finally:
        manager.close()
    torch.testing.assert_close(loaded["model"]["w"], weights["w"])
    torch.testing.assert_close(loaded["optim"]["exp_avg"], optim["exp_avg"])


class _AlwaysReshardResharder(DefaultResharder):
    """Reshards an unchanged layout too, so a one-process load reaches the executor."""

    def should_reshard(self, source_metadata: Any, target_metadata: Any) -> bool:
        return True


class _RecordingExecutor(LoadPlanExecutor):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def execute(
        self, items: Mapping[str, ReshardingInfo], context: LoadExecutionContext
    ) -> None:
        self.batches.append(list(items))
        DefaultLoadPlanExecutor().execute(items, context)


def test_load_reads_resharded_items_with_the_configured_executor(
    tmp_path: Path,
) -> None:
    executor = _RecordingExecutor()
    config = CheckpointManager.Config.with_sync_save()
    config.items = {"model": ItemSpec(resharder=_AlwaysReshardResharder())}
    config.load = CheckpointLoaderConfig(load_plan_executor=executor)
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    path = str(tmp_path / "checkpoint")
    weights = {"w": torch.arange(4, dtype=torch.float32)}
    target = {"w": torch.zeros(4)}

    manager = config.build()
    try:
        manager.save(path, {"model": weights})
        manager.load(path, into={"model": target}, strict=True)
    finally:
        manager.close()

    assert executor.batches == [["model"]]
    torch.testing.assert_close(target["w"], weights["w"])


def test_save_strict_schema_rejects_unknown_key(tmp_path: Path) -> None:
    config = CheckpointManager.Config.with_sync_save()
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    config.items = {"model": ItemSpec()}
    config.default = None  # strict: only declared keys allowed
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "strict")
    try:
        with pytest.raises(KeyError, match="strict"):
            manager.save(checkpoint_path, {"model": torch.arange(4), "extra": 1})
    finally:
        manager.close()


def test_load_on_closed_manager_raises() -> None:
    manager = CheckpointManager.Config.with_sync_save().build()
    manager.close()

    with pytest.raises(RuntimeError, match="closed"):
        manager.load("/tmp/does_not_exist", into={"model": torch.zeros(4)})
