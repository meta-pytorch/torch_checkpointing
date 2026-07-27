# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
from torch_checkpointing.checkpoint_base import CheckpointBase, CheckpointItem
from torch_checkpointing.checkpoint_manager import (
    CheckpointLoadConfig,
    CheckpointManager,
)
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import STATE_DICT


def _payload() -> dict[str, Any]:
    return {
        "model": torch.arange(4),
        "step": 10,
    }


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
        self, path: str, checkpoint: CheckpointBase
    ) -> tuple[Future[Any], Future[Any]]:
        staging_future: Future[Any] = Future()
        write_future: Future[Any] = Future()
        staging_future.set_result(None)
        write_future.set_result(None)
        return staging_future, write_future

    def close(self) -> None:
        self.closed = True


def test_async_save_config_defaults_to_pinned_memory() -> None:
    config = CheckpointManager.Config.async_save()

    assert isinstance(config.save.saver_config, AsyncCheckpointSaverConfig)
    assert config.save.saver_config.staging_config.use_pinned_memory
    assert config.save.saver_config.staging_config.use_shared_memory
    assert config.save.saver_config.staging_config.use_async_staging
    assert config.save.saver_config.staging_config.use_non_blocking_copy


def test_async_save_config_can_disable_pinned_memory() -> None:
    config = CheckpointManager.Config.async_save(pinned_memory=False)

    assert isinstance(config.save.saver_config, AsyncCheckpointSaverConfig)
    assert not config.save.saver_config.staging_config.use_pinned_memory
    assert config.save.saver_config.staging_config.use_shared_memory
    assert config.save.saver_config.staging_config.use_async_staging
    assert not config.save.saver_config.staging_config.use_non_blocking_copy


def test_sync_save_config_uses_sync_saver_config() -> None:
    config = CheckpointManager.Config.sync_save()

    assert isinstance(config.save.saver_config, SyncCheckpointSaverConfig)


def test_config_has_default_load_config() -> None:
    config = CheckpointManager.Config()

    assert isinstance(config.load, CheckpointLoadConfig)
    assert config.load.use_mmap is True


def test_sync_save_returns_none_and_writes_checkpoint(tmp_path: Path) -> None:
    config = CheckpointManager.Config.sync_save()
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "sync_checkpoint")
    try:
        assert manager.save(checkpoint_path, _payload()) is None

        assert os.path.exists(os.path.join(checkpoint_path, "model_0.pt"))
        assert os.path.exists(os.path.join(checkpoint_path, "step_0.pt"))
    finally:
        manager.close()


def test_async_save_prewarms_staging_and_writes_checkpoint(tmp_path: Path) -> None:
    config = CheckpointManager.Config.async_save(pinned_memory=False)
    assert isinstance(config.save.saver_config, AsyncCheckpointSaverConfig)
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


def test_prewarm_staging_stages_only_requires_copy_items() -> None:
    stager = FakeStager()
    config = CheckpointManager.Config.async_save(pinned_memory=False)
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
    config = CheckpointManager.Config.sync_save()
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
    config = CheckpointManager.Config.async_save(pinned_memory=False)
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


def test_save_strict_schema_rejects_unknown_key(tmp_path: Path) -> None:
    config = CheckpointManager.Config.sync_save()
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
    manager = CheckpointManager.Config.sync_save().build()
    manager.close()

    with pytest.raises(RuntimeError, match="closed"):
        manager.load("/tmp/does_not_exist", into={"model": torch.zeros(4)})
