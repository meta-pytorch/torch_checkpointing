# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
from concurrent.futures import Future
from typing import Any, Mapping
from unittest.mock import Mock

import pytest
import torch
from torch_checkpointing.checkpoint_base import CheckpointBase, CheckpointItem
from torch_checkpointing.checkpoint_saver import AsyncCheckpointSaver
from torch_checkpointing.lock import RWLock, RWLockMode
from torch_checkpointing.staging import CheckpointStagerConfig, DefaultStager


class _MiniCheckpoint(CheckpointBase):
    """Smallest checkpoint that drives the save() pipeline."""

    def get_items(self) -> dict[str, CheckpointItem]:
        return {"t": CheckpointItem(value=torch.zeros(2), requires_copy=True)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass


def _make_saver(staging_future: Future, write_future: Future) -> AsyncCheckpointSaver:
    """Build a saver whose stager/process return caller-controlled futures."""
    mock_stager = Mock()
    mock_stager.stage.return_value = staging_future
    mock_process = Mock()
    mock_process.write.return_value = write_future
    return AsyncCheckpointSaver(
        checkpoint_stager=mock_stager,
        checkpoint_process=mock_process,
    )


def test_lock_property_is_rwlock():
    saver = _make_saver(Future(), Future())
    assert isinstance(saver.staging_lock, RWLock)
    saver.close()


def test_save_holds_write_lock_during_staging():
    staging_future: Future = Future()
    write_future: Future = Future()
    saver = _make_saver(staging_future, write_future)

    saver.save("/tmp/unused", _MiniCheckpoint())
    # Staging still in flight -> lock held exclusively for write.
    assert saver.staging_lock.locked_mode() == RWLockMode.WRITE

    # Drain the pipeline for clean teardown.
    staging_future.set_result({"t": torch.zeros(2)})
    write_future.set_result(None)
    saver.close()


def test_save_downgrades_to_read_after_staging_and_releases_after_write():
    staging_future: Future = Future()
    write_future: Future = Future()
    saver = _make_saver(staging_future, write_future)

    saver.save("/tmp/unused", _MiniCheckpoint())
    assert (
        saver.staging_lock.locked_mode() == RWLockMode.WRITE
    )  # write held while staging

    staging_future.set_result({"t": torch.zeros(2)})  # staging done -> downgrade
    assert saver.staging_lock.locked_mode() == RWLockMode.READ

    write_future.set_result(None)  # write done -> release
    assert saver.staging_lock.locked_mode() is None
    saver.close()


def test_save_releases_write_lock_when_launch_fails():
    write_future: Future = Future()
    mock_stager = Mock()
    mock_stager.stage.side_effect = RuntimeError("stage boom")
    mock_process = Mock()
    mock_process.write.return_value = write_future
    saver = AsyncCheckpointSaver(
        checkpoint_stager=mock_stager,
        checkpoint_process=mock_process,
    )

    with pytest.raises(RuntimeError, match="stage boom"):
        saver.save("/tmp/unused", _MiniCheckpoint())

    # The synchronously-acquired write lock must not leak on a launch failure.
    assert saver.staging_lock.locked_mode() is None
    saver.close()


def test_save_releases_lock_when_write_fails():
    staging_future: Future = Future()
    write_future: Future = Future()
    saver = _make_saver(staging_future, write_future)

    saver.save("/tmp/unused", _MiniCheckpoint())
    assert saver.staging_lock.locked_mode() == RWLockMode.WRITE
    staging_future.set_result({"t": torch.zeros(2)})  # downgrade to read
    assert saver.staging_lock.locked_mode() == RWLockMode.READ
    write_future.set_exception(RuntimeError("write boom"))  # -> release read

    with pytest.raises(RuntimeError, match="write boom"):
        write_future.result()
    # Lock released even though the write failed.
    assert saver.staging_lock.locked_mode() is None
    saver.close()


def test_lock_released_when_write_signal_precedes_staging_signal():
    """The staging and write futures complete on different threads, so their
    done-callbacks may run in either order. Even if the write-done signal is
    processed before the staging-done downgrade, the lock must end up fully
    released -- a read release must never run before the downgrade (which would
    raise and leak the lock)."""
    staging_future: Future = Future()
    write_future: Future = Future()
    saver = _make_saver(staging_future, write_future)

    saver.save("/tmp/unused", _MiniCheckpoint())

    # Deliver the write-done signal FIRST (out of order vs. the downgrade).
    write_future.set_result(None)
    # The downgrade hasn't happened yet, so the read release must be deferred:
    # the lock is still held.
    assert saver.staging_lock.locked_mode() == RWLockMode.WRITE

    # Now deliver the staging-done signal -> downgrade, then release.
    staging_future.set_result({"t": torch.zeros(2)})
    assert saver.staging_lock.locked_mode() is None
    saver.close()


def test_second_save_blocks_until_first_write_completes():
    staging1: Future = Future()
    write1: Future = Future()
    staging2: Future = Future()
    write2: Future = Future()

    mock_stager = Mock()
    mock_stager.stage.side_effect = [staging1, staging2]
    mock_process = Mock()
    mock_process.write.side_effect = [write1, write2]
    saver = AsyncCheckpointSaver(
        checkpoint_stager=mock_stager,
        checkpoint_process=mock_process,
    )

    # First save reaches the read state (staging done) but write still pending.
    saver.save("/tmp/1", _MiniCheckpoint())
    staging1.set_result({"t": torch.zeros(2)})

    second_returned = threading.Event()

    def second_save() -> None:
        saver.save("/tmp/2", _MiniCheckpoint())
        second_returned.set()

    t = threading.Thread(target=second_save)
    t.start()
    # The second save's write acquire must block while the first holds read.
    assert not second_returned.wait(0.3)

    write1.set_result(None)  # first write done -> read released -> second proceeds
    assert second_returned.wait(2)

    staging2.set_result({"t": torch.zeros(2)})
    write2.set_result(None)
    t.join(timeout=2)
    saver.close()


def test_stage_holds_write_lock_until_staging_completes():
    """stage() acquires the write lock synchronously and releases it once the
    staging future completes (no disk write, so no downgrade-to-read)."""
    staging_future: Future = Future()
    saver = _make_saver(staging_future, Future())

    fut = saver.stage(_MiniCheckpoint())
    assert fut is staging_future
    # Write held for the duration of staging: no reader or writer may join.
    assert saver.staging_lock.locked_mode() == "write"

    staging_future.set_result({"t": torch.zeros(2)})  # staging done -> release
    assert saver.staging_lock.locked_mode() is None
    saver.close()


def test_stage_releases_write_lock_when_stage_fails():
    """A failure launching the stage must not leak the synchronously-acquired
    write lock."""
    mock_stager = Mock()
    mock_stager.stage.side_effect = RuntimeError("stage boom")
    saver = AsyncCheckpointSaver(
        checkpoint_stager=mock_stager,
        checkpoint_process=Mock(),
    )

    with pytest.raises(RuntimeError, match="stage boom"):
        saver.stage(_MiniCheckpoint())

    assert saver.staging_lock.locked_mode() is None
    saver.close()


def test_stage_does_not_touch_lock():
    """_stage stages without acquiring or releasing the lock -- the caller owns
    the lock lifecycle."""
    staging_future: Future = Future()
    saver = _make_saver(staging_future, Future())

    fut = saver._stage(_MiniCheckpoint())
    assert fut is staging_future
    # Neither acquired nor released: the lock is untouched and fully free.
    assert saver.staging_lock.locked_mode() is None

    staging_future.set_result({"t": torch.zeros(2)})
    assert saver.staging_lock.locked_mode() is None
    saver.close()


def test_stage_stages_only_copy_required_items():
    """_stage stages only items whose CheckpointItem has requires_copy=True
    (distinct CPU copy) and passes requires_copy=False items through by identity
    -- the same selective-staging contract save() has. Runs a real DefaultStager
    so a regression in the requires_copy filtering surfaces here."""
    copied = torch.randn(4, 4)
    not_copied = torch.randn(4, 4)

    class _SelectiveCheckpoint(CheckpointBase):
        def get_items(self) -> dict[str, CheckpointItem]:
            return {
                "copied": CheckpointItem(value=copied, requires_copy=True),
                "not_copied": CheckpointItem(value=not_copied, requires_copy=False),
            }

        def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
            pass

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
        checkpoint_process=Mock(),
    )
    try:
        staged = saver._stage(_SelectiveCheckpoint()).result()
        assert staged["copied"] is not copied
        assert staged["copied"].data_ptr() != copied.data_ptr()
        torch.testing.assert_close(staged["copied"], copied)
        # Non-copy item passed through unchanged (same object).
        assert staged["not_copied"] is not_copied
    finally:
        stager.close()
