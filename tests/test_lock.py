# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import time

import pytest
from torch_checkpointing.lock import RWLock, RWLockMode

# --- Basic single-threaded acquire / release ---


def test_read_acquire_release():
    lock = RWLock()
    assert lock.read.acquire() is True
    assert lock.locked_mode() == RWLockMode.READ
    lock.read.release()
    assert lock.locked_mode() is None


def test_write_acquire_release():
    lock = RWLock()
    assert lock.write.acquire() is True
    assert lock.locked_mode() == RWLockMode.WRITE
    lock.write.release()
    assert lock.locked_mode() is None


def test_multiple_readers_share():
    lock = RWLock()
    assert lock.read.acquire() is True
    assert lock.locked_mode() == RWLockMode.READ
    # A second reader can be held at the same time.
    assert lock.read.acquire(blocking=False) is True
    assert lock.locked_mode() == RWLockMode.READ
    lock.read.release()
    assert lock.locked_mode() == RWLockMode.READ
    lock.read.release()
    assert lock.locked_mode() is None


def test_writer_excludes_readers():
    lock = RWLock()
    assert lock.write.acquire() is True
    assert lock.read.acquire(blocking=False) is False
    lock.write.release()
    assert lock.read.acquire(blocking=False) is True
    lock.read.release()


def test_writer_excludes_writers():
    lock = RWLock()
    assert lock.write.acquire() is True
    assert lock.write.acquire(blocking=False) is False
    lock.write.release()


def test_reader_excludes_writer():
    lock = RWLock()
    assert lock.read.acquire() is True
    assert lock.write.acquire(blocking=False) is False
    lock.read.release()
    assert lock.write.acquire(blocking=False) is True
    lock.write.release()


# --- Non-reentrancy ---


def test_write_is_non_reentrant():
    lock = RWLock()
    assert lock.write.acquire() is True
    # Re-acquiring write (even from the same thread) must not succeed.
    assert lock.write.acquire(blocking=False) is False
    lock.write.release()


# --- blocking / timeout semantics (match threading.Lock) ---


def test_non_blocking_with_timeout_raises():
    lock = RWLock()
    with pytest.raises(ValueError):
        lock.read.acquire(blocking=False, timeout=1.0)
    with pytest.raises(ValueError):
        lock.write.acquire(blocking=False, timeout=1.0)


def test_acquire_timeout_returns_false():
    lock = RWLock()
    assert lock.write.acquire() is True
    start = time.monotonic()
    assert lock.read.acquire(timeout=0.2) is False
    assert time.monotonic() - start >= 0.2
    lock.write.release()


def test_release_unacquired_raises():
    lock = RWLock()
    with pytest.raises(RuntimeError):
        lock.read.release()
    with pytest.raises(RuntimeError):
        lock.write.release()


# --- acquire_or_raise / context-manager timeout (fail loud, don't hang) ---


def test_acquire_or_raise_succeeds_when_available():
    lock = RWLock()
    lock.write.acquire_or_raise()
    # Held for write now.
    assert lock.read.acquire(blocking=False) is False
    lock.write.release()


def test_acquire_or_raise_raises_on_timeout():
    lock = RWLock(acquire_timeout=0.2)
    assert lock.write.acquire() is True
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="deadlock"):
        lock.read.acquire_or_raise()
    assert time.monotonic() - start >= 0.2
    lock.write.release()


def test_context_manager_raises_on_timeout():
    lock = RWLock(acquire_timeout=0.2)
    assert lock.write.acquire() is True
    with pytest.raises(RuntimeError, match="deadlock"):
        with lock.read:
            pass
    # The failed acquire must not have left a read lock held.
    lock.write.release()
    assert lock.write.acquire(blocking=False) is True
    lock.write.release()


# --- Context managers ---


def test_read_context_manager():
    lock = RWLock()
    with lock.read:
        assert lock.write.acquire(blocking=False) is False
    # Released on exit.
    assert lock.write.acquire(blocking=False) is True
    lock.write.release()


def test_write_context_manager():
    lock = RWLock()
    with lock.write:
        assert lock.read.acquire(blocking=False) is False
    assert lock.read.acquire(blocking=False) is True
    lock.read.release()


def test_context_manager_releases_on_exception():
    lock = RWLock()
    with pytest.raises(RuntimeError):
        with lock.write:
            raise RuntimeError("boom")
    # Lock must be released despite the exception.
    assert lock.write.acquire(blocking=False) is True
    lock.write.release()


# --- Readers-preference ---


def test_readers_preference_new_reader_bypasses_pending_writer():
    lock = RWLock()
    assert lock.read.acquire() is True  # reader 1 holds

    writer_running = threading.Event()
    writer_acquired = threading.Event()

    def writer():
        writer_running.set()
        lock.write.acquire()  # blocks: a reader is active
        writer_acquired.set()
        lock.write.release()

    t = threading.Thread(target=writer)
    t.start()
    assert writer_running.wait(1)
    time.sleep(0.05)  # give the writer time to reach the blocking acquire

    # Readers-preference: a new reader gets in immediately despite the
    # pending writer.
    assert lock.read.acquire(blocking=False) is True
    assert not writer_acquired.is_set()

    lock.read.release()  # reader 2
    lock.read.release()  # reader 1
    assert writer_acquired.wait(2)  # writer finally proceeds
    t.join(timeout=2)


# --- Blocking hand-off between readers and writers ---


def test_writer_blocks_until_reader_releases():
    lock = RWLock()
    assert lock.read.acquire() is True
    acquired = threading.Event()

    def writer():
        assert lock.write.acquire(timeout=2) is True
        acquired.set()
        lock.write.release()

    t = threading.Thread(target=writer)
    t.start()
    assert not acquired.wait(0.2)  # blocked while the reader holds
    lock.read.release()
    assert acquired.wait(2)
    t.join(timeout=2)


def test_reader_blocks_until_writer_releases():
    lock = RWLock()
    assert lock.write.acquire() is True
    acquired = threading.Event()

    def reader():
        assert lock.read.acquire(timeout=2) is True
        acquired.set()
        lock.read.release()

    t = threading.Thread(target=reader)
    t.start()
    assert not acquired.wait(0.2)  # blocked while the writer holds
    lock.write.release()
    assert acquired.wait(2)
    t.join(timeout=2)


# --- downgrade ---


def test_downgrade_converts_write_to_read():
    lock = RWLock()
    assert lock.write.acquire() is True
    lock.downgrade()
    # Now holding read: a concurrent reader can join...
    assert lock.read.acquire(blocking=False) is True
    lock.read.release()
    # ...but a writer cannot.
    assert lock.write.acquire(blocking=False) is False
    lock.read.release()  # release the downgraded read
    # Fully released now.
    assert lock.write.acquire(blocking=False) is True
    lock.write.release()


def test_downgrade_without_write_raises():
    lock = RWLock()
    with pytest.raises(RuntimeError):
        lock.downgrade()
    # Holding read is also not a valid state to downgrade from.
    assert lock.read.acquire() is True
    with pytest.raises(RuntimeError):
        lock.downgrade()
    lock.read.release()


def test_downgrade_does_not_admit_pending_writer():
    lock = RWLock()
    assert lock.write.acquire() is True

    writer_acquired = threading.Event()

    def writer():
        if lock.write.acquire(timeout=0.5):
            writer_acquired.set()
            lock.write.release()

    t = threading.Thread(target=writer)
    t.start()
    time.sleep(0.05)  # let the writer reach its blocking acquire

    lock.downgrade()
    # The pending writer must NOT slip in during the write->read transition.
    time.sleep(0.1)
    assert not writer_acquired.is_set()
    # The downgrading thread still holds a read lock.
    assert lock.write.acquire(blocking=False) is False

    lock.read.release()
    t.join(timeout=2)


def test_downgrade_wakes_pending_reader():
    lock = RWLock()
    assert lock.write.acquire() is True

    reader_acquired = threading.Event()

    def reader():
        assert lock.read.acquire(timeout=2) is True
        reader_acquired.set()
        lock.read.release()

    t = threading.Thread(target=reader)
    t.start()
    assert not reader_acquired.wait(0.2)  # blocked by the writer

    lock.downgrade()  # readers-preference: the pending reader should proceed
    assert reader_acquired.wait(2)
    t.join(timeout=2)
    lock.read.release()
