# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
A readers-writer lock with readers-preference and write-to-read downgrade.

The lock allows any number of concurrent readers or a single exclusive writer.
It is *readers-preference*: as long as no writer currently holds the lock, a
reader acquires immediately, even while one or more writers are blocked waiting.
This can starve writers under a continuous stream of readers, which is an
acceptable (and desired) trade-off for the intended use: a fast staging buffer
that is read frequently but written rarely.

Holding the lock for write is non-reentrant (like ``threading.Lock``): a thread that
already holds the lock for write must not acquire it again, or it will deadlock.

``read`` and ``write`` are handles exposing the standard ``threading`` API
(``acquire(blocking=True, timeout=-1) -> bool`` and ``release()``). Used as
context managers, or via ``acquire_or_raise()``, they acquire blocking with the
lock's configured ``acquire_timeout`` and raise ``RuntimeError`` on failure --
we would rather fail a job loudly than let it hang forever, since a timeout here
almost always means a deadlock::

    with lock.read:          # acquire_or_raise() on enter, release() on exit
        read_stuff()

    lock.write.acquire_or_raise()   # blocking, times out -> RuntimeError
    try:
        write_stuff()
    finally:
        lock.write.release()

The raw ``acquire(blocking, timeout) -> bool`` is still available for callers
that want to poll (e.g. a non-blocking ``acquire(blocking=False)`` that cedes on
contention rather than raising).
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable

# Default deadline for a blocking context-manager / acquire_or_raise acquisition.
# Exceeding it raises RuntimeError rather than hanging (a timeout ~always means a
# deadlock). Callers with a known bound should pass their own via RWLock().
DEFAULT_ACQUIRE_TIMEOUT_SECS: float = 600.0


class RWLockMode(str, enum.Enum):
    """Acquire mode for a ``RWLock``."""

    READ = "read"
    WRITE = "write"


class _LockHandle:
    """Context-manager view over one acquire/release mode of an ``RWLock``."""

    def __init__(
        self,
        acquire: Callable[[bool, float], bool],
        release: Callable[[], None],
        acquire_timeout: float,
        mode: str,
    ) -> None:
        self._acquire = acquire
        self._release = release
        self._acquire_timeout = acquire_timeout
        self._mode = mode

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._acquire(blocking, timeout)

    def acquire_or_raise(self) -> None:
        """Acquire blocking with the lock's ``acquire_timeout``; raise on failure.

        Prefer this (and the context-manager form) over a bare, infinitely
        blocking acquire so a deadlock fails the job loudly instead of hanging.
        """
        if not self._acquire(True, self._acquire_timeout):
            raise RuntimeError(
                f"Failed to acquire {self._mode.value} lock after "
                f"{self._acquire_timeout} seconds; this is likely a deadlock"
            )

    def release(self) -> None:
        self._release()

    def __enter__(self) -> _LockHandle:
        self.acquire_or_raise()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self._release()
        return False


class RWLock:
    """Readers-preference readers-writer lock (see module docstring)."""

    def __init__(self, acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT_SECS) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._read_handle = _LockHandle(
            self._acquire_read, self._release_read, acquire_timeout, RWLockMode.READ
        )
        self._write_handle = _LockHandle(
            self._acquire_write, self._release_write, acquire_timeout, RWLockMode.WRITE
        )

    @property
    def read(self) -> _LockHandle:
        """Context-manager handle for acquiring the lock for reading."""
        return self._read_handle

    @property
    def write(self) -> _LockHandle:
        """Context-manager handle for acquiring the lock for writing."""
        return self._write_handle

    def _acquire(
        self,
        available: Callable[[], bool],
        grant: Callable[[], None],
        blocking: bool,
        timeout: float,
    ) -> bool:
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        if blocking and timeout < 0 and timeout != -1:
            raise ValueError("timeout value cannot be negative")
        with self._cond:
            if not available():
                if not blocking:
                    return False
                endtime = None if timeout < 0 else time.monotonic() + timeout
                while not available():
                    if endtime is None:
                        self._cond.wait()
                    else:
                        remaining = endtime - time.monotonic()
                        if remaining <= 0:
                            return False
                        self._cond.wait(remaining)
            grant()
            return True

    def _read_available(self) -> bool:
        return not self._writer_active

    def _grant_read(self) -> None:
        self._readers += 1

    def _acquire_read(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._acquire(self._read_available, self._grant_read, blocking, timeout)

    def _release_read(self) -> None:
        with self._cond:
            if self._readers <= 0:
                raise RuntimeError("cannot release un-acquired read lock")
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def _write_available(self) -> bool:
        return not self._writer_active and self._readers == 0

    def _grant_write(self) -> None:
        self._writer_active = True

    def _acquire_write(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._acquire(
            self._write_available, self._grant_write, blocking, timeout
        )

    def _release_write(self) -> None:
        with self._cond:
            if not self._writer_active:
                raise RuntimeError("cannot release un-acquired write lock")
            self._writer_active = False
            self._cond.notify_all()

    def downgrade(self) -> None:
        """Convert a held write lock into a read lock atomically.

        No other writer can intervene during the transition, because the whole
        conversion happens under the internal mutex. Waiting readers are woken
        (readers-preference); waiting writers stay blocked because a reader is
        now active.

        Raises ``RuntimeError`` if the lock is not currently held for write.
        """
        with self._cond:
            if not self._writer_active:
                raise RuntimeError("downgrade() called when write lock not held")
            self._writer_active = False
            self._readers += 1
            self._cond.notify_all()

    def locked_mode(self) -> RWLockMode | None:
        """
        Returns mode of lock curently held, or None if lock not held.

        Does not acquire the lock.
        """
        with self._cond:
            if self._writer_active:
                return RWLockMode.WRITE
            elif self._readers > 0:
                return RWLockMode.READ
            else:
                return None
