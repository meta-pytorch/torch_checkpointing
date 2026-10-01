# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from torch_checkpointing.experimental.cooperative_resharding import (
    shared_memory as shared_memory_module,
    transport as transport_module,
)
from torch_checkpointing.experimental.cooperative_resharding.rendezvous import (
    C10dStoreRendezvous,
    InMemoryRendezvous,
    RendezvousNamespace,
)
from torch_checkpointing.experimental.cooperative_resharding.shared_memory import (
    AmbiguousRangeError,
    ChunkPool,
    RangeNotFoundError,
    RangeNotReadyError,
    RangeSpec,
    readinto_exact,
    SegmentSlice,
)
from torch_checkpointing.experimental.cooperative_resharding.transport import (
    NodeClient,
    NodeServer,
    ProtocolMismatchError,
    RangeRequest,
    RemoteLoadError,
    TransportError,
)


class _FakeC10dStore:
    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}
        self._condition = threading.Condition()

    def set(self, key: str, value: bytes) -> None:
        with self._condition:
            self._values[key] = value
            self._condition.notify_all()

    def get(self, key: str) -> bytes:
        with self._condition:
            while key not in self._values:
                self._condition.wait()
            return self._values[key]

    def check(self, keys: list[str]) -> bool:
        with self._condition:
            return all(key in self._values for key in keys)

    def compare_set(
        self, key: str, expected_value: bytes, desired_value: bytes
    ) -> bytes:
        with self._condition:
            current = self._values.get(key, b"")
            if current == expected_value:
                self._values[key] = desired_value
                self._condition.notify_all()
                return desired_value
            return current


class _BlockingReader:
    def __init__(self, value: bytes) -> None:
        self._value = value
        self.started = threading.Event()
        self.release = threading.Event()

    def readinto(self, buffer: memoryview) -> int:
        self.started.set()
        if not self.release.wait(1):
            raise TimeoutError("test reader was not released")
        buffer[:] = self._value
        return len(self._value)


class _SecondReadBlockingReader:
    def __init__(self, value: bytes) -> None:
        self._value = value
        self._offset = 0
        self._read_count = 0
        self.second_read_started = threading.Event()
        self.release_second_read = threading.Event()

    def readinto(self, buffer: memoryview) -> int:
        self._read_count += 1
        if self._read_count == 2:
            self.second_read_started.set()
            if not self.release_second_read.wait(1):
                raise TimeoutError("test reader's second read was not released")
        count = min(len(buffer), len(self._value) - self._offset)
        buffer[:count] = self._value[self._offset : self._offset + count]
        self._offset += count
        return count


def _read_slices(ranges: tuple[tuple[SegmentSlice, ...], ...]) -> bytes:
    result = bytearray()
    for resolved_range in ranges:
        for segment in resolved_range:
            with segment.path.open("rb") as source:
                source.seek(segment.file_offset)
                result.extend(source.read(segment.length))
    return bytes(result)


class RendezvousTest(unittest.TestCase):
    def test_in_memory_values_are_namespaced_and_errors_are_first_writer_wins(
        self,
    ) -> None:
        rendezvous = InMemoryRendezvous()
        first = RendezvousNamespace(1, "job", "load-1")
        second = RendezvousNamespace(1, "job", "load-2")

        rendezvous.put_blob(first, "address", b"node-a")

        self.assertEqual(b"node-a", rendezvous.get_blob(first, "address"))
        self.assertIsNone(rendezvous.get_blob(second, "address", timeout=0))
        self.assertTrue(rendezvous.publish_error(first, "first failure"))
        self.assertFalse(rendezvous.publish_error(first, "later failure"))
        self.assertEqual("first failure", rendezvous.get_error(first))

    def test_c10d_facade_waits_for_small_namespaced_blob(self) -> None:
        store = _FakeC10dStore()
        rendezvous = C10dStoreRendezvous(store)
        namespace = RendezvousNamespace(3, "job/with/slashes", "load token")

        publisher = threading.Thread(
            target=lambda: rendezvous.put_blob(namespace, "node-address", b"127.0.0.1")
        )
        publisher.start()

        self.assertEqual(
            b"127.0.0.1",
            rendezvous.get_blob(namespace, "node-address", timeout=1),
        )
        self.assertTrue(rendezvous.publish_error(namespace, "owner failed"))
        self.assertFalse(rendezvous.publish_error(namespace, "follower failed"))
        self.assertEqual("owner failed", rendezvous.get_error(namespace, timeout=1))
        publisher.join()


class SharedMemoryTest(unittest.TestCase):
    def test_segment_index_quiescence_rejects_live_state(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="index-quiescence",
                directory=directory,
            ) as pool:
                pool.index.assert_quiescent()
                reservation = pool.reserve("batch-0", 4)
                reservation.write_from("file-0", 0, io.BytesIO(b"data"), 4)

                with self.assertRaisesRegex(
                    RuntimeError,
                    r"segments=1, owners=1, active_readers=0, retiring_owners=0",
                ):
                    pool.index.assert_quiescent()

                lease = pool.index.acquire_many([RangeSpec("file-0", 0, 4)])
                with self.assertRaisesRegex(RuntimeError, r"active_readers=1"):
                    pool.index.assert_quiescent()

                retired = threading.Event()

                def retire() -> None:
                    reservation.retire()
                    retired.set()

                retire_thread = threading.Thread(target=retire)
                retire_thread.start()
                with pool.index._condition:
                    self.assertTrue(
                        pool.index._condition.wait_for(
                            lambda: bool(pool.index._retiring), timeout=1
                        )
                    )
                with self.assertRaisesRegex(RuntimeError, r"retiring_owners=1"):
                    pool.index.assert_quiescent()

                lease.close()
                self.assertTrue(retired.wait(1))
                retire_thread.join(timeout=1)
                self.assertFalse(retire_thread.is_alive())
                pool.index.assert_quiescent()

    def test_prepare_for_reuse_rejects_active_reservation_and_writer(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="reuse-active",
                directory=directory,
            ) as pool:
                self.assertTrue(pool.reuse_supported)
                reservation = pool.reserve("batch-0", 4)
                with self.assertRaisesRegex(RuntimeError, "active reservations: 1"):
                    pool.prepare_for_reuse()

                reader = _BlockingReader(b"data")
                writer = threading.Thread(
                    target=reservation.write_from,
                    args=("file-0", 0, reader, 4),
                )
                writer.start()
                self.assertTrue(reader.started.wait(1))
                with self.assertRaisesRegex(RuntimeError, "active reservations: 1"):
                    pool.prepare_for_reuse()
                self.assertTrue(writer.is_alive())

                reader.release.set()
                writer.join(timeout=1)
                self.assertFalse(writer.is_alive())
                reservation.retire()
                pool.prepare_for_reuse().assert_quiescent()

    def test_prepare_for_reuse_preserves_backing_and_replaces_index(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="reuse-success",
                directory=directory,
            )
            self.assertTrue(pool.reuse_supported)
            original_chunks = pool._chunks
            original_paths = tuple(chunk.path for chunk in original_chunks)
            original_descriptors = tuple(
                chunk.file_descriptor for chunk in original_chunks
            )
            original_mappings = tuple(chunk.mapping for chunk in original_chunks)
            original_index = pool.index

            reservation = pool.reserve("batch-0", 8)
            reservation.write_from("file-0", 0, io.BytesIO(b"abcdefgh"), 8)
            reservation.retire()
            original_index.assert_quiescent()

            replacement_index = pool.prepare_for_reuse()

            self.assertIs(replacement_index, pool.index)
            self.assertIsNot(replacement_index, original_index)
            self.assertEqual(original_chunks, pool._chunks)
            self.assertEqual(
                original_paths, tuple(chunk.path for chunk in pool._chunks)
            )
            self.assertEqual(
                original_descriptors,
                tuple(chunk.file_descriptor for chunk in pool._chunks),
            )
            self.assertEqual(
                original_mappings, tuple(chunk.mapping for chunk in pool._chunks)
            )
            with self.assertRaisesRegex(RuntimeError, "sealed for pool reuse"):
                original_index.assert_quiescent()
            replacement_index.assert_quiescent()

            second = pool.reserve("batch-1", 4)
            second.write_from("file-1", 10, io.BytesIO(b"WXYZ"), 4)
            with replacement_index.acquire_many([RangeSpec("file-1", 10, 4)]) as lease:
                self.assertEqual(b"WXYZ", _read_slices(lease.ranges))
            second.retire()

            pool.cleanup()

            self.assertFalse(pool.reuse_supported)
            self.assertTrue(all(mapping.closed for mapping in original_mappings))
            self.assertTrue(all(not path.exists() for path in original_paths))

    def test_prepare_for_reuse_requires_dontfork_protection(self) -> None:
        with TemporaryDirectory() as directory:
            with mock.patch.object(
                shared_memory_module,
                "_apply_mapping_advice",
                return_value=False,
            ):
                pool = ChunkPool(
                    capacity_bytes=4,
                    chunk_bytes=4,
                    job_token="reuse-without-dontfork",
                    directory=directory,
                )

            self.assertFalse(pool.reuse_supported)
            with self.assertRaisesRegex(RuntimeError, "requires MADV_DONTFORK"):
                pool.prepare_for_reuse()

            reservation = pool.reserve("batch-0", 4)
            reservation.write_from("file-0", 0, io.BytesIO(b"data"), 4)
            reservation.retire()
            pool.cleanup()

    def test_inherited_pool_rejects_use_and_does_not_unlink_parent_files(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="fork-safety",
                directory=directory,
            )
            chunk_path = pool._chunks[0].path
            inherited_pid = pool._owner_pid + 1

            with mock.patch.object(
                shared_memory_module.os, "getpid", return_value=inherited_pid
            ):
                self.assertFalse(pool.reuse_supported)
                with self.assertRaisesRegex(RuntimeError, "cannot be used after fork"):
                    pool.reserve("batch-0", 4)
                with self.assertRaisesRegex(RuntimeError, "cannot be used after fork"):
                    pool.prepare_for_reuse()
                with self.assertRaisesRegex(RuntimeError, "cannot be used after fork"):
                    pool.index.assert_quiescent()
                pool.cleanup()

            self.assertTrue(chunk_path.exists())
            self.assertFalse(pool._chunks[0].mapping.closed)
            pool.cleanup()
            self.assertFalse(chunk_path.exists())

    def test_register_many_is_atomic_and_tracks_each_range_readiness(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="atomic-registration",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                with self.assertRaises(AmbiguousRangeError):
                    reservation.register_many(
                        (RangeSpec("file-0", 0, 4), RangeSpec("file-0", 2, 4))
                    )

                self.assertEqual(0, reservation.used_bytes)
                with self.assertRaises(RangeNotFoundError):
                    pool.index.acquire_many([RangeSpec("file-0", 0, 2)])

                registered = reservation.register_many(
                    (RangeSpec("file-0", 0, 4), RangeSpec("file-0", 4, 4))
                )
                with self.assertRaises(RangeNotReadyError):
                    pool.index.acquire_many([RangeSpec("file-0", 0, 8)])

                reservation.write_registered(registered[0], io.BytesIO(b"abcd"))
                with pool.index.acquire_many([RangeSpec("file-0", 0, 4)]) as lease:
                    self.assertEqual(b"abcd", _read_slices(lease.ranges))
                with self.assertRaises(RangeNotReadyError):
                    pool.index.acquire_many([RangeSpec("file-0", 4, 4)])

                reservation.write_registered(registered[1], io.BytesIO(b"efgh"))
                with pool.index.acquire_many([RangeSpec("file-0", 0, 8)]) as lease:
                    self.assertEqual(b"abcdefgh", _read_slices(lease.ranges))

    def test_atomic_registered_write_becomes_ready_only_after_full_write(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="full-range-ready",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                registered = reservation.register_many((RangeSpec("file-0", 10, 8),))[0]
                reader = _SecondReadBlockingReader(b"abcdefgh")
                errors: list[Exception] = []

                def write() -> None:
                    try:
                        reservation.write_registered(registered, reader)
                    except Exception as error:
                        errors.append(error)

                writer = threading.Thread(
                    target=write,
                )
                writer.start()
                self.assertTrue(reader.second_read_started.wait(1))

                with self.assertRaises(RangeNotReadyError):
                    pool.index.acquire_many([RangeSpec("file-0", 10, 4)])

                reader.release_second_read.set()
                writer.join(timeout=1)
                self.assertFalse(writer.is_alive())
                self.assertEqual([], errors)
                with pool.index.acquire_many([RangeSpec("file-0", 10, 8)]) as lease:
                    self.assertEqual(b"abcdefgh", _read_slices(lease.ranges))

    def test_progressive_write_exposes_only_complete_segments(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="progressive-ready",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                registered = reservation.register_many((RangeSpec("file-0", 10, 8),))[0]
                reader = _SecondReadBlockingReader(b"abcdefgh")
                errors: list[Exception] = []

                def write() -> None:
                    try:
                        reservation.write_registered_progressively(registered, reader)
                    except Exception as error:
                        errors.append(error)

                writer = threading.Thread(target=write)
                writer.start()
                self.assertTrue(reader.second_read_started.wait(1))

                with pool.index.acquire_many([RangeSpec("file-0", 10, 4)]) as lease:
                    self.assertEqual(b"abcd", _read_slices(lease.ranges))
                with self.assertRaises(RangeNotReadyError):
                    pool.index.acquire_many([RangeSpec("file-0", 10, 8)])

                reader.release_second_read.set()
                writer.join(timeout=1)
                self.assertFalse(writer.is_alive())
                self.assertEqual([], errors)
                with pool.index.acquire_many([RangeSpec("file-0", 10, 8)]) as lease:
                    self.assertEqual(b"abcdefgh", _read_slices(lease.ranges))

    def test_progressive_write_short_read_leaves_later_segment_unready(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="progressive-short-read",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                registered = reservation.register_many((RangeSpec("file-0", 0, 8),))[0]

                with self.assertRaises(EOFError):
                    reservation.write_registered_progressively(
                        registered,
                        io.BytesIO(b"abcdef"),
                    )

                with pool.index.acquire_many([RangeSpec("file-0", 0, 4)]) as lease:
                    self.assertEqual(b"abcd", _read_slices(lease.ranges))
                with self.assertRaises(RangeNotReadyError):
                    pool.index.acquire_many([RangeSpec("file-0", 4, 4)])
                with self.assertRaises(RangeNotReadyError):
                    pool.index.acquire_many([RangeSpec("file-0", 0, 8)])

    def test_progressive_retirement_waits_for_writer_and_reader(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="progressive-retirement",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                registered = reservation.register_many((RangeSpec("file-0", 0, 8),))[0]
                reader = _SecondReadBlockingReader(b"abcdefgh")
                writer = threading.Thread(
                    target=reservation.write_registered_progressively,
                    args=(registered, reader),
                )
                writer.start()
                self.assertTrue(reader.second_read_started.wait(1))
                lease = pool.index.acquire_many([RangeSpec("file-0", 0, 4)])
                retired = threading.Event()

                def retire() -> None:
                    reservation.retire()
                    retired.set()

                retire_thread = threading.Thread(target=retire)
                retire_thread.start()
                self.assertFalse(retired.wait(0.05))

                reader.release_second_read.set()
                writer.join(timeout=1)
                self.assertFalse(writer.is_alive())
                self.assertFalse(retired.wait(0.05))

                lease.close()
                self.assertTrue(retired.wait(1))
                retire_thread.join(timeout=1)
                self.assertFalse(retire_thread.is_alive())

    def test_range_can_span_multiple_chunks(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="span",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                reservation.write_from("file-0", 100, io.BytesIO(b"abcdefgh"), 8)

                with pool.index.acquire_many([RangeSpec("file-0", 102, 4)]) as lease:
                    value = _read_slices(lease.ranges)

                self.assertEqual(b"cdef", value)

    def test_reservation_is_atomic_and_unblocks_after_retirement(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="blocking",
                directory=directory,
            ) as pool:
                first = pool.reserve("batch-0", 8)
                waiter_started = threading.Event()
                waiter_finished = threading.Event()
                acquired = []

                def reserve_second() -> None:
                    waiter_started.set()
                    acquired.append(pool.reserve("batch-1", 4, timeout=1))
                    waiter_finished.set()

                waiter = threading.Thread(target=reserve_second)
                waiter.start()
                self.assertTrue(waiter_started.wait(1))
                self.assertFalse(waiter_finished.wait(0.05))

                first.retire()

                self.assertTrue(waiter_finished.wait(1))
                self.assertEqual("batch-1", acquired[0].owner_id)
                acquired[0].retire()
                waiter.join()

    def test_waiting_duplicate_reservation_cannot_replace_active_owner(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="duplicate-owner",
                directory=directory,
            ) as pool:
                blocker = pool.reserve("blocker", 8)
                waiter_entered_wait = threading.Event()
                waiter_finished = threading.Event()
                errors: list[Exception] = []
                original_wait = pool._condition.wait

                def tracked_wait(timeout: float | None = None) -> bool:
                    waiter_entered_wait.set()
                    return original_wait(timeout)

                pool._condition.wait = tracked_wait

                def reserve_duplicate() -> None:
                    try:
                        pool.reserve("batch-0", 4, timeout=1)
                    except Exception as error:
                        errors.append(error)
                    finally:
                        waiter_finished.set()

                waiter = threading.Thread(target=reserve_duplicate)
                waiter.start()
                self.assertTrue(waiter_entered_wait.wait(1))

                with pool._condition:
                    blocker.retire()
                    active = pool.reserve("batch-0", 4)

                self.assertTrue(waiter_finished.wait(1))
                self.assertEqual(1, len(errors))
                self.assertIsInstance(errors[0], ValueError)
                self.assertIs(active, pool._reservations["batch-0"])
                active.retire()
                waiter.join()

    def test_retirement_waits_for_active_read_lease(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="lease",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 4)
                reservation.write_from("file-0", 0, io.BytesIO(b"data"), 4)
                lease = pool.index.acquire_many([RangeSpec("file-0", 0, 4)])
                retired = threading.Event()

                def retire_reservation() -> None:
                    reservation.retire()
                    retired.set()

                retire_thread = threading.Thread(target=retire_reservation)
                retire_thread.start()
                self.assertFalse(retired.wait(0.05))

                lease.close()

                self.assertTrue(retired.wait(1))
                with self.assertRaises(RangeNotFoundError):
                    pool.index.acquire_many([RangeSpec("file-0", 0, 4)])
                retire_thread.join()

    def test_retirement_waits_for_active_writer_before_chunk_reuse(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="writer-lease",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 4)
                reader = _BlockingReader(b"AAAA")
                writer = threading.Thread(
                    target=lambda: reservation.write_from("file-0", 0, reader, 4)
                )
                writer.start()
                self.assertTrue(reader.started.wait(1))
                retired = threading.Event()

                def retire_reservation() -> None:
                    reservation.retire()
                    retired.set()

                retire_thread = threading.Thread(target=retire_reservation)
                retire_thread.start()
                self.assertFalse(retired.wait(0.05))

                reader.release.set()
                writer.join()
                self.assertTrue(retired.wait(1))
                second = pool.reserve("batch-1", 4)
                second.write_from("file-1", 0, io.BytesIO(b"BBBB"), 4)
                with pool.index.acquire_many([RangeSpec("file-1", 0, 4)]) as lease:
                    value = _read_slices(lease.ranges)

                self.assertEqual(b"BBBB", value)
                retire_thread.join()

    def test_cleanup_is_idempotent_and_removes_job_directory(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="cleanup",
                directory=directory,
            )
            job_directory = pool.job_directory
            self.assertTrue(job_directory.is_dir())
            self.assertEqual(2, len(tuple(job_directory.iterdir())))

            pool.cleanup()
            pool.cleanup()

            self.assertFalse(job_directory.exists())

    def test_pool_counts_track_chunks_and_active_reservations(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="counts",
                directory=directory,
            )

            self.assertEqual(2, pool.chunk_count)
            self.assertEqual(0, pool.active_reservation_count)
            reservation = pool.reserve("batch-0", 4)
            self.assertEqual(1, pool.active_reservation_count)
            reservation.retire()
            self.assertEqual(0, pool.active_reservation_count)

            pool.cleanup()

            self.assertEqual(0, pool.chunk_count)

    def test_cleanup_closes_chunks_with_bounded_parallelism(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=10,
                chunk_bytes=1,
                job_token="bounded-parallel-cleanup",
                directory=directory,
            )
            chunk_type = type(pool._chunks[0])
            original_close = chunk_type.close
            release = threading.Event()
            saturated = threading.Event()
            state_lock = threading.Lock()
            active = 0
            peak = 0
            errors: list[Exception] = []

            def blocking_close(chunk: Any) -> None:
                nonlocal active, peak
                with state_lock:
                    active += 1
                    peak = max(peak, active)
                    if active == 8:
                        saturated.set()
                try:
                    if not release.wait(2):
                        raise TimeoutError("cleanup workers were not released")
                    original_close(chunk)
                finally:
                    with state_lock:
                        active -= 1

            def cleanup() -> None:
                try:
                    pool.cleanup()
                except Exception as error:
                    errors.append(error)

            with mock.patch.object(
                chunk_type,
                "close",
                autospec=True,
                side_effect=blocking_close,
            ):
                cleaner = threading.Thread(target=cleanup)
                cleaner.start()
                try:
                    self.assertTrue(saturated.wait(1))
                    with state_lock:
                        self.assertEqual(8, active)
                        self.assertEqual(8, peak)
                finally:
                    release.set()
                    cleaner.join(timeout=3)

            self.assertFalse(cleaner.is_alive())
            self.assertEqual([], errors)
            self.assertFalse(pool.job_directory.exists())

    def test_cleanup_falls_back_when_thread_pool_is_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="serial-cleanup-fallback",
                directory=directory,
            )
            chunks = pool._chunks

            with mock.patch.object(
                shared_memory_module,
                "ThreadPoolExecutor",
                side_effect=RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                ),
            ):
                pool.cleanup()

            self.assertTrue(all(chunk.mapping.closed for chunk in chunks))
            self.assertFalse(pool.job_directory.exists())

    def test_pool_reuse_rejects_missing_backing_file(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="missing-reuse-backing",
                directory=directory,
            )
            pool._chunks[0].path.unlink()

            self.assertFalse(pool.reuse_supported)
            with self.assertRaisesRegex(RuntimeError, "intact backing files"):
                pool.prepare_for_reuse()

            pool.cleanup()

    def test_concurrent_cleanup_closes_chunks_once(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="concurrent-cleanup",
                directory=directory,
            )
            job_directory = pool.job_directory
            errors: list[Exception] = []

            def cleanup() -> None:
                try:
                    pool.cleanup()
                except Exception as error:
                    errors.append(error)

            cleaners = [threading.Thread(target=cleanup) for _ in range(2)]
            for cleaner in cleaners:
                cleaner.start()
            for cleaner in cleaners:
                cleaner.join()

            self.assertEqual([], errors)
            self.assertFalse(job_directory.exists())

    def test_cleanup_retry_retains_chunk_after_close_failure(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="cleanup-close-retry",
                directory=directory,
            )
            chunk = pool._chunks[0]
            original_close = chunk.close
            attempts = 0

            def flaky_close() -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("injected close failure")
                original_close()

            with mock.patch.object(chunk, "close", side_effect=flaky_close):
                with self.assertRaisesRegex(OSError, "injected close failure"):
                    pool.cleanup()

                self.assertEqual((chunk,), pool._chunks)
                self.assertFalse(chunk.mapping.closed)

                pool.cleanup()

            self.assertEqual(2, attempts)
            self.assertTrue(chunk.mapping.closed)
            self.assertEqual((), pool._chunks)

    def test_cleanup_retry_retains_chunk_after_unlink_failure(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="cleanup-unlink-retry",
                directory=directory,
            )
            chunk = pool._chunks[0]
            original_unlink = Path.unlink
            attempts = 0

            def flaky_unlink(path: Path, *, missing_ok: bool = False) -> None:
                nonlocal attempts
                if path == chunk.path:
                    attempts += 1
                    if attempts == 1:
                        raise OSError("injected unlink failure")
                original_unlink(path, missing_ok=missing_ok)

            with mock.patch.object(
                Path, "unlink", autospec=True, side_effect=flaky_unlink
            ):
                with self.assertRaisesRegex(OSError, "injected unlink failure"):
                    pool.cleanup()

                self.assertEqual((chunk,), pool._chunks)
                self.assertTrue(chunk.path.exists())

                pool.cleanup()

            self.assertEqual(2, attempts)
            self.assertFalse(chunk.path.exists())
            self.assertEqual((), pool._chunks)

    def test_parallel_cleanup_retains_failed_chunks_in_original_order(self) -> None:
        with TemporaryDirectory() as directory:
            pool = ChunkPool(
                capacity_bytes=12,
                chunk_bytes=4,
                job_token="parallel-cleanup-retry",
                directory=directory,
            )
            chunks = pool._chunks
            chunk_type = type(chunks[0])
            original_close = chunk_type.close
            later_failure = threading.Event()
            attempts_lock = threading.Lock()
            attempts = [0, 0, 0]

            def flaky_close(chunk: Any) -> None:
                with attempts_lock:
                    attempts[chunk.index] += 1
                    attempt = attempts[chunk.index]
                if attempt == 1:
                    if chunk.index == 0:
                        if not later_failure.wait(1):
                            raise TimeoutError("later cleanup did not run")
                        raise OSError("first chunk failed")
                    if chunk.index == 2:
                        later_failure.set()
                        raise OSError("later chunk failed")
                original_close(chunk)

            with mock.patch.object(
                chunk_type,
                "close",
                autospec=True,
                side_effect=flaky_close,
            ):
                with self.assertRaisesRegex(OSError, "first chunk failed"):
                    pool.cleanup()

                self.assertEqual((chunks[0], chunks[2]), pool._chunks)
                self.assertEqual(2, pool.chunk_count)

                pool.cleanup()

            self.assertEqual([2, 1, 2], attempts)
            self.assertEqual(0, pool.chunk_count)
            self.assertFalse(pool.job_directory.exists())

    def test_readinto_exact_rejects_short_reader(self) -> None:
        destination = memoryview(bytearray(4))
        try:
            with self.assertRaises(EOFError):
                readinto_exact(io.BytesIO(b"abc"), destination)
        finally:
            destination.release()


class TransportTest(unittest.TestCase):
    def test_listener_loops_use_short_poll_interval_and_close_while_idle(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="listener-poll-interval",
                directory=directory,
            ) as pool:
                original_serve_forever = (
                    transport_module._BoundedHTTPServer.serve_forever
                )
                poll_intervals: dict[str, float] = {}
                both_listeners_started = threading.Event()
                intervals_lock = threading.Lock()

                def record_serve_forever(
                    server: Any,
                    poll_interval: float = 0.5,
                ) -> None:
                    with intervals_lock:
                        poll_intervals[server.RequestHandlerClass.__name__] = (
                            poll_interval
                        )
                        if len(poll_intervals) == 2:
                            both_listeners_started.set()
                    original_serve_forever(
                        server,
                        poll_interval=poll_interval,
                    )

                with mock.patch.object(
                    transport_module._BoundedHTTPServer,
                    "serve_forever",
                    autospec=True,
                    side_effect=record_serve_forever,
                ):
                    server = NodeServer(
                        pool.index,
                        protocol_version=1,
                        load_token="load-0",
                        control_worker_count=1,
                        data_worker_count=1,
                    ).start()
                    self.assertTrue(both_listeners_started.wait(timeout=1))
                    control_thread = server._control_thread
                    data_thread = server._data_thread
                    assert control_thread is not None
                    assert data_thread is not None

                    server.close()

                self.assertEqual(
                    {
                        "_ControlRequestHandler": 0.05,
                        "_DataRequestHandler": 0.05,
                    },
                    poll_intervals,
                )
                self.assertFalse(control_thread.is_alive())
                self.assertFalse(data_thread.is_alive())

    def test_control_connections_disable_nagle_without_changing_bulk_fetch(
        self,
    ) -> None:
        payload = bytes(range(256)) * 4096
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=len(payload),
                chunk_bytes=64 * 1024,
                job_token="control-nodelay",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", len(payload))
                reservation.write_from(
                    "file-0",
                    0,
                    io.BytesIO(payload),
                    len(payload),
                )
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    data_worker_count=1,
                    keepalive_idle_seconds=2,
                ) as server:
                    with (
                        NodeClient(
                            server.control_base_url,
                            protocol_version=1,
                            load_token="load-0",
                            max_attempts=1,
                        ) as control_client,
                        NodeClient(
                            server.data_base_url,
                            protocol_version=1,
                            load_token="load-0",
                            max_attempts=1,
                        ) as data_client,
                    ):
                        control_client.put_blob("tiny", b"x")
                        self.assertEqual(b"x", control_client.get_blob("tiny"))
                        destination = bytearray(len(payload))
                        self.assertEqual(
                            len(payload),
                            data_client.fetch_into(
                                [RangeRequest("file-0", 0, len(payload))],
                                destination,
                            ),
                        )
                        self.assertTrue(control_client.health()["ok"])

                        control_server = server._control_server
                        data_server = server._data_server
                        assert control_server is not None
                        assert data_server is not None
                        with control_server._sockets_lock:
                            control_sockets = tuple(control_server._sockets)
                        with data_server._sockets_lock:
                            data_sockets = tuple(data_server._sockets)

                        self.assertTrue(control_sockets)
                        self.assertTrue(data_sockets)
                        self.assertTrue(
                            all(
                                connection.getsockopt(
                                    socket.IPPROTO_TCP,
                                    socket.TCP_NODELAY,
                                )
                                == 1
                                for connection in control_sockets
                            )
                        )
                        self.assertTrue(
                            all(
                                connection.getsockopt(
                                    socket.IPPROTO_TCP,
                                    socket.TCP_NODELAY,
                                )
                                == 0
                                for connection in data_sockets
                            )
                        )
                        self.assertEqual(payload, destination)

    def test_loopback_control_plane_and_segment_spanning_fetch(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="http",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                reservation.write_from(
                    "file-0",
                    0,
                    io.BytesIO(b"abcdefgh"),
                    8,
                    mark_ready=False,
                )
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    data_worker_count=2,
                ) as server:
                    self.assertNotEqual(
                        server.control_base_url,
                        server.data_base_url,
                    )
                    with (
                        NodeClient(
                            server.control_base_url,
                            protocol_version=1,
                            load_token="load-0",
                            max_attempts=1,
                        ) as control_client,
                        NodeClient(
                            server.data_base_url,
                            protocol_version=1,
                            load_token="load-0",
                            max_attempts=1,
                        ) as data_client,
                    ):
                        self.assertEqual(1, control_client.health()["protocol_version"])
                        control_client.put_blob("node-plan", b"plan")
                        self.assertEqual(b"plan", control_client.get_blob("node-plan"))
                        self.assertTrue(control_client.delete_blob("node-plan"))
                        self.assertFalse(control_client.delete_blob("node-plan"))
                        self.assertIsNone(
                            control_client.get_blob("node-plan", timeout=0.01)
                        )
                        self.assertIsNone(
                            control_client.get_blob("missing", timeout=0.01)
                        )

                        destination = bytearray(8)
                        with self.assertRaises(RangeNotReadyError):
                            data_client.fetch_into(
                                [
                                    RangeRequest("file-0", 1, 6),
                                    RangeRequest("file-0", 0, 2),
                                ],
                                destination,
                            )
                        with self.assertRaises(RangeNotReadyError):
                            data_client.resolve_ranges([RangeRequest("file-0", 1, 6)])

                        reservation.mark_ready("file-0", 0, 8)
                        resolved = data_client.resolve_ranges(
                            [RangeRequest("file-0", 1, 6)]
                        )
                        self.assertEqual(
                            8,
                            data_client.fetch_into(
                                [
                                    RangeRequest("file-0", 1, 6),
                                    RangeRequest("file-0", 0, 2),
                                ],
                                destination,
                            ),
                        )
                        self.assertEqual(b"bcdefg", _read_slices(resolved))
                        self.assertEqual(b"bcdefgab", destination)
                        self.assertTrue(control_client.publish_error("first failure"))
                        self.assertFalse(control_client.publish_error("later failure"))
                        self.assertEqual("first failure", control_client.get_error())
                        with self.assertRaisesRegex(RemoteLoadError, "first failure"):
                            data_client.fetch_into(
                                [RangeRequest("file-0", 0, 2)], bytearray(2)
                            )

    def test_control_and_data_endpoints_are_confined_to_their_listeners(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="endpoint-confinement",
                directory=directory,
            ) as pool:
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    data_worker_count=1,
                ) as server:
                    with (
                        NodeClient(
                            server.control_base_url,
                            protocol_version=1,
                            load_token="load-0",
                            max_attempts=1,
                        ) as control_client,
                        NodeClient(
                            server.data_base_url,
                            protocol_version=1,
                            load_token="load-0",
                            max_attempts=1,
                        ) as data_client,
                    ):
                        body = data_client._encode_range_requests(
                            [RangeRequest("file-0", 0, 4)]
                        )
                        for path in ("/v1/fetch", "/v1/resolve"):
                            with self.subTest(control_path=path):
                                status, _ = control_client._small_request(
                                    "POST",
                                    path,
                                    body=body,
                                    content_type="application/json",
                                )
                                self.assertEqual(404, status)

                        for operation in (
                            data_client.health,
                            lambda: data_client.put_blob("plan", b"value"),
                            lambda: data_client.get_blob("plan"),
                            lambda: data_client.delete_blob("plan"),
                            lambda: data_client.publish_error("failure"),
                            data_client.get_error,
                        ):
                            with self.subTest(operation=operation):
                                with self.assertRaises(TransportError):
                                    operation()

    def test_fetch_retries_until_range_becomes_ready(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="retry",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 4)
                registered = reservation.register_many((RangeSpec("file-0", 10, 4),))[0]
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    data_worker_count=2,
                ) as server:
                    client = NodeClient(
                        server.data_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        max_attempts=8,
                        retry_delay=0.01,
                    )

                    def finish_download() -> None:
                        time.sleep(0.05)
                        reservation.write_registered(
                            registered,
                            io.BytesIO(b"data"),
                        )

                    downloader = threading.Thread(target=finish_download)
                    downloader.start()
                    destination = bytearray(4)

                    received = client.fetch_into(
                        [RangeRequest("file-0", 10, 4)],
                        destination,
                        ready_timeout=1,
                    )

                    self.assertEqual(4, received)
                    self.assertEqual(b"data", destination)
                    downloader.join()
                    client.close()

    def test_fetch_reads_progressive_prefix_before_full_range_is_ready(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="progressive-http",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                registered = reservation.register_many((RangeSpec("file-0", 10, 8),))[0]
                reader = _SecondReadBlockingReader(b"abcdefgh")
                errors: list[Exception] = []

                def write() -> None:
                    try:
                        reservation.write_registered_progressively(registered, reader)
                    except Exception as error:
                        errors.append(error)

                writer = threading.Thread(target=write)
                writer.start()
                self.assertTrue(reader.second_read_started.wait(1))
                try:
                    with (
                        NodeServer(
                            pool.index,
                            protocol_version=3,
                            load_token="load-0",
                            data_worker_count=2,
                        ) as server,
                        NodeClient(
                            server.data_base_url,
                            protocol_version=3,
                            load_token="load-0",
                            max_attempts=1,
                        ) as client,
                    ):
                        destination = bytearray(4)
                        self.assertEqual(
                            4,
                            client.fetch_into(
                                [RangeRequest("file-0", 10, 4)],
                                destination,
                            ),
                        )
                        self.assertEqual(b"abcd", destination)
                        with self.assertRaises(RangeNotReadyError):
                            client.fetch_into(
                                [RangeRequest("file-0", 10, 8)],
                                bytearray(8),
                            )
                        self.assertTrue(writer.is_alive())
                finally:
                    reader.release_second_read.set()
                    writer.join(timeout=1)

                self.assertFalse(writer.is_alive())
                self.assertEqual([], errors)

    def test_unready_fetch_does_not_hold_the_data_server_worker(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=8,
                chunk_bytes=4,
                job_token="nonblocking-ready",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 8)
                unready, ready = reservation.register_many(
                    (
                        RangeSpec("file-0", 0, 4),
                        RangeSpec("file-0", 4, 4),
                    )
                )
                reservation.write_registered(ready, io.BytesIO(b"done"))
                first_attempt = threading.Event()
                original_acquire_many = pool.index.acquire_many

                def observe_acquire(specs: Any) -> Any:
                    first_attempt.set()
                    return original_acquire_many(specs)

                with (
                    mock.patch.object(
                        pool.index,
                        "acquire_many",
                        side_effect=observe_acquire,
                    ),
                    NodeServer(
                        pool.index,
                        protocol_version=1,
                        load_token="load-0",
                        data_worker_count=1,
                        data_pending_requests=1,
                        keepalive_idle_seconds=5,
                    ) as server,
                    NodeClient(
                        server.data_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        max_attempts=20,
                        retry_delay=0.2,
                    ) as fetch_client,
                    NodeClient(
                        server.data_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        request_timeout=1,
                    ) as ready_client,
                ):
                    destination = bytearray(4)
                    errors: list[Exception] = []

                    def fetch() -> None:
                        try:
                            fetch_client.fetch_into(
                                [RangeRequest("file-0", 0, 4)],
                                destination,
                                ready_timeout=1,
                            )
                        except Exception as error:
                            errors.append(error)

                    fetch_thread = threading.Thread(target=fetch)
                    fetch_thread.start()
                    self.assertTrue(first_attempt.wait(timeout=1))

                    ready_destination = bytearray(4)
                    self.assertEqual(
                        4,
                        ready_client.fetch_into(
                            [RangeRequest("file-0", 4, 4)],
                            ready_destination,
                        ),
                    )
                    self.assertEqual(b"done", ready_destination)
                    self.assertTrue(fetch_thread.is_alive())

                    reservation.write_registered(
                        unready,
                        io.BytesIO(b"data"),
                    )
                    fetch_thread.join(timeout=1)

                self.assertFalse(fetch_thread.is_alive())
                self.assertEqual([], errors)
                self.assertEqual(b"data", destination)

    def test_data_saturation_does_not_block_control_listener(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="isolated-control",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 4)
                reservation.write_from("file-0", 0, io.BytesIO(b"data"), 4)
                first_started = threading.Event()
                release_first = threading.Event()
                original_acquire_many = pool.index.acquire_many
                call_lock = threading.Lock()
                call_count = 0

                def block_first_acquire(specs: Any) -> Any:
                    nonlocal call_count
                    with call_lock:
                        call_count += 1
                        is_first = call_count == 1
                    if is_first:
                        first_started.set()
                        if not release_first.wait(2):
                            raise TimeoutError("blocked data request was not released")
                    return original_acquire_many(specs)

                with (
                    mock.patch.object(
                        pool.index,
                        "acquire_many",
                        side_effect=block_first_acquire,
                    ),
                    NodeServer(
                        pool.index,
                        protocol_version=1,
                        load_token="load-0",
                        data_worker_count=1,
                        data_pending_requests=1,
                        control_worker_count=1,
                    ) as server,
                    NodeClient(
                        server.data_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        request_timeout=1,
                        max_attempts=1,
                    ) as first_data_client,
                    NodeClient(
                        server.data_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        request_timeout=1,
                        max_attempts=1,
                    ) as second_data_client,
                    NodeClient(
                        server.control_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        request_timeout=1,
                        max_attempts=1,
                    ) as control_client,
                ):
                    errors: list[Exception] = []

                    def fetch(client: NodeClient) -> None:
                        try:
                            client.fetch_into(
                                [RangeRequest("file-0", 0, 4)],
                                bytearray(4),
                            )
                        except Exception as error:
                            errors.append(error)

                    first = threading.Thread(target=fetch, args=(first_data_client,))
                    second = threading.Thread(target=fetch, args=(second_data_client,))
                    first.start()
                    self.assertTrue(first_started.wait(timeout=1))
                    second.start()
                    time.sleep(0.05)

                    self.assertTrue(control_client.health()["ok"])
                    control_client.put_blob("still-responsive", b"yes")
                    self.assertEqual(
                        b"yes", control_client.get_blob("still-responsive")
                    )

                    release_first.set()
                    first.join(timeout=1)
                    second.join(timeout=1)

                self.assertFalse(first.is_alive())
                self.assertFalse(second.is_alive())
                self.assertEqual([], errors)

    def test_missing_range_is_a_permanent_error(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="missing-range",
                directory=directory,
            ) as pool:
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    data_worker_count=1,
                ) as server:
                    with NodeClient(
                        server.data_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        max_attempts=8,
                    ) as client:
                        request = [RangeRequest("missing-file", 0, 4)]

                        with self.assertRaises(RangeNotFoundError):
                            client.fetch_into(request, bytearray(4), ready_timeout=1)
                        with self.assertRaises(RangeNotFoundError):
                            client.resolve_ranges(request, ready_timeout=1)

    def test_control_blob_storage_is_bounded_and_reclaimed(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="bounded-control",
                directory=directory,
            ) as pool:
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    max_control_storage_bytes=4,
                ) as server:
                    with NodeClient(
                        server.control_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        max_attempts=1,
                    ) as client:
                        client.put_blob("first", b"data")
                        with self.assertRaises(TransportError):
                            client.put_blob("second", b"x")

                        self.assertTrue(client.delete_blob("first"))
                        client.put_blob("second", b"x")
                        self.assertEqual(b"x", client.get_blob("second"))

    def test_empty_control_blobs_consume_aggregate_storage(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="bounded-empty-control",
                directory=directory,
            ) as pool:
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    max_control_storage_bytes=2,
                ) as server:
                    with NodeClient(
                        server.control_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        max_attempts=1,
                    ) as client:
                        client.put_blob("first", b"")
                        client.put_blob("second", b"")
                        client.put_blob("first", b"")
                        with self.assertRaises(TransportError):
                            client.put_blob("third", b"")

                        self.assertTrue(client.delete_blob("first"))
                        client.put_blob("third", b"")
                        self.assertEqual(b"", client.get_blob("third"))

    def test_delete_retry_preserves_success_after_lost_response(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="delete-retry",
                directory=directory,
            ) as pool:
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                ) as server:
                    with NodeClient(
                        server.control_base_url,
                        protocol_version=1,
                        load_token="load-0",
                        max_attempts=2,
                        retry_delay=0,
                    ) as client:
                        client.put_blob("plan", b"data")
                        read_response = client._read_small_response
                        response_count = 0

                        def lose_first_response(response: Any) -> bytes:
                            nonlocal response_count
                            response_count += 1
                            body = read_response(response)
                            if response_count == 1:
                                raise EOFError("simulated lost DELETE response")
                            return body

                        with mock.patch.object(
                            client,
                            "_read_small_response",
                            side_effect=lose_first_response,
                        ):
                            self.assertTrue(client.delete_blob("plan"))

                        self.assertEqual(2, response_count)
                        self.assertFalse(client.delete_blob("plan"))

    def test_server_close_clears_and_disables_direct_control_writes(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="direct-control-close",
                directory=directory,
            ) as pool:
                server = NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    data_worker_count=1,
                ).start()
                control_url = server.control_base_url
                data_url = server.data_base_url
                server.put_blob("plan", b"data")
                self.assertTrue(server.publish_error("failure"))

                server.close()
                server.close()

                self.assertIsNone(server.get_blob("plan"))
                self.assertIsNone(server.get_error())
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    server.put_blob("late-plan", b"data")
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    server.publish_error("late failure")
                with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
                    server.start()
                with NodeClient(
                    control_url,
                    protocol_version=1,
                    load_token="load-0",
                    request_timeout=0.1,
                    max_attempts=1,
                ) as control_client:
                    with self.assertRaises(TransportError):
                        control_client.health()
                with NodeClient(
                    data_url,
                    protocol_version=1,
                    load_token="load-0",
                    request_timeout=0.1,
                    max_attempts=1,
                ) as data_client:
                    with self.assertRaises(TransportError):
                        data_client.fetch_into(
                            [RangeRequest("file-0", 0, 4)],
                            bytearray(4),
                        )

    def test_second_listener_bind_failure_closes_first_listener(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="partial-start",
                directory=directory,
            ) as pool:
                control_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                control_probe.bind(("127.0.0.1", 0))
                control_port = control_probe.getsockname()[1]
                control_probe.close()
                occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                occupied.bind(("127.0.0.1", 0))
                occupied.listen()
                data_port = occupied.getsockname()[1]
                server = NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    data_worker_count=1,
                    control_port=control_port,
                    data_port=data_port,
                )
                try:
                    with self.assertRaises(OSError):
                        server.start()
                finally:
                    occupied.close()

                rebound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    rebound.bind(("127.0.0.1", control_port))
                finally:
                    rebound.close()
                server.close()
                with self.assertRaisesRegex(RuntimeError, "cannot be restarted"):
                    server.start()

    def test_second_listener_thread_failure_closes_both_listeners(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="partial-thread-start",
                directory=directory,
            ) as pool:
                ports: list[int] = []
                for _ in range(2):
                    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    probe.bind(("127.0.0.1", 0))
                    ports.append(probe.getsockname()[1])
                    probe.close()
                server = NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    data_worker_count=1,
                    control_port=ports[0],
                    data_port=ports[1],
                )
                original_start = threading.Thread.start

                def fail_data_start(thread: threading.Thread) -> None:
                    if thread.name == "coop-data-http-server":
                        raise RuntimeError("injected data listener start failure")
                    original_start(thread)

                with (
                    mock.patch.object(
                        threading.Thread,
                        "start",
                        autospec=True,
                        side_effect=fail_data_start,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "injected data listener start failure",
                    ),
                ):
                    server.start()

                for port in ports:
                    rebound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        rebound.bind(("127.0.0.1", port))
                    finally:
                        rebound.close()
                server.close()

    def test_server_rejects_wrong_load_token(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="token",
                directory=directory,
            ) as pool:
                with NodeServer(
                    pool.index,
                    protocol_version=2,
                    load_token="correct",
                    control_worker_count=1,
                    data_worker_count=1,
                ) as server:
                    with (
                        NodeClient(
                            server.control_base_url,
                            protocol_version=2,
                            load_token="wrong",
                            max_attempts=1,
                        ) as control_client,
                        NodeClient(
                            server.data_base_url,
                            protocol_version=2,
                            load_token="wrong",
                            max_attempts=1,
                        ) as data_client,
                    ):
                        with self.assertRaises(ProtocolMismatchError):
                            control_client.health()
                        with self.assertRaises(ProtocolMismatchError):
                            data_client.fetch_into(
                                [RangeRequest("file-0", 0, 4)],
                                bytearray(4),
                            )

    def test_client_reconnects_after_server_rotates_keepalive_connection(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="connection-rotation",
                directory=directory,
            ) as pool:
                with NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    max_requests_per_connection=2,
                ) as server:
                    with NodeClient(
                        server.control_base_url,
                        protocol_version=1,
                        load_token="load-0",
                    ) as client:
                        health_results = [client.health()["ok"] for _ in range(5)]

                    self.assertEqual([True] * 5, health_results)

    def test_server_shutdown_interrupts_active_and_queued_requests(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="shutdown",
                directory=directory,
            ) as pool:
                server = NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    control_worker_count=1,
                    control_pending_requests=0,
                    max_poll_seconds=5,
                ).start()
                first = NodeClient(
                    server.control_base_url,
                    protocol_version=1,
                    load_token="load-0",
                    request_timeout=2,
                    max_attempts=1,
                )
                second = NodeClient(
                    server.control_base_url,
                    protocol_version=1,
                    load_token="load-0",
                    request_timeout=2,
                    max_attempts=1,
                )
                outcomes: list[str] = []

                def wait_for_blob() -> None:
                    try:
                        first.get_blob("missing", timeout=5)
                        outcomes.append("active-completed")
                    except TransportError:
                        outcomes.append("active-interrupted")

                def wait_for_worker() -> None:
                    try:
                        second.health()
                        outcomes.append("queued-completed")
                    except TransportError:
                        outcomes.append("queued-interrupted")

                active = threading.Thread(target=wait_for_blob)
                queued = threading.Thread(target=wait_for_worker)
                active.start()
                time.sleep(0.05)
                queued.start()
                time.sleep(0.05)

                started = time.monotonic()
                server.close()
                elapsed = time.monotonic() - started

                active.join(1)
                queued.join(1)
                first.close()
                second.close()
                self.assertLess(elapsed, 1)
                self.assertFalse(active.is_alive())
                self.assertFalse(queued.is_alive())
                self.assertEqual(2, len(outcomes))

    def test_server_shutdown_interrupts_active_data_request(self) -> None:
        with TemporaryDirectory() as directory:
            with ChunkPool(
                capacity_bytes=4,
                chunk_bytes=4,
                job_token="data-shutdown",
                directory=directory,
            ) as pool:
                reservation = pool.reserve("batch-0", 4)
                reservation.write_from("file-0", 0, io.BytesIO(b"data"), 4)
                acquire_started = threading.Event()
                release_acquire = threading.Event()
                original_acquire_many = pool.index.acquire_many

                def block_acquire(specs: Any) -> Any:
                    acquire_started.set()
                    if not release_acquire.wait(2):
                        raise TimeoutError("blocked data request was not released")
                    return original_acquire_many(specs)

                server = NodeServer(
                    pool.index,
                    protocol_version=1,
                    load_token="load-0",
                    data_worker_count=1,
                    data_pending_requests=0,
                ).start()
                client = NodeClient(
                    server.data_base_url,
                    protocol_version=1,
                    load_token="load-0",
                    request_timeout=1,
                    max_attempts=1,
                )
                errors: list[Exception] = []

                def fetch() -> None:
                    try:
                        client.fetch_into(
                            [RangeRequest("file-0", 0, 4)],
                            bytearray(4),
                        )
                    except Exception as error:
                        errors.append(error)

                with mock.patch.object(
                    pool.index,
                    "acquire_many",
                    side_effect=block_acquire,
                ):
                    fetch_thread = threading.Thread(target=fetch)
                    fetch_thread.start()
                    self.assertTrue(acquire_started.wait(timeout=1))
                    close_thread = threading.Thread(target=server.close)
                    close_thread.start()
                    time.sleep(0.05)
                    release_acquire.set()
                    fetch_thread.join(timeout=1)
                    close_thread.join(timeout=1)

                client.close()
                self.assertFalse(fetch_thread.is_alive())
                self.assertFalse(close_thread.is_alive())
                self.assertEqual(1, len(errors))
                self.assertIsInstance(errors[0], TransportError)
