# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded shared-memory storage for cooperative checkpoint byte ranges."""

from __future__ import annotations

import mmap
import os
import re
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

DEFAULT_SHARED_MEMORY_FRACTION: float = 0.4
_CLEANUP_WORKERS = 8


class RangeNotReadyError(LookupError):
    pass


class RangeNotFoundError(LookupError):
    pass


class AmbiguousRangeError(LookupError):
    pass


class ReadableInto(Protocol):
    def readinto(self, buffer: memoryview) -> int | None: ...


@dataclass(frozen=True)
class RangeSpec:
    file_id: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        if not self.file_id:
            raise ValueError("file_id must not be empty")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")


@dataclass(frozen=True)
class SegmentSlice:
    path: Path
    file_offset: int
    length: int


@dataclass
class _SegmentRecord:
    owner_id: str
    file_id: str
    source_start: int
    source_end: int
    path: Path
    chunk_offset: int
    ready: bool = False


class SegmentReadLease:
    """Pins the reservations backing resolved ranges until the send completes."""

    def __init__(
        self,
        index: SegmentIndex,
        owners: tuple[str, ...],
        ranges: tuple[tuple[SegmentSlice, ...], ...],
    ) -> None:
        self._index = index
        self._owners = owners
        self.ranges = ranges
        self._closed = False
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._index._release_readers(self._owners)

    def __enter__(self) -> SegmentReadLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class SegmentIndex:
    """Thread-safe source-offset index over active shared-memory segments."""

    def __init__(self) -> None:
        self._owner_pid = os.getpid()
        self._segments: dict[str, list[_SegmentRecord]] = {}
        self._active_readers: dict[str, int] = {}
        self._retiring: set[str] = set()
        self._sealed = False
        self._condition = threading.Condition()

    def register_many(self, owner_id: str, records: Sequence[_SegmentRecord]) -> None:
        self._assert_owner_process()
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        with self._condition:
            self._assert_open_locked()
            if not records:
                return
            if owner_id in self._retiring:
                raise RuntimeError(f"reservation {owner_id!r} is retiring")
            combined: dict[str, list[_SegmentRecord]] = {}
            for record in records:
                if record.owner_id != owner_id:
                    raise ValueError("all segment records must have the same owner")
                combined.setdefault(record.file_id, []).append(record)
            for file_id, new_records in combined.items():
                candidates = [*self._segments.get(file_id, ()), *new_records]
                candidates.sort(key=lambda item: (item.source_start, item.source_end))
                for previous, current in zip(candidates, candidates[1:]):
                    if current.source_start < previous.source_end:
                        raise AmbiguousRangeError(
                            f"overlapping active segments for {file_id!r}: "
                            f"[{previous.source_start}, {previous.source_end}) and "
                            f"[{current.source_start}, {current.source_end})"
                        )
            for file_id, new_records in combined.items():
                entries = self._segments.setdefault(file_id, [])
                entries.extend(new_records)
                entries.sort(key=lambda item: item.source_start)
            self._active_readers.setdefault(owner_id, 0)
            self._condition.notify_all()

    def mark_ready(self, owner_id: str, file_id: str, offset: int, length: int) -> None:
        self._assert_owner_process()
        spec = RangeSpec(file_id, offset, length)
        with self._condition:
            self._assert_open_locked()
            records = self._covering_records_locked(spec, allow_unready=True)
            if any(record.owner_id != owner_id for record in records):
                raise RangeNotFoundError(
                    f"range {spec} is not wholly owned by {owner_id!r}"
                )
            if (
                records[0].source_start != spec.offset
                or records[-1].source_end != spec.offset + spec.length
            ):
                raise ValueError(
                    "ready range must exactly match one or more allocated segments"
                )
            for record in records:
                record.ready = True
            self._condition.notify_all()

    def acquire_many(self, specs: Sequence[RangeSpec]) -> SegmentReadLease:
        self._assert_owner_process()
        if not specs:
            raise ValueError("at least one range is required")
        with self._condition:
            self._assert_open_locked()
            resolved: list[tuple[SegmentSlice, ...]] = []
            owners: set[str] = set()
            for spec in specs:
                records = self._covering_records_locked(spec, allow_unready=False)
                current = spec.offset
                slices: list[SegmentSlice] = []
                for record in records:
                    start = max(current, record.source_start)
                    end = min(spec.offset + spec.length, record.source_end)
                    slices.append(
                        SegmentSlice(
                            path=record.path,
                            file_offset=record.chunk_offset
                            + (start - record.source_start),
                            length=end - start,
                        )
                    )
                    owners.add(record.owner_id)
                    current = end
                resolved.append(tuple(slices))
            ordered_owners = tuple(sorted(owners))
            for owner_id in ordered_owners:
                if owner_id in self._retiring:
                    raise RangeNotReadyError(
                        f"reservation {owner_id!r} is being retired"
                    )
            for owner_id in ordered_owners:
                self._active_readers[owner_id] += 1
            return SegmentReadLease(self, ordered_owners, tuple(resolved))

    def retire_owner(self, owner_id: str, timeout: float | None = None) -> None:
        self._assert_owner_process()
        deadline = _deadline(timeout)
        with self._condition:
            self._assert_open_locked()
            if owner_id not in self._active_readers:
                return
            self._retiring.add(owner_id)
            while self._active_readers[owner_id] > 0:
                remaining = _remaining(deadline)
                if remaining == 0:
                    raise TimeoutError(
                        f"timed out waiting for readers of reservation {owner_id!r}"
                    )
                self._condition.wait(remaining)
            for file_id, entries in tuple(self._segments.items()):
                retained = [entry for entry in entries if entry.owner_id != owner_id]
                if retained:
                    self._segments[file_id] = retained
                else:
                    del self._segments[file_id]
            del self._active_readers[owner_id]
            self._retiring.remove(owner_id)
            self._condition.notify_all()

    def assert_quiescent(self) -> None:
        """Reject indexes that still carry state from an active load."""

        self._assert_owner_process()
        with self._condition:
            self._assert_open_locked()
            self._assert_quiescent_locked()

    def _seal_for_reuse(self) -> None:
        self._assert_owner_process()
        with self._condition:
            self._assert_open_locked()
            self._assert_quiescent_locked()
            self._sealed = True

    def _assert_quiescent_locked(self) -> None:
        segment_count = sum(len(entries) for entries in self._segments.values())
        active_reader_count = sum(self._active_readers.values())
        if (
            segment_count
            or self._active_readers
            or active_reader_count
            or self._retiring
        ):
            raise RuntimeError(
                "segment index is not quiescent: "
                f"segments={segment_count}, "
                f"owners={len(self._active_readers)}, "
                f"active_readers={active_reader_count}, "
                f"retiring_owners={len(self._retiring)}"
            )

    def _assert_open_locked(self) -> None:
        if self._sealed:
            raise RuntimeError("segment index has been sealed for pool reuse")

    def _assert_owner_process(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                "segment index cannot be used after fork: "
                f"created in process {self._owner_pid}, used in {current_pid}"
            )

    def _covering_records_locked(
        self, spec: RangeSpec, *, allow_unready: bool
    ) -> list[_SegmentRecord]:
        entries = self._segments.get(spec.file_id, ())
        end = spec.offset + spec.length
        current = spec.offset
        result: list[_SegmentRecord] = []
        while current < end:
            candidates = [
                entry
                for entry in entries
                if entry.source_start <= current < entry.source_end
            ]
            if not candidates:
                raise RangeNotFoundError(
                    f"no shared-memory segment covers {spec.file_id!r} at {current}"
                )
            if len(candidates) != 1:
                raise AmbiguousRangeError(
                    f"multiple shared-memory segments cover {spec.file_id!r} "
                    f"at {current}"
                )
            record = candidates[0]
            if record.owner_id in self._retiring or (
                not allow_unready and not record.ready
            ):
                raise RangeNotReadyError(
                    f"range {spec.file_id!r} at {current} is not ready"
                )
            result.append(record)
            current = min(end, record.source_end)
        return result

    def _release_readers(self, owners: tuple[str, ...]) -> None:
        self._assert_owner_process()
        with self._condition:
            self._assert_open_locked()
            for owner_id in owners:
                readers = self._active_readers.get(owner_id)
                if readers is None or readers <= 0:
                    raise RuntimeError(
                        f"reader accounting underflow for reservation {owner_id!r}"
                    )
                self._active_readers[owner_id] = readers - 1
            self._condition.notify_all()


class _Chunk:
    def __init__(self, index: int, path: Path, file_descriptor: int, size: int) -> None:
        self._owner_pid = os.getpid()
        self.index = index
        self.path = path
        self.file_descriptor = file_descriptor
        self.size = size
        self.mapping = mmap.mmap(file_descriptor, size, access=mmap.ACCESS_WRITE)
        backing_stat = os.fstat(file_descriptor)
        self._backing_device = backing_stat.st_dev
        self._backing_inode = backing_stat.st_ino
        self.reuse_safe = _apply_mapping_advice(
            self.mapping, getattr(mmap, "MADV_DONTFORK", None)
        )
        _apply_mapping_advice(self.mapping, getattr(mmap, "MADV_DONTDUMP", None))
        self._mapping_closed = False
        self._descriptor_closed = False

    def backing_file_is_intact(self) -> bool:
        try:
            path_stat = os.stat(self.path, follow_symlinks=False)
            descriptor_stat = os.fstat(self.file_descriptor)
        except OSError:
            return False
        return (
            stat.S_ISREG(path_stat.st_mode)
            and path_stat.st_dev == self._backing_device
            and path_stat.st_ino == self._backing_inode
            and path_stat.st_size == self.size
            and descriptor_stat.st_dev == self._backing_device
            and descriptor_stat.st_ino == self._backing_inode
            and descriptor_stat.st_size == self.size
        )

    def assert_owner_process(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                "shared-memory chunk cannot be used after fork: "
                f"created in process {self._owner_pid}, used in {current_pid}"
            )

    def close(self) -> None:
        if not self._mapping_closed:
            self.mapping.close()
            self._mapping_closed = True
        if not self._descriptor_closed:
            os.close(self.file_descriptor)
            self._descriptor_closed = True


class WritableSegment:
    """Writable view metadata for one contiguous shared-memory segment."""

    def __init__(
        self,
        *,
        file_id: str,
        source_offset: int,
        length: int,
        chunk: _Chunk,
        chunk_offset: int,
    ) -> None:
        self.file_id = file_id
        self.source_offset = source_offset
        self.length = length
        self.path = chunk.path
        self.chunk_offset = chunk_offset
        self._chunk = chunk

    def _view(self) -> memoryview:
        self._chunk.assert_owner_process()
        return memoryview(self._chunk.mapping)[
            self.chunk_offset : self.chunk_offset + self.length
        ]


@dataclass(frozen=True)
class RegisteredRange:
    """One logical source range registered before its bytes are written."""

    owner_id: str
    file_id: str
    source_offset: int
    length: int
    segments: tuple[WritableSegment, ...]


class ChunkReservation:
    """Exclusive reservation of whole chunks from a :class:`ChunkPool`."""

    def __init__(
        self,
        pool: ChunkPool,
        owner_id: str,
        required_bytes: int,
        chunks: tuple[_Chunk, ...],
    ) -> None:
        self._pool = pool
        self.owner_id = owner_id
        self.required_bytes = required_bytes
        self._chunks = chunks
        self._cursor = 0
        self._retired = False
        self._retiring = False
        self._active_writers = 0
        self._registered_ranges: dict[int, RegisteredRange] = {}
        self._claimed_ranges: set[int] = set()
        self._condition = threading.Condition()

    @property
    def capacity_bytes(self) -> int:
        return sum(chunk.size for chunk in self._chunks)

    @property
    def used_bytes(self) -> int:
        with self._condition:
            return self._cursor

    def allocate(
        self, file_id: str, source_offset: int, length: int
    ) -> tuple[WritableSegment, ...]:
        registered = self.register_many((RangeSpec(file_id, source_offset, length),))
        return registered[0].segments

    def register_many(self, specs: Sequence[RangeSpec]) -> tuple[RegisteredRange, ...]:
        """Atomically register logical ranges without marking them ready."""

        requested = tuple(specs)
        if not requested:
            return ()
        with self._condition:
            self._assert_active()
            total_bytes = sum(spec.length for spec in requested)
            if self._cursor + total_bytes > self.required_bytes:
                raise MemoryError(
                    f"reservation {self.owner_id!r} requested {total_bytes} more "
                    f"bytes with only {self.required_bytes - self._cursor} bytes "
                    "unallocated"
                )
            cursor = self._cursor
            registered: list[RegisteredRange] = []
            records: list[_SegmentRecord] = []
            for spec in requested:
                segments = tuple(self._segments_at_cursor(spec, cursor))
                registered_range = RegisteredRange(
                    owner_id=self.owner_id,
                    file_id=spec.file_id,
                    source_offset=spec.offset,
                    length=spec.length,
                    segments=segments,
                )
                registered.append(registered_range)
                records.extend(
                    _SegmentRecord(
                        owner_id=self.owner_id,
                        file_id=segment.file_id,
                        source_start=segment.source_offset,
                        source_end=segment.source_offset + segment.length,
                        path=segment.path,
                        chunk_offset=segment.chunk_offset,
                    )
                    for segment in segments
                )
                cursor += spec.length
            self._pool.index.register_many(self.owner_id, records)
            self._cursor = cursor
            for registered_range in registered:
                self._registered_ranges[id(registered_range)] = registered_range
            return tuple(registered)

    def write_from(
        self,
        file_id: str,
        source_offset: int,
        reader: ReadableInto,
        length: int,
        *,
        mark_ready: bool = True,
    ) -> tuple[WritableSegment, ...]:
        registered = self.register_many((RangeSpec(file_id, source_offset, length),))[0]
        self.write_registered(registered, reader, mark_ready=mark_ready)
        return registered.segments

    def write_registered(
        self,
        registered: RegisteredRange,
        reader: ReadableInto,
        *,
        mark_ready: bool = True,
    ) -> None:
        """Fill one preregistered range and publish readiness after the full read."""

        registration_id = id(registered)
        with self._condition:
            self._assert_active()
            if (
                self._registered_ranges.get(registration_id) is not registered
                or registered.owner_id != self.owner_id
            ):
                raise ValueError(
                    f"range is not registered to reservation {self.owner_id!r}"
                )
            if registration_id in self._claimed_ranges:
                raise RuntimeError("registered range has already been claimed")
            self._claimed_ranges.add(registration_id)
            self._active_writers += 1
        try:
            for segment in registered.segments:
                view = segment._view()
                try:
                    readinto_exact(reader, view)
                finally:
                    view.release()
            if mark_ready:
                self._pool.index.mark_ready(
                    self.owner_id,
                    registered.file_id,
                    registered.source_offset,
                    registered.length,
                )
        finally:
            with self._condition:
                self._active_writers -= 1
                self._condition.notify_all()

    def write_registered_progressively(
        self,
        registered: RegisteredRange,
        reader: ReadableInto,
    ) -> None:
        """Fill one preregistered range and publish each completed segment."""

        registration_id = id(registered)
        with self._condition:
            self._assert_active()
            if (
                self._registered_ranges.get(registration_id) is not registered
                or registered.owner_id != self.owner_id
            ):
                raise ValueError(
                    f"range is not registered to reservation {self.owner_id!r}"
                )
            if registration_id in self._claimed_ranges:
                raise RuntimeError("registered range has already been claimed")
            self._claimed_ranges.add(registration_id)
            self._active_writers += 1
        try:
            for segment in registered.segments:
                view = segment._view()
                try:
                    readinto_exact(reader, view)
                finally:
                    view.release()
                self._pool.index.mark_ready(
                    self.owner_id,
                    segment.file_id,
                    segment.source_offset,
                    segment.length,
                )
        finally:
            with self._condition:
                self._active_writers -= 1
                self._condition.notify_all()

    def mark_ready(self, file_id: str, source_offset: int, length: int) -> None:
        with self._condition:
            self._assert_active()
            self._pool.index.mark_ready(self.owner_id, file_id, source_offset, length)

    def retire(self, timeout: float | None = None) -> None:
        self._pool._retire(self, timeout)

    def _segments_at_cursor(
        self, spec: RangeSpec, cursor: int
    ) -> list[WritableSegment]:
        remaining = spec.length
        logical_offset = spec.offset
        reservation_offset = cursor
        segments: list[WritableSegment] = []
        for chunk in self._chunks:
            if reservation_offset >= chunk.size:
                reservation_offset -= chunk.size
                continue
            length = min(remaining, chunk.size - reservation_offset)
            segments.append(
                WritableSegment(
                    file_id=spec.file_id,
                    source_offset=logical_offset,
                    length=length,
                    chunk=chunk,
                    chunk_offset=reservation_offset,
                )
            )
            remaining -= length
            logical_offset += length
            reservation_offset = 0
            if remaining == 0:
                return segments
        raise MemoryError(
            f"reservation {self.owner_id!r} does not have {spec.length} writable bytes"
        )

    def _assert_active(self) -> None:
        self._pool._assert_owner_process()
        if self._retired or self._retiring:
            raise RuntimeError(
                f"reservation {self.owner_id!r} is retiring or has been retired"
            )


class ChunkPool:
    """A fixed-capacity, whole-chunk reservation pool backed by tmpfs mmaps."""

    def __init__(
        self,
        *,
        capacity_bytes: int,
        chunk_bytes: int,
        job_token: str,
        directory: str | os.PathLike[str] = "/dev/shm",
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if not job_token:
            raise ValueError("job_token must not be empty")
        self.capacity_bytes = capacity_bytes
        self.chunk_bytes = chunk_bytes
        self._owner_pid = os.getpid()
        self.index = SegmentIndex()
        self._condition = threading.Condition()
        self._reservations: dict[str, ChunkReservation] = {}
        self._cleanup_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._job_directory = Path(
            tempfile.mkdtemp(
                prefix=f"python_coop_{_sanitize_token(job_token)}_",
                dir=os.fspath(directory),
            )
        )
        self._chunks: tuple[_Chunk, ...] = ()
        self._free_chunks: list[_Chunk] = []
        try:
            self._chunks = self._create_chunks()
            self._free_chunks = list(self._chunks)
        except Exception:
            self._cleanup_files()
            raise

    @property
    def job_directory(self) -> Path:
        return self._job_directory

    @property
    def chunk_count(self) -> int:
        with self._condition:
            return len(self._chunks)

    @property
    def active_reservation_count(self) -> int:
        with self._condition:
            return len(self._reservations)

    @property
    def reuse_supported(self) -> bool:
        if os.getpid() != self._owner_pid:
            return False
        with self._condition:
            return (
                not self._closing
                and not self._closed
                and bool(self._chunks)
                and self._job_directory.is_dir()
                and all(
                    chunk.reuse_safe and chunk.backing_file_is_intact()
                    for chunk in self._chunks
                )
            )

    def reserve(
        self,
        owner_id: str,
        required_bytes: int,
        *,
        timeout: float | None = None,
    ) -> ChunkReservation:
        self._assert_owner_process()
        if not owner_id:
            raise ValueError("owner_id must not be empty")
        if required_bytes <= 0:
            raise ValueError("required_bytes must be positive")
        if required_bytes > self.capacity_bytes:
            raise MemoryError(
                f"request for {required_bytes} bytes exceeds pool capacity "
                f"of {self.capacity_bytes} bytes"
            )
        deadline = _deadline(timeout)
        with self._condition:
            while True:
                if owner_id in self._reservations:
                    raise ValueError(f"reservation {owner_id!r} already exists")
                if self._closing or self._closed:
                    raise RuntimeError("chunk pool is closing")
                selected = _select_chunks(self._free_chunks, required_bytes)
                if selected is not None:
                    selected_ids = {chunk.index for chunk in selected}
                    self._free_chunks = [
                        chunk
                        for chunk in self._free_chunks
                        if chunk.index not in selected_ids
                    ]
                    reservation = ChunkReservation(
                        self, owner_id, required_bytes, tuple(selected)
                    )
                    self._reservations[owner_id] = reservation
                    return reservation
                remaining = _remaining(deadline)
                if remaining == 0:
                    raise TimeoutError(
                        f"timed out reserving {required_bytes} bytes for {owner_id!r}"
                    )
                self._condition.wait(remaining)

    def prepare_for_reuse(self) -> SegmentIndex:
        """Reset logical state while retaining the pool's mapped backing files."""

        self._assert_owner_process()
        with self._cleanup_lock:
            with self._condition:
                if self._closing or self._closed:
                    raise RuntimeError("chunk pool is closing")
                if not all(chunk.reuse_safe for chunk in self._chunks):
                    raise RuntimeError(
                        "chunk pool reuse requires MADV_DONTFORK on every mapping"
                    )
                if not self._job_directory.is_dir() or not all(
                    chunk.backing_file_is_intact() for chunk in self._chunks
                ):
                    raise RuntimeError("chunk pool reuse requires intact backing files")
                if self._reservations:
                    raise RuntimeError(
                        "chunk pool cannot be reused with active reservations: "
                        f"{len(self._reservations)}"
                    )
                if tuple(self._free_chunks) != self._chunks:
                    raise RuntimeError(
                        "chunk pool cannot be reused unless every chunk is free"
                    )
                previous_index = self.index
                previous_index._seal_for_reuse()
                self.index = SegmentIndex()
                return self.index

    def cleanup(self, timeout: float | None = None) -> None:
        if os.getpid() != self._owner_pid:
            return
        with self._cleanup_lock:
            with self._condition:
                if self._closed:
                    return
                self._closing = True
                reservations = tuple(self._reservations.values())
                self._condition.notify_all()
            deadline = _deadline(timeout)
            for reservation in reservations:
                reservation.retire(_remaining(deadline))
            self._cleanup_files()
            with self._condition:
                self._closed = True

    def _retire(self, reservation: ChunkReservation, timeout: float | None) -> None:
        self._assert_owner_process()
        deadline = _deadline(timeout)
        with reservation._condition:
            if reservation._retired:
                return
            reservation._retiring = True
            while reservation._active_writers > 0:
                remaining = _remaining(deadline)
                if remaining == 0:
                    raise TimeoutError(
                        f"timed out waiting for writers of reservation "
                        f"{reservation.owner_id!r}"
                    )
                reservation._condition.wait(remaining)
            with self._condition:
                active = self._reservations.get(reservation.owner_id)
                if active is not reservation:
                    raise RuntimeError(
                        f"reservation identity mismatch for {reservation.owner_id!r}"
                    )
            self.index.retire_owner(reservation.owner_id, _remaining(deadline))
            with self._condition:
                active = self._reservations.pop(reservation.owner_id, None)
                if active is reservation:
                    self._free_chunks.extend(reservation._chunks)
                    self._free_chunks.sort(key=lambda chunk: chunk.index)
                reservation._retired = True
                self._condition.notify_all()

    def _create_chunks(self) -> tuple[_Chunk, ...]:
        chunks: list[_Chunk] = []
        remaining = self.capacity_bytes
        try:
            while remaining > 0:
                size = min(self.chunk_bytes, remaining)
                file_descriptor, raw_path = tempfile.mkstemp(
                    prefix=f"chunk_{len(chunks):05d}_",
                    suffix=".dat",
                    dir=self._job_directory,
                )
                path = Path(raw_path)
                try:
                    os.ftruncate(file_descriptor, size)
                    chunks.append(_Chunk(len(chunks), path, file_descriptor, size))
                except Exception:
                    os.close(file_descriptor)
                    path.unlink(missing_ok=True)
                    raise
                remaining -= size
            return tuple(chunks)
        except Exception:
            for chunk in chunks:
                chunk.close()
                chunk.path.unlink(missing_ok=True)
            raise

    def _cleanup_files(self) -> None:
        with self._condition:
            chunks = self._chunks
        if len(chunks) <= 1:
            errors = tuple(_cleanup_chunk(chunk) for chunk in chunks)
        else:
            try:
                with ThreadPoolExecutor(
                    max_workers=min(_CLEANUP_WORKERS, len(chunks)),
                    thread_name_prefix="cooperative-shm-cleanup",
                ) as executor:
                    errors = tuple(executor.map(_cleanup_chunk, chunks))
            except RuntimeError:
                # Python shuts down its global thread-pool machinery before
                # running application atexit handlers. Cleanup must still
                # remove retained shared-memory files during interpreter exit.
                errors = tuple(_cleanup_chunk(chunk) for chunk in chunks)
        first_error = next((error for error in errors if error is not None), None)
        failed_chunks = [
            chunk for chunk, error in zip(chunks, errors) if error is not None
        ]
        try:
            self._job_directory.rmdir()
        except FileNotFoundError:
            pass
        except Exception as error:
            first_error = first_error or error
        with self._condition:
            self._chunks = tuple(failed_chunks)
            self._free_chunks = []
        if first_error is not None:
            raise first_error

    def _assert_owner_process(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                "chunk pool cannot be used after fork: "
                f"created in process {self._owner_pid}, used in {current_pid}"
            )

    def __enter__(self) -> ChunkPool:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.cleanup()


def _cleanup_chunk(chunk: _Chunk) -> Exception | None:
    first_error: Exception | None = None
    try:
        chunk.close()
    except Exception as error:
        first_error = error
    try:
        chunk.path.unlink(missing_ok=True)
    except Exception as error:
        first_error = first_error or error
    return first_error


def _apply_mapping_advice(mapping: mmap.mmap, advice: int | None) -> bool:
    if advice is None:
        return False
    madvise = getattr(mapping, "madvise", None)
    if madvise is None:
        return False
    try:
        madvise(advice)
    except (OSError, ValueError):
        return False
    return True


def readinto_exact(reader: ReadableInto, destination: memoryview) -> int:
    """Fill ``destination`` through ``readinto`` or fail on a short read."""

    if destination.readonly:
        raise TypeError("destination must be writable")
    if not destination.c_contiguous:
        raise TypeError("destination must be C-contiguous")
    byte_view = destination.cast("B")
    total = 0
    try:
        while total < len(byte_view):
            remaining = byte_view[total:]
            try:
                count = reader.readinto(remaining)
            finally:
                remaining.release()
            if count is None or count <= 0:
                raise EOFError(
                    f"short read after {total} of {len(byte_view)} requested bytes"
                )
            if count > len(byte_view) - total:
                raise OSError("readinto returned more bytes than the destination size")
            total += count
        return total
    finally:
        if byte_view is not destination:
            byte_view.release()


def recommended_capacity_bytes(
    *,
    fraction: float = DEFAULT_SHARED_MEMORY_FRACTION,
    directory: str | os.PathLike[str] = "/dev/shm",
) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    stats = os.statvfs(directory)
    total = stats.f_blocks * stats.f_frsize
    available = stats.f_bavail * stats.f_frsize
    return min(int(total * fraction), available)


def _select_chunks(
    chunks: Sequence[_Chunk], required_bytes: int
) -> list[_Chunk] | None:
    selected: list[_Chunk] = []
    available = 0
    for chunk in chunks:
        selected.append(chunk)
        available += chunk.size
        if available >= required_bytes:
            return selected
    return None


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", value)[:64]
    return sanitized or "load"


def _deadline(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    return time.monotonic() + timeout


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
