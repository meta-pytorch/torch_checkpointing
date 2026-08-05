# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-serialized load helpers that target a ``Storage`` backend.

Provides:
- ``load_torch_serialized_from_storage()``: Loads a ``torch.save``'d checkpoint
  via a ``Storage`` backend using a single contiguous ``UntypedStorage`` to
  back all loaded tensors as slices — mmap-like memory behavior for backends
  where ``torch.load(path, mmap=True)`` can't be used directly (e.g., streaming
  storage backends that expose bytes via ``stream_read`` rather than a real
  file descriptor).
"""

import concurrent.futures
import logging
import mmap
import os
import pickle
import time
from pathlib import Path
from typing import Any

import torch

from .base_storage import ReadArgs, Storage

logger: logging.Logger = logging.getLogger(__name__)

# Number of concurrent ranged reads used to fill the mmap. Backends whose
# per-stream throughput is latency-bound rather than bandwidth-bound gain from
# more than one, because a single stream leaves the link idle between round
# trips. Set to 1 to restore the historical single-stream behavior.
_PARALLEL_FILL_WORKERS_ENV = "TORCH_CKPT_PARALLEL_FILL_WORKERS"
_DEFAULT_PARALLEL_FILL_WORKERS = 8
# Size of each ranged read when filling concurrently.
_PARALLEL_FILL_CHUNK_MB_ENV = "TORCH_CKPT_PARALLEL_FILL_CHUNK_MB"
_DEFAULT_PARALLEL_FILL_CHUNK_MB = 64


def _mmap_as_storage(mm_file: mmap.mmap) -> torch.UntypedStorage:
    """Return an UntypedStorage view over an mmap without copying bytes."""
    # ``torch.serialization._load`` treats ``overall_storage`` as raw checkpoint
    # bytes: it slices by zip-record byte offsets and then wraps each slice in a
    # ``TypedStorage`` with that tensor's checkpoint dtype. The wrapper over the
    # mmap must therefore be uint8 regardless of the tensors saved inside it.
    return torch.frombuffer(mm_file, dtype=torch.uint8).untyped_storage()


def _fill_serial(
    buf: memoryview,
    expected: int,
    path: Path,
    storage: Storage,
    read_args: ReadArgs,
) -> int:
    """Fill ``buf`` with one sequential stream. Returns bytes written."""
    pos = 0
    with storage.stream_read(path, read_args) as stream:
        while pos < expected:
            n = stream.readinto(buf[pos:])
            if not n:
                break
            pos += n
    return pos


def _fill_parallel(
    buf: memoryview,
    expected: int,
    path: Path,
    storage: Storage,
    read_args: ReadArgs,
    *,
    workers: int,
    chunk_bytes: int,
) -> int:
    """Fill ``buf`` with concurrent ranged reads. Returns bytes written.

    Each worker opens its own stream and seeks to its own offset — a single
    stream cannot be shared, because seek-then-read is not atomic. The target
    ranges are disjoint slices of the same buffer, so no locking is needed on
    the write side.
    """
    ranges = [
        (offset, min(chunk_bytes, expected - offset))
        for offset in range(0, expected, chunk_bytes)
    ]

    def fill_range(args: tuple[int, int]) -> int:
        offset, length = args
        got = 0
        with storage.stream_read(path, read_args) as stream:
            # Check where the seek landed, not just how many bytes came back: a
            # backend that clamps or narrows the offset returns a full-length
            # read of the WRONG bytes, and nothing downstream checksums it
            # (``torch.serialization._load`` slices this buffer raw), so the
            # ``pos != expected`` guard below proves length only.
            #
            # This catches only a backend that reports a bad position honestly.
            # One that returns the requested offset while reading from somewhere
            # else is indistinguishable from a correct read here.
            landed = stream.seek(offset)
            if landed != offset:
                raise OSError(
                    f"seek to offset {offset} in {path} landed at {landed}; "
                    "backend cannot position reads reliably"
                )
            while got < length:
                n = stream.readinto(buf[offset + got : offset + length])
                if not n:
                    break
                got += n
        return got

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="ckpt-fill"
    ) as pool:
        return sum(pool.map(fill_range, ranges))


def _stream_storage_into_mmap(
    mm_file: mmap.mmap,
    path: Path,
    storage: Storage,
    *,
    direct_io: bool = False,
    num_workers: int | None = None,
) -> None:
    """Stream ``path`` from ``storage`` directly into the backing memory
    of ``mm_file`` via ``readinto``. For a backend whose ``stream.readinto``
    is a zero-copy copy into the caller-provided buffer, only one transient
    chunk is the working set during the fill. The same mmap later backs both
    the PyTorch zip reader and the returned tensors.

    This issues ``TORCH_CKPT_PARALLEL_FILL_WORKERS`` concurrent ranged reads of
    ``TORCH_CKPT_PARALLEL_FILL_CHUNK_MB`` each, which is faster on backends
    where a single stream is limited by per-request latency rather than by
    available bandwidth. One worker gives the historical single sequential
    stream. If the backend's stream cannot seek -- or cannot seek accurately --
    this falls back to the sequential fill.

    Args:
        num_workers: Concurrent ranged reads to use. ``None`` (the default)
            reads ``TORCH_CKPT_PARALLEL_FILL_WORKERS``. Callers that ALREADY
            load files concurrently should pass 1, so the two levels of
            parallelism do not multiply into ``outer_threads x num_workers``
            in-flight reads against one storage client.
    """
    expected = len(mm_file)
    buf = memoryview(mm_file)
    # This helper is the full-file read: enabling ``pre_read_full_file`` would
    # allocate a second whole-file buffer inside the storage layer before we copy
    # into the mmap.
    read_args = ReadArgs(pre_read_full_file=False, direct_io=direct_io)
    workers = (
        num_workers
        if num_workers is not None
        else int(
            os.environ.get(_PARALLEL_FILL_WORKERS_ENV, _DEFAULT_PARALLEL_FILL_WORKERS)
        )
    )
    chunk_mb = int(
        os.environ.get(_PARALLEL_FILL_CHUNK_MB_ENV, _DEFAULT_PARALLEL_FILL_CHUNK_MB)
    )
    # Not an ``assert``: ``python -O`` strips those, and a zero chunk size would
    # then surface as an opaque ``range()`` error instead of a named cause.
    if workers < 1 or chunk_mb < 1:
        raise ValueError(
            f"{_PARALLEL_FILL_WORKERS_ENV}={workers} and "
            f"{_PARALLEL_FILL_CHUNK_MB_ENV}={chunk_mb} must both be positive"
        )
    chunk_bytes = chunk_mb * 1024 * 1024
    pos = 0
    mode = "serial"
    attempted_parallel = False
    fill_t0 = time.time()
    try:
        if workers > 1 and expected > chunk_bytes:
            attempted_parallel = True
            try:
                pos = _fill_parallel(
                    buf,
                    expected,
                    path,
                    storage,
                    read_args,
                    workers=workers,
                    chunk_bytes=chunk_bytes,
                )
                mode = "parallel"
            except OSError:
                # A backend that cannot seek (or fails mid-range) must not break
                # the load: redo the whole fill sequentially, which overwrites
                # anything the partial parallel attempt wrote. Not ``Exception``
                # -- a bug in here should not silently degrade every load.
                logger.warning(
                    "parallel fill of %s failed; falling back to sequential read",
                    path,
                    exc_info=True,
                )
                pos = 0
                mode = "serial-after-parallel-failure"
        if pos != expected:
            # A parallel fill can also come up short WITHOUT raising, in which
            # case the sequential retry below is what actually produced the
            # bytes. Say so, rather than leaving ``mode=parallel`` on a load that
            # finished serially.
            if mode == "parallel":
                mode = "serial-after-short-parallel"
            pos = _fill_serial(buf, expected, path, storage, read_args)
    finally:
        buf.release()
    # Which path actually ran is not otherwise observable: a silent fallback to
    # the sequential fill differs from the parallel one only in wall clock, so
    # without this an operator cannot tell "concurrency is on" from "concurrency
    # quietly turned itself off".
    elapsed = time.time() - fill_t0
    logger.info(
        "checkpoint mmap fill: mode=%s bytes=%d elapsed=%.2fs rate_mbps=%.1f "
        "workers=%d chunk_mb=%d ranges=%d path=%s",
        mode,
        pos,
        elapsed,
        (pos / (1024 * 1024)) / elapsed if elapsed > 0 else -1.0,
        workers,
        chunk_bytes // (1024 * 1024),
        -(-expected // chunk_bytes) if attempted_parallel else 1,
        path,
    )
    if pos != expected:
        raise IOError(
            f"short read from storage for {path}: got {pos} bytes, expected {expected}"
        )
    mm_file.seek(0)


def load_torch_serialized_from_storage(
    path: Path,
    storage: Storage,
    *,
    map_location: Any = "cpu",
    direct_io: bool = False,
    num_workers: int | None = None,
) -> Any:
    """Load a torch.save'd zipfile checkpoint from a ``Storage`` backend
    using a single contiguous ``torch.UntypedStorage`` to back all
    loaded tensors as slices.

    This is the equivalent of ``torch.load(path, mmap=True)`` for
    backends where torch cannot directly mmap the file. The motivating
    case is a streaming storage backend that exposes bytes via a
    ``stream_read`` interface — no file descriptor, no mmap-able
    handle — so ``torch.load(..., mmap=True)`` raises
    ``ValueError("f must be a file path in order to use the mmap
    argument")``.

    The default ``torch.load(BytesIO)`` path allocates one fresh CPU
    storage per tensor via ``c10::alloc_cpu``; on a real model state_dict
    with hundreds of tensors, the medium-sized buffers interleave with
    small C++ ``StorageImpl`` / Python wrapper allocations in glibc's
    heap arena. After the caller drops the loaded dict, the small
    interleaved allocations act as fragmentation anchors that prevent
    ``malloc_trim`` from shrinking the heap, leaving ~40-65% of file
    size stranded in process RSS for the rest of the process's lifetime.

    This function avoids that by:
      1. Allocating one large anonymous ``mmap(file_size)``. The
         allocation is large enough that glibc routes it through ``mmap``
         directly (above ``M_MMAP_THRESHOLD``, typically 128KB), so
         it gets its own VMA.
      2. Streaming the entire file from ``storage`` into that mmap via
         ``readinto`` (no intermediate Python buffer).
      3. Wrapping the mmap with ``torch.frombuffer(...).untyped_storage()``
         and calling ``torch.serialization._load`` with
         ``overall_storage=overall_storage``. Every per-tensor storage
         becomes a slice into the one big allocation — zero per-tensor
         ``c10::alloc_cpu`` calls.

    Memory profile: peak ≈ 1× file size, post-load residual ≈ 0.

    Args:
        path: Path to the checkpoint file (as understood by ``storage``).
        storage: A ``Storage`` backend with ``stream_read``, ``getsize``
            (or the path's ``stat().st_size`` if the path is a real fs path).
        map_location: Forwarded to ``torch.serialization._load``.
        direct_io: Whether to enable ``direct_io`` on the underlying
            ``Storage.stream_read`` read. Forwarded into ``ReadArgs``.
            Defaults to ``False``. Callers that previously read the same
            checkpoint via ``ReadArgs(direct_io=...)`` should pass the same
            value here to preserve that behavior on the mmap-backed path.
        num_workers: Concurrent ranged reads used to fill the mmap. ``None``
            (the default) falls back to ``TORCH_CKPT_PARALLEL_FILL_WORKERS``.
            Pass 1 explicitly from callers that already load several files
            concurrently, so the two levels of parallelism do not multiply.

    Returns:
        The deserialized Python object (typically a state_dict). Every
        tensor's storage is a slice into the single ``UntypedStorage``
        that this function allocated; the storage is released (and the
        underlying mapping unmapped) when the returned object is dropped.
    """
    # Private API: matches what torch.load(mmap=True) calls internally.
    from torch.serialization import _load, _open_zipfile_reader

    file_size = storage.getsize(path)
    mm_file = mmap.mmap(-1, file_size)
    _stream_storage_into_mmap(
        mm_file, path, storage, direct_io=direct_io, num_workers=num_workers
    )
    overall_storage = _mmap_as_storage(mm_file)
    with _open_zipfile_reader(mm_file) as zip_file:
        return _load(
            zip_file,
            map_location=map_location,
            pickle_module=pickle,
            overall_storage=overall_storage,
        )
