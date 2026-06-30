"""Torch-serialized load helpers that target a ``Storage`` backend.

Provides:
- ``load_torch_serialized_from_storage()``: Loads a ``torch.save``'d checkpoint
  via a ``Storage`` backend using a single contiguous ``UntypedStorage`` to
  back all loaded tensors as slices — mmap-like memory behavior for backends
  where ``torch.load(path, mmap=True)`` can't be used directly (e.g., streaming
  storage backends that expose bytes via ``stream_read`` rather than a real
  file descriptor).
"""

import mmap
import pickle
from pathlib import Path
from typing import Any

import torch

from .base_storage import ReadArgs, Storage


def _mmap_as_storage(mm_file: mmap.mmap) -> torch.UntypedStorage:
    """Return an UntypedStorage view over an mmap without copying bytes."""
    # ``torch.serialization._load`` treats ``overall_storage`` as raw checkpoint
    # bytes: it slices by zip-record byte offsets and then wraps each slice in a
    # ``TypedStorage`` with that tensor's checkpoint dtype. The wrapper over the
    # mmap must therefore be uint8 regardless of the tensors saved inside it.
    return torch.frombuffer(mm_file, dtype=torch.uint8).untyped_storage()


def _stream_storage_into_mmap(
    mm_file: mmap.mmap,
    path: Path,
    storage: Storage,
    *,
    direct_io: bool = False,
) -> None:
    """Stream ``path`` from ``storage`` directly into the backing memory
    of ``mm_file`` via ``readinto``. For a backend whose ``stream.readinto``
    is a zero-copy copy into the caller-provided buffer, only one transient
    chunk is the working set during the fill. The same mmap later backs both
    the PyTorch zip reader and the returned tensors.
    """
    expected = len(mm_file)
    buf = memoryview(mm_file)
    # This helper is the full-file read: enabling ``pre_read_full_file`` would
    # allocate a second whole-file buffer inside the storage layer before we copy
    # into the mmap.
    read_args = ReadArgs(pre_read_full_file=False, direct_io=direct_io)
    pos = 0
    try:
        with storage.stream_read(path, read_args) as stream:
            while pos < expected:
                n = stream.readinto(buf[pos:])
                if not n:
                    break
                pos += n
    finally:
        buf.release()
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
    _stream_storage_into_mmap(mm_file, path, storage, direct_io=direct_io)
    overall_storage = _mmap_as_storage(mm_file)
    with _open_zipfile_reader(mm_file) as zip_file:
        return _load(
            zip_file,
            map_location=map_location,
            pickle_module=pickle,
            overall_storage=overall_storage,
        )
