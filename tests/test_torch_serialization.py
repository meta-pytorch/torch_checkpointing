# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import io
import mmap
from pathlib import Path
from typing import Any

import torch
from torch_checkpointing.storage.torch_serialization import (
    _mmap_as_storage,
    load_torch_serialized_from_storage,
)


class _BytesStorage:
    """Minimal ``Storage`` stand-in backed by in-memory bytes.

    Records the ``ReadArgs`` of every ``stream_read``, so a test can assert both
    how the read was configured and how many streams the fill opened.
    """

    def __init__(self, data_by_path: dict[Path, bytes]) -> None:
        self._data_by_path = data_by_path
        self.read_args: list[Any] = []

    def stream_read(self, path: Path, read_args: Any | None = None) -> io.BytesIO:
        self.read_args.append(read_args)
        return io.BytesIO(self._data_by_path[path])

    def getsize(self, path: Path) -> int:
        return len(self._data_by_path[path])


def test_mmap_as_storage_aliases_buffer_without_copy() -> None:
    mm_file = mmap.mmap(-1, 5)
    storage = None
    tensor = None
    try:
        mm_file[:] = b"abcde"
        storage = _mmap_as_storage(mm_file)
        tensor = torch.empty(0, dtype=torch.uint8).set_(storage)

        assert bytes(tensor.tolist()) == b"abcde"
        mm_file[0] = ord("z")
        assert int(tensor[0]) == ord("z")
        tensor[1] = ord("y")
        assert mm_file[:5] == b"zycde"
    finally:
        del tensor
        del storage
        gc.collect()
        mm_file.close()


def test_mmap_as_storage_keeps_backing_mmap_alive_after_local_refs_drop() -> None:
    def make_storage() -> torch.UntypedStorage:
        mm_file = mmap.mmap(-1, 4)
        mm_file[:] = b"abcd"
        storage = _mmap_as_storage(mm_file)
        del mm_file
        gc.collect()
        return storage

    storage = make_storage()
    tensor = torch.empty(0, dtype=torch.uint8).set_(storage)

    try:
        assert bytes(tensor.tolist()) == b"abcd"
        tensor[0] = ord("z")
        assert bytes(tensor.tolist()) == b"zbcd"
    finally:
        del tensor
        del storage
        gc.collect()


def test_load_torch_serialized_from_storage_round_trips_from_streamed_mmap() -> None:
    expected = {
        "weights": {"a": torch.arange(5, dtype=torch.float32)},
        "step": 12,
    }
    buffer = io.BytesIO()
    torch.save(expected, buffer)
    path = Path("checkpoint.pt")
    storage = _BytesStorage({path: buffer.getvalue()})

    loaded = load_torch_serialized_from_storage(path, storage, map_location="cpu")

    torch.testing.assert_close(loaded["weights"]["a"], expected["weights"]["a"])
    assert loaded["step"] == expected["step"]
    assert len(storage.read_args) == 1
    assert storage.read_args[0].pre_read_full_file is False


class _NonSeekableStream(io.RawIOBase):
    """A read-only stream that refuses to seek, like a pure forward pipe."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    # pyre-ignore[14]
    def readinto(self, b: Any) -> int:
        return self._buf.readinto(b)

    def seek(self, *args: Any, **kwargs: Any) -> int:
        raise io.UnsupportedOperation("seek")


class _NonSeekableStorage(_BytesStorage):
    def stream_read(  # pyre-ignore[15]
        self, path: Path, read_args: Any | None = None
    ) -> io.RawIOBase:
        self.read_args.append(read_args)
        return _NonSeekableStream(self._data_by_path[path])


class _MisSeekingStream(io.RawIOBase):
    """A stream whose ``seek`` silently lands somewhere else.

    Models a backend that truncates or clamps large offsets -- the failure mode
    a byte-count check cannot see, because every read still returns full length.
    """

    def __init__(self, data: bytes, mask: int) -> None:
        self._buf = io.BytesIO(data)
        self._mask = mask

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    # pyre-ignore[14]
    def readinto(self, b: Any) -> int:
        return self._buf.readinto(b)

    def seek(self, offset: int, whence: int = 0) -> int:
        # Silently drop high bits, then honestly report where we landed.
        return self._buf.seek(offset & self._mask, whence)


class _MisSeekingStorage(_BytesStorage):
    def __init__(self, data_by_path: dict[Path, bytes], mask: int) -> None:
        super().__init__(data_by_path)
        self._mask = mask

    def stream_read(  # pyre-ignore[15]
        self, path: Path, read_args: Any | None = None
    ) -> io.RawIOBase:
        self.read_args.append(read_args)
        return _MisSeekingStream(self._data_by_path[path], self._mask)


def _big_checkpoint() -> tuple[dict[str, Any], bytes]:
    # Must exceed one fill chunk so the parallel path actually engages.
    expected = {"w": torch.arange(1_000_000, dtype=torch.float32), "step": 7}
    buffer = io.BytesIO()
    torch.save(expected, buffer)
    return expected, buffer.getvalue()


def test_parallel_fill_round_trips_identically_to_serial(monkeypatch: Any) -> None:
    expected, raw = _big_checkpoint()
    path = Path("checkpoint.pt")

    serial = load_torch_serialized_from_storage(
        path, _BytesStorage({path: raw}), map_location="cpu", num_workers=1
    )

    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_WORKERS", "4")
    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_CHUNK_MB", "1")
    storage = _BytesStorage({path: raw})
    parallel = load_torch_serialized_from_storage(path, storage, map_location="cpu")

    # Byte-identical payload, and it really did fan out into several ranged reads.
    torch.testing.assert_close(parallel["w"], expected["w"])
    torch.testing.assert_close(parallel["w"], serial["w"])
    assert parallel["step"] == expected["step"]
    assert len(storage.read_args) > 1, "parallel fill should open one stream per range"
    assert all(a.pre_read_full_file is False for a in storage.read_args)


def test_parallel_fill_falls_back_when_stream_cannot_seek(monkeypatch: Any) -> None:
    expected, raw = _big_checkpoint()
    path = Path("checkpoint.pt")

    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_WORKERS", "4")
    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_CHUNK_MB", "1")
    storage = _NonSeekableStorage({path: raw})

    loaded = load_torch_serialized_from_storage(path, storage, map_location="cpu")

    torch.testing.assert_close(loaded["w"], expected["w"])
    assert loaded["step"] == expected["step"]


def test_parallel_fill_rejects_a_stream_that_lands_on_the_wrong_offset(
    monkeypatch: Any,
) -> None:
    """A mis-positioned read must not be accepted just because it is full length.

    Note this only covers a backend that reports the wrong position; one that
    returns the requested offset while reading elsewhere cannot be caught here.
    """
    expected, raw = _big_checkpoint()
    path = Path("checkpoint.pt")

    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_WORKERS", "4")
    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_CHUNK_MB", "1")
    # Mask off bit 20, so any offset >= 1MiB lands short -- like a 32-bit
    # truncation would on a file larger than 4 GiB.
    storage = _MisSeekingStorage({path: raw}, mask=(1 << 20) - 1)

    loaded = load_torch_serialized_from_storage(path, storage, map_location="cpu")

    # The bad seek is detected and the whole fill is redone sequentially, so the
    # result is still correct rather than silently corrupt.
    torch.testing.assert_close(loaded["w"], expected["w"])
    assert loaded["step"] == expected["step"]


def test_num_workers_argument_overrides_the_env(monkeypatch: Any) -> None:
    """Callers that already read files concurrently pass 1 and stay sequential."""
    expected, raw = _big_checkpoint()
    path = Path("checkpoint.pt")

    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_WORKERS", "8")
    monkeypatch.setenv("TORCH_CKPT_PARALLEL_FILL_CHUNK_MB", "1")

    serial = _BytesStorage({path: raw})
    loaded = load_torch_serialized_from_storage(
        path, serial, map_location="cpu", num_workers=1
    )
    torch.testing.assert_close(loaded["w"], expected["w"])
    assert len(serial.read_args) == 1, "num_workers=1 must beat the env and stay serial"

    # None means "no opinion" -> the env still applies.
    env_driven = _BytesStorage({path: raw})
    load_torch_serialized_from_storage(
        path, env_driven, map_location="cpu", num_workers=None
    )
    assert len(env_driven.read_args) > 1, "num_workers=None should honor the env"
