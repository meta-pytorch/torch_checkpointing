# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import mmap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch_checkpointing.checkpoint_layout import TorchSerialization
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.storage.base_storage import MmapFill, Storage
from torch_checkpointing.types import RankInfo


class _BytesStorage(Storage):
    def __init__(self, data_by_path: dict[Path, bytes]) -> None:
        self._data_by_path = data_by_path
        self.read_args: list[Any] = []
        self.getsize_calls = 0

    def stream_read(self, path: Path, read_args: Any | None = None) -> Any:
        self.read_args.append(read_args)
        return io.BytesIO(self._data_by_path[path])

    def getsize(self, path: Path) -> int:
        self.getsize_calls += 1
        return len(self._data_by_path[path])

    def stream_write(self, path: Path) -> io.IOBase:
        raise NotImplementedError

    def write(self, path: Path, data: Any) -> None:
        raise NotImplementedError

    def delete(self, path: Path) -> None:
        raise NotImplementedError

    def mkdir(self, path: Path, recursive: bool = True) -> None:
        raise NotImplementedError

    def rmdir(self, path: Path) -> None:
        raise NotImplementedError

    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        raise NotImplementedError

    def ls(self, path: Path) -> list[str]:
        raise NotImplementedError

    def exists(self, path: Path) -> bool:
        raise NotImplementedError

    def isdir(self, path: Path) -> bool:
        raise NotImplementedError

    def remap_path(self, path: Path) -> str:
        raise NotImplementedError


class _StorageConfig:
    def __init__(self, storage: _BytesStorage) -> None:
        self._storage = storage

    def create_storage(self) -> _BytesStorage:
        return self._storage


def _rank_info() -> RankInfo:
    return RankInfo(
        global_rank=0,
        global_world_size=1,
        role_rank=0,
        role_world_size=1,
    )


def _torch_payload() -> bytes:
    buffer = io.BytesIO()
    torch.save({"weights": torch.arange(4, dtype=torch.float32)}, buffer)
    return buffer.getvalue()


def _load_with_reader(
    *, disable_use_mmap_backed_storage_on_load: bool
) -> tuple[Any, _BytesStorage]:
    path = Path("checkpoint.pt")
    storage = _BytesStorage({path: _torch_payload()})
    reader = CheckpointReader(
        rank_info=_rank_info(),
        storage_config=_StorageConfig(storage),
        disable_use_mmap_backed_storage_on_load=disable_use_mmap_backed_storage_on_load,
    )
    loaded = reader._load_full_file(
        path,
        SimpleNamespace(serialization_format=TorchSerialization()),
        map_location="cpu",
    )
    return loaded, storage


def test_checkpoint_reader_uses_stream_load_when_disabled() -> None:
    loaded, storage = _load_with_reader(disable_use_mmap_backed_storage_on_load=True)

    torch.testing.assert_close(loaded["weights"], torch.arange(4, dtype=torch.float32))
    assert storage.getsize_calls == 0
    assert storage.read_args == [None]


def test_checkpoint_reader_uses_mmap_backed_storage_by_default() -> None:
    loaded, storage = _load_with_reader(disable_use_mmap_backed_storage_on_load=False)

    torch.testing.assert_close(loaded["weights"], torch.arange(4, dtype=torch.float32))
    assert storage.getsize_calls == 1
    assert len(storage.read_args) == 1
    assert storage.read_args[0].pre_read_full_file is False


def test_checkpoint_reader_uses_storage_specific_mmap_fill() -> None:
    path = Path("checkpoint.pt")
    payload = _torch_payload()
    storage = _BytesStorage({path: payload})
    factory_calls: list[Any] = []
    fill_calls: list[tuple[int, Path, int, int]] = []

    def mmap_fill_factory(candidate: Any):
        factory_calls.append(candidate)

        def mmap_fill(
            file_size: int,
            callback_path: Path,
            workers: int,
            chunk_bytes: int,
        ) -> mmap.mmap:
            fill_calls.append((file_size, callback_path, workers, chunk_bytes))
            mm_file = mmap.mmap(-1, file_size)
            mm_file[:] = payload
            return mm_file

        return mmap_fill

    reader = CheckpointReader(
        rank_info=_rank_info(),
        storage_config=_StorageConfig(storage),
        mmap_fill_factory=mmap_fill_factory,
    )
    loaded = reader._load_full_file(
        path,
        SimpleNamespace(serialization_format=TorchSerialization()),
        map_location="cpu",
    )

    torch.testing.assert_close(loaded["weights"], torch.arange(4, dtype=torch.float32))
    assert factory_calls == [storage]
    assert len(fill_calls) == 1
    assert fill_calls[0][0] == len(payload)
    assert fill_calls[0][1] == path
    assert storage.read_args == []


def test_checkpoint_reader_does_not_resolve_mmap_fill_when_disabled() -> None:
    path = Path("checkpoint.pt")
    storage = _BytesStorage({path: _torch_payload()})
    factory_calls: list[Any] = []
    reader = CheckpointReader(
        rank_info=_rank_info(),
        storage_config=_StorageConfig(storage),
        disable_use_mmap_backed_storage_on_load=True,
        mmap_fill_factory=lambda candidate: factory_calls.append(candidate),
    )

    reader._load_full_file(
        path,
        SimpleNamespace(serialization_format=TorchSerialization()),
        map_location="cpu",
    )

    assert factory_calls == []


def test_storage_fill_defaults_to_none() -> None:
    assert _BytesStorage({}).mmap_fill is None


@pytest.mark.parametrize(
    "selection", ["default", "capability", "explicit_source", "disabled"]
)
def test_storage_fill_capability_requires_explicit_reader_factory(
    selection: str,
) -> None:
    path = Path("checkpoint.pt")
    payload = _torch_payload()
    calls: list[tuple[int, Path, int, int]] = []

    class StorageWithFill(_BytesStorage):
        @property
        def mmap_fill(self) -> MmapFill:
            def fill(
                file_size: int, callback_path: Path, workers: int, chunk_bytes: int
            ) -> mmap.mmap:
                calls.append((file_size, callback_path, workers, chunk_bytes))
                result = mmap.mmap(-1, file_size)
                result[:] = payload
                return result

            return fill

    storage = StorageWithFill({path: payload})
    reader = CheckpointReader(
        rank_info=_rank_info(),
        storage_config=_StorageConfig(storage),
        disable_use_mmap_backed_storage_on_load=selection == "disabled",
        mmap_fill_factory=(
            (lambda candidate: candidate.mmap_fill)
            if selection in ("capability", "disabled")
            else (lambda candidate: None)
            if selection == "explicit_source"
            else None
        ),
    )
    loaded = reader._load_full_file(
        path,
        SimpleNamespace(serialization_format=TorchSerialization()),
        map_location="cpu",
    )
    torch.testing.assert_close(loaded["weights"], torch.arange(4, dtype=torch.float32))
    assert len(calls) == int(selection == "capability")
    assert len(storage.read_args) == int(selection != "capability")
