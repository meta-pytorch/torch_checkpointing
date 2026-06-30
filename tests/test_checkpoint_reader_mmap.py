import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch_checkpointing.checkpoint_layout import TorchSerialization
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.types import RankInfo


class _BytesStorage:
    def __init__(self, data_by_path: dict[Path, bytes]) -> None:
        self._data_by_path = data_by_path
        self.read_args: list[Any] = []
        self.getsize_calls = 0

    def stream_read(self, path: Path, read_args: Any | None = None) -> io.BytesIO:
        self.read_args.append(read_args)
        return io.BytesIO(self._data_by_path[path])

    def getsize(self, path: Path) -> int:
        self.getsize_calls += 1
        return len(self._data_by_path[path])


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
