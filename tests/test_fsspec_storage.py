# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import uuid
from collections.abc import Iterator
from pathlib import Path

import fsspec
import numpy as np
import pytest
import torch
from fsspec.core import split_protocol
from fsspec.implementations.memory import MemoryFileSystem
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.config import AsyncCheckpointSaverConfig
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.base_storage import ReadArgs
from torch_checkpointing.storage.fsspec_storage import (
    FsspecStorage,
    FsspecStorageConfig,
)

# Point this at a real store (for example gs://bucket/prefix) to run the tests
# that take the ``storage`` fixture against its fsspec driver. Defaults to the
# in-memory filesystem.
_TEST_URL = os.environ.get(
    "TORCH_CHECKPOINTING_TEST_FSSPEC_URL", "memory://torch-checkpointing-tests"
)


@pytest.fixture
def storage() -> FsspecStorage:
    protocol, _ = split_protocol(_TEST_URL)
    return FsspecStorageConfig(protocol=protocol).create_storage()


@pytest.fixture
def root(storage: FsspecStorage) -> Iterator[Path]:
    _, base = split_protocol(_TEST_URL)
    path = Path(base) / uuid.uuid4().hex
    yield path
    if storage.exists(path):
        storage.rmdir(path)


def _put(storage: FsspecStorage, path: Path, data: bytes = b"payload") -> Path:
    storage.mkdir(path.parent)
    storage.write(path, data)
    return path


@pytest.mark.parametrize(
    "data",
    [
        b"Hello bytes! ",
        bytearray(b"Hello bytearray! "),
        memoryview(b"Hello memoryview! "),
        np.array([1, 2, 3, 4, 5], dtype=np.int32),
        torch.tensor([10, 20, 30], dtype=torch.int32).numpy(),
    ],
    ids=["bytes", "bytearray", "memoryview", "numpy", "torch"],
)
def test_write_then_read_round_trips_every_buffer_type(storage, root, data):
    storage.mkdir(root)
    storage.write(root / "a.bin", data)

    assert storage.read(root / "a.bin") == bytes(memoryview(data))


def test_stream_write_commits_on_close(storage, root):
    storage.mkdir(root)
    with storage.stream_write(root / "a.bin") as stream:
        stream.write(b"ab")
        stream.write(bytearray(b"cd"))

    assert storage.read(root / "a.bin") == b"abcd"


def test_ranged_read_supports_seek_and_readinto(storage, root):
    _put(storage, root / "a.bin", b"0123456789")

    read_args = ReadArgs(pre_read_full_file=False, direct_io=True)
    with storage.stream_read(root / "a.bin", read_args) as stream:
        stream.seek(3)
        buffer = bytearray(4)
        assert stream.readinto(buffer) == 4
        assert bytes(buffer) == b"3456"
        assert stream.tell() == 7


def test_read_of_a_missing_file_raises(storage, root):
    with pytest.raises(FileNotFoundError):
        storage.read(root / "missing.bin")


def test_ls_returns_entry_basenames(storage, root):
    _put(storage, root / "dir" / "a.bin")
    _put(storage, root / "dir" / "sub" / "b.bin")

    assert sorted(storage.ls(root / "dir")) == ["a.bin", "sub"]


def test_probes_distinguish_files_directories_and_absent_paths(storage, root):
    _put(storage, root / "dir" / "a.bin", b"0123456789")

    assert storage.exists(root / "dir" / "a.bin")
    assert storage.exists(root / "dir")
    assert storage.isdir(root / "dir")
    assert not storage.isdir(root / "dir" / "a.bin")
    assert storage.getsize(root / "dir" / "a.bin") == 10
    # A checkpoint folder that was never written is absent, not an error.
    assert not storage.exists(root / "never-written")
    assert not storage.isdir(root / "never-written")
    with pytest.raises(FileNotFoundError):
        storage.getsize(root / "dir" / "missing.bin")


def test_delete_removes_one_file(storage, root):
    _put(storage, root / "dir" / "a.bin")
    _put(storage, root / "dir" / "b.bin")

    storage.delete(root / "dir" / "a.bin")

    assert storage.ls(root / "dir") == ["b.bin"]


def test_rmdir_removes_a_populated_directory_and_nothing_else(storage, root):
    _put(storage, root / "step-1" / "a.bin")
    _put(storage, root / "step-1" / "sub" / "b.bin")
    # "step-1" is a string prefix of "step-10". A prefix-based delete must not
    # reach the sibling.
    _put(storage, root / "step-10" / "a.bin", b"keep")

    storage.rmdir(root / "step-1")

    assert not storage.exists(root / "step-1")
    assert storage.read(root / "step-10" / "a.bin") == b"keep"


def test_rmdir_of_a_missing_directory_raises(storage, root):
    with pytest.raises(FileNotFoundError):
        storage.rmdir(root / "missing")


def test_remap_path_returns_the_full_uri(storage, root):
    path = _put(storage, root / "a.bin")

    protocol, stripped = split_protocol(storage.remap_path(path))

    assert protocol is not None
    assert stripped.lstrip("/") == str(path).lstrip("/")


def test_config_survives_pickling():
    config = FsspecStorageConfig(protocol="memory")

    restored = pickle.loads(pickle.dumps(config))

    assert restored == config
    assert isinstance(restored.create_storage(), FsspecStorage)


def test_rename_directory_onto_absent_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    _put(storage, root / "src" / "sub" / "nested.bin")

    storage.rename(root / "src", root / "dst", is_directory=True)

    assert not storage.exists(root / "src")
    assert sorted(storage.ls(root / "dst")) == ["payload.bin", "sub"]
    assert storage.ls(root / "dst" / "sub") == ["nested.bin"]


def test_rename_directory_replaces_empty_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    storage.mkdir(root / "dst")

    storage.rename(root / "src", root / "dst", is_directory=True)

    assert not storage.exists(root / "src")
    # The contents land directly at dst, not nested under dst/src.
    assert storage.ls(root / "dst") == ["payload.bin"]


def test_rename_directory_rejects_populated_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    _put(storage, root / "dst" / "existing.bin")

    with pytest.raises(OSError):
        storage.rename(root / "src", root / "dst", is_directory=True)

    # Neither side is touched: no nesting, no merge, no deletion.
    assert storage.ls(root / "src") == ["payload.bin"]
    assert storage.ls(root / "dst") == ["existing.bin"]


def test_rename_directory_rejects_file_destination(storage, root):
    _put(storage, root / "src" / "payload.bin")
    _put(storage, root / "dst", b"not a directory")

    with pytest.raises(OSError):
        storage.rename(root / "src", root / "dst", is_directory=True)

    assert storage.read(root / "dst") == b"not a directory"
    assert storage.ls(root / "src") == ["payload.bin"]


def test_rename_directory_leaves_prefix_siblings_alone(storage, root):
    _put(storage, root / "tmp_step-2" / "payload.bin")
    # "step-2" is a string prefix of "step-20".
    _put(storage, root / "step-20" / "keep.bin", b"keep")

    storage.rename(root / "tmp_step-2", root / "step-2", is_directory=True)

    assert storage.ls(root / "step-2") == ["payload.bin"]
    assert storage.read(root / "step-20" / "keep.bin") == b"keep"
    assert sorted(storage.ls(root)) == ["step-2", "step-20"]


def test_rename_file_onto_absent_destination(storage, root):
    _put(storage, root / "a.bin", b"one")

    storage.rename(root / "a.bin", root / "b.bin")

    assert not storage.exists(root / "a.bin")
    assert storage.read(root / "b.bin") == b"one"


def test_rename_file_replaces_existing_file(storage, root):
    _put(storage, root / "a.bin", b"one")
    _put(storage, root / "b.bin", b"two")

    storage.rename(root / "a.bin", root / "b.bin")

    assert not storage.exists(root / "a.bin")
    assert storage.read(root / "b.bin") == b"one"


def test_rename_file_rejects_directory_destination(storage, root):
    _put(storage, root / "a.bin", b"one")
    _put(storage, root / "dst" / "existing.bin")

    with pytest.raises(OSError):
        storage.rename(root / "a.bin", root / "dst")

    assert storage.read(root / "a.bin") == b"one"
    assert storage.ls(root / "dst") == ["existing.bin"]


def _empty_directory(storage: FsspecStorage, path: Path) -> Path:
    storage.mkdir(path)
    if not storage.isdir(path):
        pytest.skip("the store keeps no empty directories")
    return path


def test_rename_onto_itself_leaves_the_directory_alone(storage, root):
    _put(storage, root / "full" / "a.bin")
    empty = _empty_directory(storage, root / "empty")

    storage.rename(root / "full", root / "full", is_directory=True)
    storage.rename(empty, empty, is_directory=True)

    assert storage.ls(root / "full") == ["a.bin"]
    assert storage.isdir(empty)


def test_rename_of_a_missing_source_raises_and_spares_the_destination(storage, root):
    empty = _empty_directory(storage, root / "empty")

    with pytest.raises(FileNotFoundError):
        storage.rename(root / "missing", empty, is_directory=True)
    with pytest.raises(FileNotFoundError):
        storage.rename(root / "missing", root / "missing")

    assert storage.isdir(empty)


def test_rename_file_flagged_as_directory_rejects_empty_directory(storage, root):
    # The source's type decides, not the is_directory hint.
    _put(storage, root / "a.bin", b"one")
    empty = _empty_directory(storage, root / "empty")

    with pytest.raises(OSError):
        storage.rename(root / "a.bin", empty, is_directory=True)

    assert storage.read(root / "a.bin") == b"one"
    assert storage.isdir(empty)


def test_directory_operations_reject_a_file(storage, root):
    path = _put(storage, root / "a.bin", b"one")

    with pytest.raises(NotADirectoryError):
        storage.ls(path)
    with pytest.raises(NotADirectoryError):
        storage.rmdir(path)

    assert storage.read(path) == b"one"


def test_ranged_reads_have_independent_positions(storage, root):
    # Parallel loads give each worker its own stream of the same file.
    path = _put(storage, root / "a.bin", b"0123456789")
    read_args = ReadArgs(pre_read_full_file=False)

    with storage.stream_read(path, read_args) as first:
        first.seek(6)
        with storage.stream_read(path, read_args) as second:
            assert second.read(2) == b"01"
        assert first.read(2) == b"67"


class _ListingCachingMemoryFileSystem(MemoryFileSystem):
    """In-memory filesystem that caches directory listings the way gcsfs does.

    Listings are cached per instance, so changes made through another
    instance are not seen. A cached listing of the parent or grandparent is
    trusted when looking a path up, so an entry missing from it reads as
    absent, and opening a file for reading looks it up first. The plain memory
    filesystem never caches, so it cannot go stale.
    """

    protocol = "listingcachingmemory"

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        try:
            entries = self.dircache[path]
        except KeyError:
            entries = super().ls(path, detail=True, **kwargs)
            self.dircache[path] = entries
        return entries if detail else sorted(entry["name"] for entry in entries)

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        # Each raises FileNotFoundError when a cached listing lacks the entry.
        self._ls_from_cache(self._parent(path))
        self._ls_from_cache(path)
        return super().info(path, **kwargs)

    def _open(self, path, mode="rb", **kwargs):
        if "r" in mode:
            self.info(path)
        return super()._open(path, mode, **kwargs)

    def invalidate_cache(self, path=None):
        # Drops the listing of path and of every ancestor.
        if path is None:
            self.dircache.clear()
            return
        path = self._strip_protocol(path).rstrip("/")
        while path:
            self.dircache.pop(path, None)
            parent = self._parent(path).rstrip("/")
            if parent == path:
                break
            path = parent


@pytest.mark.parametrize(
    "storage_options",
    [{}, {"use_listings_cache": True}],
    ids=["default", "caching_requested"],
)
def test_sees_a_checkpoint_written_through_another_client(storage_options):
    # A long-lived reader lists the folder; another rank or the async-save
    # subprocess then writes a checkpoint through its own client, and the
    # reader loads it with ranged reads.
    fsspec.register_implementation(
        "listingcachingmemory", _ListingCachingMemoryFileSystem, clobber=True
    )
    storage = FsspecStorageConfig(
        protocol="listingcachingmemory", storage_options=storage_options
    ).create_storage()
    root = Path(f"test-{uuid.uuid4().hex}")
    other_client = fsspec.filesystem("memory")
    try:
        storage.write(root / "step-5" / "metadata.pkl", b"m")
        assert storage.ls(root) == ["step-5"]

        other_client.pipe_file(f"/{root}/step-10/model_0.pt", b"0123456789")

        read_args = ReadArgs(pre_read_full_file=False)
        with storage.stream_read(root / "step-10" / "model_0.pt", read_args) as stream:
            stream.seek(3)
            assert stream.read(4) == b"3456"
        assert storage.getsize(root / "step-10" / "model_0.pt") == 10
        assert sorted(storage.ls(root)) == ["step-10", "step-5"]
    finally:
        _ListingCachingMemoryFileSystem.clear_instance_cache()
        if other_client.exists(f"/{root}"):
            other_client.rm(f"/{root}", recursive=True)


def _payload() -> dict:
    return {"model": torch.arange(4), "step": 10}


def test_sync_save_then_load_round_trip_over_fsspec(tmp_path: Path) -> None:
    config = CheckpointManager.Config.with_sync_save()
    config.storage_config = FsspecStorageConfig(protocol="file")
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "sync_round_trip")
    try:
        assert manager.save(checkpoint_path, _payload()) is None

        loaded = manager.load(
            checkpoint_path,
            into={"model": torch.zeros(4, dtype=torch.long), "step": 0},
        )

        assert torch.equal(loaded["model"], torch.arange(4))
        assert loaded["step"] == 10
    finally:
        manager.close()


def test_async_save_then_load_round_trip_over_fsspec(tmp_path: Path) -> None:
    # The memory filesystem is per process, so the save subprocess needs a
    # store both processes can see.
    config = CheckpointManager.Config(
        save=AsyncCheckpointSaverConfig(
            staging_config=CheckpointStagerConfig(
                use_pinned_memory=False,
                use_non_blocking_copy=False,
            )
        )
    )
    config.save.wait_timeout_secs = 30
    config.storage_config = FsspecStorageConfig(protocol="file")
    manager = config.build()

    checkpoint_path = os.path.join(tmp_path, "async_round_trip")
    try:
        write_future = manager.save(checkpoint_path, _payload())
        assert write_future is not None
        write_future.result(timeout=30)

        loaded = manager.load(
            checkpoint_path,
            into={"model": torch.zeros(4, dtype=torch.long), "step": 0},
        )

        assert torch.equal(loaded["model"], torch.arange(4))
        assert loaded["step"] == 10
    finally:
        manager.close()
