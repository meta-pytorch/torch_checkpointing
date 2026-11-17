# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import errno
import io
import os
import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from typing_extensions import Buffer, override

from .base_storage import ReadArgs, Storage, StorageConfig
from .filesystem import _occupied_destination_error


@dataclass
class FsspecStorageConfig(StorageConfig):
    """Storage on a filesystem reached through an fsspec driver.

    Args:
        protocol: The fsspec protocol, e.g. ``"gs"`` or ``"s3"``. Its driver
            (``gcsfs``, ``s3fs``, ...) must be installed.
        storage_options: Keyword arguments for the driver's filesystem class.

    Paths handed to the storage carry no scheme: ``bucket/prefix/step-10``
    rather than ``gs://bucket/prefix/step-10``. The checkpoint stack moves
    paths around as ``pathlib.Path``, which would collapse the double slash of
    a URI, so the protocol lives here and ``remap_path`` restores the URI.

    ``use_listings_cache=False`` is always passed to the driver, overriding
    ``storage_options``, to keep fsspec's directory-listing cache empty: other
    ranks and the async-save subprocess write through their own clients, so a
    client's cached listing goes stale, and a driver that trusts it (gcsfs
    does) fails to find paths written since.
    """

    protocol: str
    storage_options: dict[str, Any] = field(default_factory=dict)

    @override
    def create_storage(self) -> "FsspecStorage":
        return FsspecStorage(config=self)


class FsspecStorage(Storage):
    """fsspec implementation of the Storage ABC."""

    def __init__(self, config: FsspecStorageConfig) -> None:
        # Imported here so this module imports without fsspec installed. The
        # client is built here rather than held by the config, so every process
        # that receives the pickled config builds its own.
        import fsspec

        self._fs = fsspec.filesystem(
            config.protocol,
            **{**config.storage_options, "use_listings_cache": False},
        )

    @override
    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> io.RawIOBase:
        # direct_io and timeout_us are ignored.
        if read_args is None or read_args.pre_read_full_file:
            return cast(io.RawIOBase, io.BytesIO(self._fs.cat_file(str(path))))
        stream = self._fs.open(str(path), "rb")
        if isinstance(stream, io.BytesIO):
            # The memory filesystem hands every opener the one stored file, so
            # concurrent ranged readers would move each other's cursor.
            return cast(io.RawIOBase, io.BytesIO(stream.getvalue()))
        return cast(io.RawIOBase, stream)

    @override
    def stream_write(self, path: Path) -> io.IOBase:
        # Drivers may buffer and upload in parts, so the file is only complete
        # once the stream is closed.
        return self._fs.open(str(path), "wb")

    @override
    def write(self, path: Path, data: Buffer) -> None:
        with self._fs.open(str(path), "wb") as stream:
            # A flat byte view, because drivers may truth-test or len() what
            # they are given: gcsfs' zonal-bucket writer rejects a numpy array,
            # and len() of a typed buffer counts elements, not bytes.
            stream.write(memoryview(data).cast("B"))

    @override
    def delete(self, path: Path) -> None:
        self._fs.rm_file(str(path))

    @override
    def mkdir(self, path: Path, recursive: bool = True) -> None:
        # Left to the driver; typically a no-op on stores that have no
        # directories.
        if recursive:
            self._fs.makedirs(str(path), exist_ok=True)
        else:
            self._fs.mkdir(str(path), create_parents=False)

    @override
    def rmdir(self, path: Path) -> None:
        # A recursive rm deletes a file too; rmdir must not.
        if not self.isdir(path):
            if self.exists(path):
                raise NotADirectoryError(
                    errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(path)
                )
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        self._fs.rm(str(path), recursive=True)

    @override
    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        # background_cleanup is ignored: it only takes cleanup off the caller's
        # path, and here the source is removed before this returns.
        #
        # Look the source up before touching the destination: an empty
        # destination directory is removed below, which must not happen for a
        # source that is missing or is the destination itself. Its type, not
        # is_directory, picks the branch, so a file never replaces a directory.
        src_type = self._fs.info(str(src_path))["type"]
        if self._fs._strip_protocol(str(src_path)).rstrip("/") == (
            self._fs._strip_protocol(str(dst_path)).rstrip("/")
        ):
            return  # As POSIX rename: a path renamed onto itself is unchanged.
        if src_type == "directory":
            self._rename_directory(src_path, dst_path)
        else:
            self._rename_file(src_path, dst_path)

    def _rename_directory(self, src_path: Path, dst_path: Path) -> None:
        """Move a directory so its contents land directly at ``dst_path``.

        The destination check and the driver's move are separate calls. Where
        the move is a single directory rename, that step is atomic. On a flat
        object store each object is copied and then deleted, so a failure
        midway can leave ``dst_path`` partly written.
        """
        src, dst = str(src_path), str(dst_path)
        if self.exists(dst_path):
            if not self.isdir(dst_path):
                raise _occupied_destination_error(src_path, dst_path, errno.ENOTDIR)
            if self.ls(dst_path):
                raise _occupied_destination_error(src_path, dst_path, errno.ENOTEMPTY)
            # An empty destination is unoccupied. Remove it first: fsspec moves
            # a directory *into* an existing destination directory, which would
            # nest the contents under dst_path/<basename of src_path>.
            self._fs.rm(dst, recursive=True)
        self._fs.mv(src, dst, recursive=True)

    def _rename_file(self, src_path: Path, dst_path: Path) -> None:
        """Move a file onto ``dst_path``, replacing an existing file."""
        if self.isdir(dst_path):
            raise _occupied_destination_error(src_path, dst_path, errno.EISDIR)
        self._fs.mv(str(src_path), str(dst_path))

    @override
    def ls(self, path: Path) -> list[str]:
        target = str(path)
        # Object stores may list a marker entry for the directory itself, which
        # a directory listing must not contain. A file lists as only itself.
        self_entry = self._fs._strip_protocol(target).rstrip("/")
        entries = self._fs.ls(target, detail=True)
        names = [entry["name"].rstrip("/") for entry in entries]
        if names == [self_entry] and entries[0]["type"] == "file":
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), target)
        return [posixpath.basename(name) for name in names if name != self_entry]

    @override
    def exists(self, path: Path) -> bool:
        return self._fs.exists(str(path))

    @override
    def getsize(self, path: Path) -> int:
        return self._fs.size(str(path))

    @override
    def isdir(self, path: Path) -> bool:
        return self._fs.isdir(str(path))

    @override
    def remap_path(self, path: Path) -> str:
        return self._fs.unstrip_protocol(str(path))
