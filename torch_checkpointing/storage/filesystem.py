# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import errno
import io
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Buffer, override

from .base_storage import (
    _occupied_destination_error,
    ReadArgs,
    Storage,
    StorageConfig,
)

try:
    import fcntl
except ModuleNotFoundError:
    fcntl = None

logger = logging.getLogger(__name__)

# O_DIRECT is Linux-only. Platforms without it (macOS, Windows) take the
# buffered I/O path, the same fallback used when a filesystem refuses O_DIRECT.
# It is only used where fcntl can clear it again.
_O_DIRECT: int | None = getattr(os, "O_DIRECT", None) if fcntl is not None else None


@dataclass
class LocalFileSystemStorageConfig(StorageConfig):
    # Put backend specific configs here.
    use_direct_io: bool = True

    def create_storage(self) -> "LocalFileSystemStorage":
        return LocalFileSystemStorage(config=self)


def _buffer_size(fd: int) -> int:
    """The buffer size ``open()`` would pick for ``fd``."""
    block_size = os.fstat(fd).st_blksize
    return block_size if block_size > 1 else io.DEFAULT_BUFFER_SIZE


def _clear_o_direct(fd: int) -> None:
    """Switch ``fd`` to buffered I/O at its current offset.

    Clears O_DIRECT on the open file description, so the offset carries over
    and the file is not reopened by path.
    """
    assert fcntl is not None, "O_DIRECT is only used where fcntl is available"
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_DIRECT)


class _DirectIOWriter(io.FileIO):
    """Raw writer for a file opened with O_DIRECT, with a buffered fallback.

    A serialized byte stream cannot keep O_DIRECT's alignment rules: it writes
    small headers and payloads at arbitrary offsets, from buffers that are not
    block aligned. FUSE mounts and btrfs accept such writes (btrfs by routing
    them through the page cache), while ext4 and XFS accept the O_DIRECT open
    but reject the writes with EINVAL. On the first rejected write this clears
    O_DIRECT on the descriptor and retries the write. A failed write(2) leaves
    the file offset where it was, so the retry continues at the right place.
    The refusal is reported through ``on_unsupported``, so the owner can stop
    asking for O_DIRECT on later files.

    The fallback lives in the raw file because the ``io.BufferedWriter`` that
    wraps it issues the actual writes, including from ``flush()`` and
    ``close()``, so only here are a failed write's bytes and offset known.
    """

    def __init__(
        self, fd: int, path: Path, on_unsupported: Callable[[Path, str], None]
    ) -> None:
        super().__init__(fd, "w", closefd=True)
        self._path = path
        self._on_unsupported = on_unsupported
        self._direct_io = True

    @override
    def write(self, b: Buffer, /) -> int:
        if self._direct_io:
            try:
                return os.write(self.fileno(), b)
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
            _clear_o_direct(self.fileno())
            self._direct_io = False
            self._on_unsupported(self._path, "write")
        return os.write(self.fileno(), b)


class _DirectIOReader(io.RawIOBase):
    """Read stream for a file opened with O_DIRECT, with a buffered fallback.

    Offset reads target arbitrary tensor slices, so their reads are unaligned,
    and filesystems differ in whether they allow that. FUSE mounts and btrfs
    accept O_DIRECT reads at any offset, length and buffer address, while XFS
    and ext4 accept the O_DIRECT open but reject unaligned reads with EINVAL. On
    the first rejected read this clears O_DIRECT on the descriptor and retries
    the read. A failed read(2) leaves the file offset where it was, so the
    retry continues at the right place. The refusal is reported through
    ``on_unsupported``, so the owner can stop asking for O_DIRECT on later
    files.

    This wraps the file rather than subclassing ``io.FileIO``, because
    ``FileIO.read()`` and ``FileIO.readall()`` do not go through ``readinto``;
    here every read does.
    """

    def __init__(
        self, fd: int, path: Path, on_unsupported: Callable[[Path, str], None]
    ) -> None:
        self._file: io.RawIOBase = os.fdopen(fd, "rb", buffering=0)
        self._path = path
        self._on_unsupported = on_unsupported
        self._direct_io = True

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        return self._file.tell()

    def readinto(self, buffer: Buffer) -> int | None:
        if self._direct_io:
            try:
                return self._file.readinto(buffer)
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
            _clear_o_direct(self._file.fileno())
            self._direct_io = False
            self._on_unsupported(self._path, "read")
        return self._file.readinto(buffer)

    def close(self) -> None:
        if not self.closed:
            self._file.close()
        super().close()


class LocalFileSystemStorage(Storage):
    """
    Local file system implementation of the Storage ABC.
    """

    def __init__(self, config: LocalFileSystemStorageConfig):
        """
        Initialize LocalFileSystemStorage.
        """
        if config is None:
            raise ValueError("config must be provided")
        self._use_direct_io = config.use_direct_io
        # Cleared once the filesystem refuses an O_DIRECT open, read or write,
        # so later files are read and written with buffered I/O without trying
        # O_DIRECT again. One flag covers both directions: filesystems apply the
        # same alignment rules to O_DIRECT reads and writes.
        self._try_direct_io = _O_DIRECT is not None

    def _disable_direct_io(self, path: Path, operation: str) -> None:
        if self._try_direct_io:
            self._try_direct_io = False
            logger.warning(
                f"O_DIRECT {operation} failed for {path}; using buffered I/O for "
                "it and for later files"
            )

    def _open_for_write(self, path: Path) -> io.BufferedWriter:
        """Open file for writing, with O_DIRECT where the filesystem allows it."""
        if self._use_direct_io and self._try_direct_io:
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT)
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
                self._disable_direct_io(path, "open")
            else:
                return io.BufferedWriter(
                    _DirectIOWriter(fd, path, self._disable_direct_io),
                    buffer_size=_buffer_size(fd),
                )
        return open(path, "wb")

    def _open_for_read(self, path: Path, use_direct_io: bool) -> io.RawIOBase:
        """Open file for reading, with O_DIRECT if requested and the filesystem
        allows it."""
        if use_direct_io and self._try_direct_io:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
                self._disable_direct_io(path, "open")
            else:
                return _DirectIOReader(fd, path, self._disable_direct_io)
        return open(path, "rb", buffering=0)

    def _read_full_file(self, path: Path, use_direct_io: bool) -> io.BytesIO:
        """Read entire file into memory with O_DIRECT fallback on read failure."""
        try:
            with self._open_for_read(path, use_direct_io) as f:
                return io.BytesIO(f.read())
        except OSError as e:
            if e.errno == 22 and use_direct_io:
                # O_DIRECT read failed (e.g. in containers with volume mounts)
                logger.warning(
                    f"O_DIRECT read failed for {path}, retrying with buffered I/O"
                )
                with open(path, "rb") as f:
                    return io.BytesIO(f.read())
            raise

    @override
    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> io.RawIOBase:
        """Return a file-like object for reading with optional direct I/O."""
        use_direct_io = read_args.direct_io if read_args else False
        pre_read_full_file = read_args.pre_read_full_file if read_args else True

        if pre_read_full_file:
            return self._read_full_file(path, use_direct_io)

        return self._open_for_read(path, use_direct_io)

    @override
    def stream_write(self, path: Path) -> io.IOBase:
        """Return a file-like object for writing. Object must have a write() method."""
        return self._open_for_write(path)

    @override
    def write(self, path: Path, data: Buffer) -> None:
        with self._open_for_write(path) as f:
            f.write(data)

    @override
    def mkdir(self, path: Path, recursive: bool = True) -> None:
        if recursive:
            os.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path)

    @override
    def rmdir(self, path: Path) -> None:
        """
        Delete a checkpoint directory.

        First attempts os.rmdir(), which some backends can satisfy cheaply,
        and falls back to shutil.rmtree() if that fails.
        Args:
            path: Path to checkpoint directory to delete

        Raises:
            Exception: If both rmdir and rmtree fail
        """
        try:
            # Try the cheap single-call delete first
            os.rmdir(path)
            logger.debug(f"Deleted checkpoint directory using rmdir: {path}")
        except OSError:
            # Fall back to standard recursive delete
            try:
                shutil.rmtree(path, ignore_errors=False)
                logger.debug(f"Deleted checkpoint directory using rmtree: {path}")
            except Exception as e:
                logger.error(f"Both rmdir and rmtree failed for {path}: {e}")
                raise

    @override
    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        if os.path.isdir(src_path):
            self._rename_directory(src_path, dst_path)
        else:
            self._rename_file(src_path, dst_path)

    def _rename_directory(self, src_path: Path, dst_path: Path) -> None:
        """Move a directory so its contents land directly at ``dst_path``.

        ``os.rename`` gives exactly the contract documented on
        ``Storage.rename``: it replaces an absent or empty destination
        atomically and refuses a populated one. ``shutil.move`` must not be
        used here, because it moves the source *inside* an existing destination
        directory rather than replacing it.
        """
        try:
            os.rename(src_path, dst_path)
            return
        except OSError as e:
            if e.errno in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOTDIR):
                raise _occupied_destination_error(src_path, dst_path, e.errno) from e
            if e.errno != errno.EXDEV:
                raise
            logger.warning(
                f"Cannot rename {src_path} to {dst_path} across filesystems, "
                "falling back to a non-atomic copy"
            )

        # Cross-filesystem move: copy the contents over, then drop the source.
        # The destination is vetted up front because copytree would merge into
        # it, which loses the "commit onto unoccupied space" guarantee.
        if os.path.exists(dst_path) and (
            not os.path.isdir(dst_path) or os.listdir(dst_path)
        ):
            raise _occupied_destination_error(
                src_path,
                dst_path,
                errno.ENOTEMPTY if os.path.isdir(dst_path) else errno.ENOTDIR,
            )
        os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=True)
        shutil.rmtree(str(src_path))

    def _rename_file(self, src_path: Path, dst_path: Path) -> None:
        """Move a file so it lands at ``dst_path`` itself.

        ``os.replace`` overwrites an existing file, which the atomic-write
        callers depend on, and refuses a directory. ``shutil.move`` must not be
        used here either: given a directory it would drop the file inside.
        """
        try:
            os.replace(src_path, dst_path)
            return
        except OSError as e:
            if e.errno in (errno.EISDIR, errno.ENOTEMPTY, errno.EEXIST):
                raise _occupied_destination_error(src_path, dst_path, e.errno) from e
            if e.errno != errno.EXDEV:
                raise

        if os.path.isdir(dst_path):
            raise _occupied_destination_error(src_path, dst_path, errno.EISDIR)
        shutil.copy2(str(src_path), str(dst_path))
        os.remove(src_path)

    @override
    def ls(self, path: Path) -> list[str]:
        return os.listdir(path)

    @override
    def delete(self, path: Path) -> None:
        os.remove(path)

    @override
    def exists(self, path: Path) -> bool:
        """Check if a file or directory exists."""
        return os.path.exists(path)

    @override
    def getsize(self, path: Path) -> int:
        """Get the size of a file in bytes."""
        return os.path.getsize(path)

    @override
    def isdir(self, path: Path) -> bool:
        """Check if path is a directory."""
        return os.path.isdir(path)

    @override
    def remap_path(self, path: Path) -> str:
        return str(path)
