import io
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from typing_extensions import Buffer, override

from .base_storage import ReadArgs, Storage, StorageConfig

logger = logging.getLogger(__name__)


@dataclass
class LocalFileSystemStorageConfig(StorageConfig):
    # Put backend specific configs here.
    use_direct_io: bool = True

    def create_storage(self) -> "LocalFileSystemStorage":
        return LocalFileSystemStorage(config=self)


class WriteStream(io.IOBase):
    """
    File-like write stream with directio support.
    Automatically tries O_DIRECT and falls back to regular I/O if needed.
    """

    def __init__(self, path: str | Path, mode: str = "wb", use_direct_io: bool = True):
        self._path = path
        self._mode = mode
        self._use_direct_io = use_direct_io
        self._fd: int | None = None
        self._file: IO | None = None
        self._using_direct_io = False

    def __enter__(self):
        # Check if parent directory exists and raise an error if it doesn't.
        parent_dir = os.path.dirname(self._path)
        if parent_dir and not os.path.exists(parent_dir):
            raise RuntimeError(
                f"Parent directory does not exist, create it!: {parent_dir}"
            )

        if self._use_direct_io:
            # Try directio first
            f_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
            try:
                self._fd = os.open(self._path, f_mode | os.O_DIRECT)
                self._file = os.fdopen(self._fd, self._mode)
                self._fd = None  # fdopen takes ownership
                self._using_direct_io = True
                logger.debug(f"Opened {self._path} with O_DIRECT")
            except OSError as e:
                if (
                    e.errno == 22
                ):  # Invalid argument - filesystem doesn't support O_DIRECT
                    logger.warning(
                        f"File system does not support O_DIRECT, falling back to default mode for {self._path}"
                    )
                    # Fall through to regular open
                    if self._fd is not None:
                        try:
                            os.close(self._fd)
                        except OSError:
                            pass
                        self._fd = None
                else:
                    raise

        # Use regular I/O (either by choice or as fallback)
        if self._file is None:
            self._file = open(self._path, self._mode)
            self._using_direct_io = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, data: Buffer) -> int:
        if self._file is None:
            raise ValueError("File is not open")
        if isinstance(data, (bytes, bytearray)):
            return self._file.write(data)
        else:
            mv = memoryview(data)
            return self._file.write(mv)

    def flush(self):
        if self._file is not None:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    @property
    def using_direct_io(self) -> bool:
        """Returns True if this stream is using direct I/O."""
        return self._using_direct_io


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

    def _open_for_read(self, path: Path, use_direct_io: bool) -> io.RawIOBase:
        """Open file for reading, with optional O_DIRECT support."""
        if use_direct_io:
            try:
                fd = os.open(path, os.O_DIRECT | os.O_RDONLY)
                return os.fdopen(fd, "rb", buffering=0)
            except OSError as e:
                if e.errno == 22:  # Filesystem doesn't support O_DIRECT
                    logger.warning(
                        f"File system does not support O_DIRECT, falling back to default mode for {path}"
                    )
                else:
                    raise
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
        return WriteStream(path, "wb", self._use_direct_io)

    @override
    def write(self, path: Path, data: Buffer) -> None:
        with WriteStream(path, "wb", self._use_direct_io) as f:
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
        is_cross_link: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        if is_cross_link and os.path.isdir(src_path):
            # Cross-filesystem move: copy contents then delete source
            # This avoids shutil.move's nesting behavior when dst exists
            os.makedirs(dst_path, exist_ok=True)
            shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=True)
            shutil.rmtree(str(src_path))
        else:
            shutil.move(str(src_path), str(dst_path))

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
