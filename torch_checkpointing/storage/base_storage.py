# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Buffer


@dataclass
class StorageConfig(ABC):
    """
    Abstract base class for configuring storage backends. Each backend must implement a config class inheriting from this.
    """

    @abstractmethod
    def create_storage(self) -> "Storage":
        """Create a storage instance from the config."""
        pass


@dataclass
class ReadArgs:
    """
    Arguments for reading from storage.
    Args:
        pre_read_full_file: Whether to pre-read the entire file before initializing the stream.
        direct_io: Whether to use direct I/O for reading.
        timeout_us: Timeout in microseconds for the read operation. Currently
            only honored by streaming storage backends that support a
            client-side read timeout.
            TODO: Remove this client-side timeout once server-side timeouts are
            reliable and configurable. This exists as a stopgap because the
            default server-side timeout can be too short for large-scale
            checkpoint reads under storage contention.
    """

    pre_read_full_file: bool = True
    direct_io: bool = False
    timeout_us: int = 900_000_000


class Storage(ABC):
    """
    Abstract base class for storage backends.
    """

    # Returns RawIOBase to have readinto support which is crucial for performance
    @abstractmethod
    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> io.RawIOBase:
        """Stream data from the given file path in chunks."""
        pass

    @abstractmethod
    def stream_write(self, path: Path) -> io.IOBase:
        """Return a file-like object for writing."""
        pass

    def read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> bytes:
        """Read the entire contents of a file.

        Args:
            path: Path to the file to read.

        Returns:
            The file contents as bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        with self.stream_read(path, read_args) as f:
            return f.read()

    @abstractmethod
    def write(self, path: Path, data: Buffer) -> None:
        """Write the entire buffer to a file."""
        pass

    @abstractmethod
    def delete(self, path: Path) -> None:
        """Delete a file."""
        pass

    @abstractmethod
    def mkdir(self, path: Path, recursive: bool = True) -> None:
        """Create a directory."""
        pass

    @abstractmethod
    def rmdir(self, path: Path) -> None:
        """Remove a directory."""
        pass

    @abstractmethod
    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        is_cross_link: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        """Rename a file or directory.

        Args:
            src_path: Source path
            dst_path: Destination path
            is_directory: If True, rename all objects under src prefix
            background_cleanup: If True, delete source objects in a background
                thread after copying completes, so the caller is not blocked
                on the low-priority cleanup. Only applies to directory renames.
        """
        pass

    @abstractmethod
    def ls(self, path: Path) -> list[str]:
        """List contents of a directory."""
        pass

    @abstractmethod
    def exists(self, path: Path) -> bool:
        """Check if a file or directory exists."""
        pass

    @abstractmethod
    def getsize(self, path: Path) -> int:
        """Get the size of a file in bytes.

        Args:
            path: Path to the file.

        Returns:
            Size of the file in bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        pass

    def glob(self, pattern: str) -> list[str]:
        """Match files/directories using glob pattern.

        Default implementation uses ls() and fnmatch filtering.
        Note: This is single level only - does not support recursive patterns.

        Args:
            pattern: Full path glob pattern (e.g., "/path/to/checkpoints/checkpoint_*")

        Returns:
            List of matching paths as strings.
        """

        # Extract directory and pattern parts
        directory = os.path.dirname(pattern)
        file_pattern = os.path.basename(pattern)

        if not self.exists(Path(directory)):
            return []

        try:
            entries = self.ls(Path(directory))
            matches = fnmatch.filter(entries, file_pattern)
            return [os.path.join(directory, entry) for entry in matches]
        except Exception:
            return []

    @abstractmethod
    def isdir(self, path: Path) -> bool:
        """Check if path is a directory.

        Args:
            path: Path to check.

        Returns:
            True if path is a directory, False otherwise.
        """
        pass

    @abstractmethod
    def remap_path(self, path: Path) -> str:
        """Remaps a local path to the corresponding URL of the storage backend.

        E.g. a backend may map a local mount path to a remote object-store URL.
        """
        pass
