# Storage Backends

`torch_checkpointing` reads and writes checkpoints through a pluggable storage
layer. All I/O — streaming reads and writes, directory management, renames,
existence checks — flows through a small `Storage` interface, so the same
checkpointing logic can target a local filesystem, a networked filesystem, or a
remote object store without changes to the rest of the stack.

Only the local filesystem backend ships with the package. Networked filesystems
can be used through a local mount; a native remote or object-store integration
requires a custom `StorageConfig` / `Storage` implementation.

> **Power-user / extensibility track.** This guide is for building a bespoke
> storage backend. Most users just use the default local filesystem backend — see
> [Extensibility](./extensibility.md) for the full set of extension points and
> [Key concepts](./key_concepts.md) for the common path.

This document covers:

- The `Storage` / `StorageConfig` abstract base classes and their key methods.
- How `create_storage()` connects a `StorageConfig` to its `Storage`.
- The shipped `LocalFileSystemStorage` / `LocalFileSystemStorageConfig`,
  including `use_direct_io` (`O_DIRECT`) with automatic buffered-I/O fallback,
  and `ReadArgs`.
- A sketch of implementing your own backend.

See also the [tutorial](./tutorials.md) and
[Key concepts](./key_concepts.md).

## The two ABCs

The storage layer is defined by two abstract base classes in
`storage/base_storage.py`: `StorageConfig` describes *how to construct* a
backend, and `Storage` is the backend itself. Splitting configuration from the
live object keeps configs cheap to build, serialize, and pass around, while the
`Storage` instance holds any real resources (file descriptors, clients).

### StorageConfig

`StorageConfig` is a dataclass ABC with a single abstract method:

```python
@dataclass
class StorageConfig(ABC):
    @abstractmethod
    def create_storage(self) -> "Storage":
        """Create a storage instance from the config."""
        pass
```

Every backend ships a config class inheriting from `StorageConfig` and
implements `create_storage()` to return its concrete `Storage`.

### Storage

`Storage` is the ABC every backend implements. Its methods fall into a few
groups.

**Streaming I/O (abstract):**

```python
def stream_read(
    self,
    path: Path,
    read_args: ReadArgs | None = None,
) -> io.RawIOBase:
    """Stream data from the given file path in chunks."""

def stream_write(self, path: Path) -> io.IOBase:
    """Return a file-like object for writing."""
```

`stream_read` returns a `RawIOBase` so callers get `readinto` support, which is
important for performance. `stream_write` returns a writable file-like object.

**Whole-file I/O:**

```python
def read(
    self,
    path: Path,
    read_args: ReadArgs | None = None,
) -> bytes:
    """Read the entire contents of a file."""

def write(self, path: Path, data: Buffer) -> None:
    """Write the entire buffer to a file."""
```

`read` is a concrete convenience wrapper: the base class implements it by
opening `stream_read(path, read_args)` in a `with` block and returning
`f.read()`, so a subclass gets it for free once `stream_read` is implemented.
`write` is abstract and must be implemented per backend. `data` is a
`typing_extensions.Buffer`, so implementations should accept any buffer-protocol
object (`bytes`, `bytearray`, `memoryview`, etc.).

**Files and directories (abstract):**

```python
def delete(self, path: Path) -> None:
    """Delete a file."""

def mkdir(self, path: Path, recursive: bool = True) -> None:
    """Create a directory."""

def rmdir(self, path: Path) -> None:
    """Remove a directory."""

def rename(
    self,
    src_path: Path,
    dst_path: Path,
    is_directory: bool = False,
    is_cross_link: bool = False,
    background_cleanup: bool = False,
) -> None:
    """Rename a file or directory."""

def ls(self, path: Path) -> list[str]:
    """List contents of a directory."""
```

`rename` carries several flags used by checkpoint commit logic. Per the base
docstring, `is_directory` renames all objects under the source prefix, and
`background_cleanup` (directory renames only) deletes source objects on a
background thread after copying completes, so the caller is not blocked on the
low-priority cleanup. `is_cross_link` signals a cross-filesystem move; see how
`LocalFileSystemStorage.rename` uses it below.

**Metadata and lookups:**

```python
def exists(self, path: Path) -> bool:
    """Check if a file or directory exists."""

def getsize(self, path: Path) -> int:
    """Get the size of a file in bytes."""

def isdir(self, path: Path) -> bool:
    """Check if path is a directory."""

def glob(self, pattern: str) -> list[str]:
    """Match files/directories using glob pattern."""

def remap_path(self, path: Path) -> str:
    """Remaps a local path to the corresponding URL of the storage backend."""
```

`exists`, `getsize`, `isdir`, and `remap_path` are abstract. `glob` is concrete
with a default implementation built on `ls()` + `fnmatch` filtering; it is
single-level only (no recursive patterns) and takes a full-path pattern such as
`"/path/to/checkpoints/checkpoint_*"`. `remap_path` lets a backend translate a
local mount path into the storage backend's own URL (for example, mapping a
mount point to a remote object-store URL); backends with no such mapping can
just return `str(path)`.

### ReadArgs

Reads are tuned through a `ReadArgs` dataclass, passed to `read` / `stream_read`:

```python
@dataclass
class ReadArgs:
    pre_read_full_file: bool = True
    direct_io: bool = False
    timeout_us: int = 900_000_000
```

- `pre_read_full_file` — read the entire file up front before initializing the
  stream.
- `direct_io` — request direct I/O for reading.
- `timeout_us` — read timeout in microseconds, currently honored only by
  streaming backends that support a client-side read timeout.

When `read_args` is `None`, backends fall back to their own defaults (see how
`LocalFileSystemStorage.stream_read` treats a missing `read_args` below).

## Wiring a backend into a checkpoint

You do **not** call `create_storage()` yourself for the common path. Hand the
*config* to the manager and it constructs the live `Storage` for you:

```python
from torch_checkpointing import CheckpointManager
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig

manager = CheckpointManager(CheckpointManager.Config(
    storage_config=LocalFileSystemStorageConfig(use_direct_io=True),
))
```

Every `save()` and `load()` on that manager then reads and writes through your
backend. When no `storage_config` is supplied, the manager defaults to
`LocalFileSystemStorageConfig()`.

The connection between a config and its backend *is* `create_storage()` — there
is no registry or factory function to memorize. The manager calls it internally,
but you can also call it directly when driving the [lower-level savers and
loaders](./api_reference.md#lower-level-savers--loaders-advanced) by hand, or
when you just want a `Storage` to poke at:

```python
config = LocalFileSystemStorageConfig(use_direct_io=True)
storage = config.create_storage()  # -> LocalFileSystemStorage
```

Because the config owns construction, each backend decides exactly what its
`Storage` needs. This keeps configs serializable and lets code that only needs
to *describe* a backend stay decoupled from code that instantiates it.

## The shipped backend: LocalFileSystemStorage

`storage/filesystem.py` provides the reference backend for a local (or
locally-mounted) filesystem.

### Config

```python
@dataclass
class LocalFileSystemStorageConfig(StorageConfig):
    use_direct_io: bool = True

    def create_storage(self) -> "LocalFileSystemStorage":
        return LocalFileSystemStorage(config=self)
```

The single knob is `use_direct_io`, which defaults to `True`. It is threaded
into the `LocalFileSystemStorage` instance (as `self._use_direct_io`) and
controls whether writes (and reads that opt in via `ReadArgs`) attempt
`O_DIRECT`.

### O_DIRECT with automatic buffered-I/O fallback

`O_DIRECT` bypasses the OS page cache, which avoids cache churn during large
checkpoint I/O — but not every filesystem supports it. The backend attempts
`O_DIRECT` and transparently falls back to buffered I/O when it is not
available, so the same config works everywhere.

**Writes** go through the `WriteStream` file-like object. On entry it first
verifies the parent directory exists (raising `RuntimeError` otherwise), then,
when `use_direct_io` is set, opens the file with
`os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT`. If that raises `OSError`
with `errno == 22` (`EINVAL` — the filesystem does not support `O_DIRECT`), it
logs a warning and falls through to a regular `open(...)`; any other error is
re-raised. You can inspect which path was taken via the `using_direct_io`
property. Both `stream_write` and the whole-file `write` use `WriteStream`:

```python
def stream_write(self, path: Path) -> io.IOBase:
    return WriteStream(path, "wb", self._use_direct_io)

def write(self, path: Path, data: Buffer) -> None:
    with WriteStream(path, "wb", self._use_direct_io) as f:
        f.write(data)
```

**Reads** are governed by `ReadArgs`, *not* by the config's `use_direct_io`.
When no `read_args` is passed, `stream_read` defaults to
`use_direct_io=False` and `pre_read_full_file=True`:

```python
def stream_read(
    self,
    path: Path,
    read_args: ReadArgs | None = None,
) -> io.RawIOBase:
    use_direct_io = read_args.direct_io if read_args else False
    pre_read_full_file = read_args.pre_read_full_file if read_args else True

    if pre_read_full_file:
        return self._read_full_file(path, use_direct_io)

    return self._open_for_read(path, use_direct_io)
```

Read fallback happens in two places. `_open_for_read` opens with
`os.O_DIRECT | os.O_RDONLY` and, on `errno == 22`, warns and falls back to a
buffered `open(path, "rb", buffering=0)`. `_read_full_file` adds a second guard:
if the `O_DIRECT` *read itself* fails with `errno == 22` (for example, inside
containers with volume mounts), it retries the whole read with buffered I/O.

### Directory and rename semantics

`LocalFileSystemStorage` maps the remaining `Storage` methods onto the standard
library:

- `mkdir` uses `os.makedirs(path, exist_ok=True)` when `recursive` (default),
  else `os.mkdir(path)`.
- `rmdir` tries the cheap `os.rmdir(path)` first and falls back to
  `shutil.rmtree(path, ignore_errors=False)`; if both fail it logs and re-raises.
- `rename` uses `shutil.move` for the common case. When `is_cross_link` is set
  *and* the source is a directory, it instead `os.makedirs` the destination,
  `shutil.copytree(..., dirs_exist_ok=True)`, then `shutil.rmtree` the source —
  avoiding `shutil.move`'s nesting behavior when the destination already exists.
- `ls` → `os.listdir`, `delete` → `os.remove`, `exists` → `os.path.exists`,
  `getsize` → `os.path.getsize`, `isdir` → `os.path.isdir`.
- `remap_path` returns `str(path)` — a local path is already its own URL.

## Writing your own backend

Implementing a backend means subclassing both ABCs: a `StorageConfig` whose
`create_storage()` returns your `Storage`, and the `Storage` itself with the
abstract methods filled in. You get `read` and `glob` for free from the base
class. The sketch below targets an in-memory dictionary to keep the mechanics
clear; swap the body of each method for your real object-store or filesystem
client.

```python
import io
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Buffer, override

from torch_checkpointing.storage.base_storage import (
    ReadArgs,
    Storage,
    StorageConfig,
)


@dataclass
class InMemoryStorageConfig(StorageConfig):
    # Backend-specific config lives here.
    root: str = "/"

    @override
    def create_storage(self) -> "InMemoryStorage":
        return InMemoryStorage(config=self)


class InMemoryStorage(Storage):
    def __init__(self, config: InMemoryStorageConfig) -> None:
        self._root = config.root
        self._files: dict[str, bytes] = {}

    @override
    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> io.RawIOBase:
        key = str(path)
        if key not in self._files:
            raise FileNotFoundError(key)
        # BytesIO is a RawIOBase-compatible file-like object.
        return io.BytesIO(self._files[key])

    @override
    def stream_write(self, path: Path) -> io.IOBase:
        # Return a writable buffer that commits to the store on close().
        store, key = self._files, str(path)

        class _CommitOnClose(io.BytesIO):
            def close(self) -> None:
                store[key] = self.getvalue()
                super().close()

        return _CommitOnClose()

    @override
    def write(self, path: Path, data: Buffer) -> None:
        self._files[str(path)] = bytes(memoryview(data))

    @override
    def delete(self, path: Path) -> None:
        self._files.pop(str(path), None)

    @override
    def mkdir(self, path: Path, recursive: bool = True) -> None:
        # No-op: a flat key/value store has no real directories.
        pass

    @override
    def rmdir(self, path: Path) -> None:
        prefix = str(path)
        for key in [k for k in self._files if k.startswith(prefix)]:
            del self._files[key]

    @override
    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        is_cross_link: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        self._files[str(dst_path)] = self._files.pop(str(src_path))

    @override
    def ls(self, path: Path) -> list[str]:
        prefix = str(path).rstrip("/") + "/"
        names = {
            key[len(prefix):].split("/", 1)[0]
            for key in self._files
            if key.startswith(prefix)
        }
        return sorted(names)

    @override
    def exists(self, path: Path) -> bool:
        return str(path) in self._files

    @override
    def getsize(self, path: Path) -> int:
        return len(self._files[str(path)])

    @override
    def isdir(self, path: Path) -> bool:
        prefix = str(path).rstrip("/") + "/"
        return any(key.startswith(prefix) for key in self._files)

    @override
    def remap_path(self, path: Path) -> str:
        return str(path)
```

Note that `read` and `glob` are intentionally absent — the base class supplies
them (`read` via `stream_read`, `glob` via `ls` + `fnmatch`). Override `glob`
only if your backend can match patterns more efficiently server-side.

### Notes for remote object stores

A few contracts matter more when the backend is a remote object store than a local filesystem:

- **`stream_write` durability.** The framework reads a written file only *after* `stream_write(...).close()` returns (and, for barrier-coordinated saves, after the commit rename completes). A backend may therefore buffer writes and flush on `close()` — for example, accumulate in memory or a temp part and issue a single `PUT` / complete a multipart upload in `close()`, as the sketch above does.
- **`isdir` / `ls` on a flat store.** Object stores have no real directories. Implement `isdir(path)` as "does any object exist under this prefix?" and `ls(path)` as a single-level prefix listing (again, as above).
- **`rename` and atomicity.** For barrier-coordinated multi-rank saves the writer stages files under a temporary directory and renames it into place, so readers never observe a half-written checkpoint. If your store has no cheap rename, implement it as copy-then-delete — optionally passing `background_cleanup=True` to defer the source deletes off the critical path.

Once defined, your backend plugs in exactly like the shipped one — pass the
config to the manager and every save and load goes through it:

```python
from torch_checkpointing import CheckpointManager

manager = CheckpointManager(CheckpointManager.Config(
    storage_config=InMemoryStorageConfig(root="/checkpoints"),
))
```

## Reference

- Abstract base classes: `storage/base_storage.py`
  (`Storage`, `StorageConfig`, `ReadArgs`).
- Reference backend: `storage/filesystem.py`
  (`LocalFileSystemStorage`, `LocalFileSystemStorageConfig`, `WriteStream`).
- Related docs: [./key_concepts.md](./key_concepts.md), [../README.md](../README.md).
