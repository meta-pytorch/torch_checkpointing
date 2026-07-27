# Design & Internals

> **Optional deep-dive.** You do not need this document to *use* the library — start with the [README](../README.md) and [Key concepts](./key_concepts.md). This is a contributor-oriented tour of the internals; read it when extending the library or debugging its behavior. For the extension points at a glance, see [Extensibility](./extensibility.md).

This document describes the internal architecture of `torch_checkpointing`: the
end-to-end save and load pipelines, the difference between the synchronous and
asynchronous savers, the staging subsystem, the background write subprocess, and
how the reader reconstructs a state dict from storage.

For a higher-level overview, see [../README.md](../README.md). For the core
vocabulary (`CheckpointBase`, `CheckpointItem`, layout, metadata), see
[./key_concepts.md](./key_concepts.md).

Unlike the README — which focuses on the asynchronous path — this document also
documents the synchronous saver in full, because the two paths share most of
their machinery and differ only in *when* and *where* the write happens.

## The two savers

Both savers implement the abstract base class `CheckpointSaver`:

```python
class CheckpointSaver(abc.ABC):
    @abc.abstractmethod
    def save(
        self,
        path: str,
        checkpoint: CheckpointBase,
    ) -> tuple[Future, Future] | None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    def stager(self) -> CheckpointStager | None:
        return None
```

`save` takes a destination `path` and a `CheckpointBase` — the user-facing object
whose `get_items()` returns the `dict[str, CheckpointItem]` to persist. The
return type is the key behavioral difference: the synchronous saver returns
`None` (everything is done by the time `save` returns), while the asynchronous
saver returns a `(stage_future, write_future)` pair.

The `stager` property returns the staging cache for savers that have one, or
`None` for savers that write directly without pinning host memory. It is used to
pre-allocate the staging pool before the first real save and to report
`pinned_num_bytes`.

### Construction

Use the factory functions in `builder.py` rather than constructing savers by
hand. They auto-detect rank information from the default process group (falling
back to single-rank when distributed is not initialized) and wire up sensible
defaults:

```python
from torch_checkpointing.builder import (
    make_sync_checkpoint_saver,
    make_async_checkpoint_saver,
)

sync_saver = make_sync_checkpoint_saver()
async_saver = make_async_checkpoint_saver()
```

Both factories accept `config`, `rank_info`, `pre_finalize_callback`,
`finalize_callback`, `storage_config`, and `checkpoint_metadata_manager`. The
async factory additionally accepts `subprocess_init_fn` and
`subprocess_init_args` (see [The write subprocess](#the-write-subprocess)).

## The synchronous save pipeline

`SyncCheckpointSaver` is the simple case. It holds a `CheckpointWriter` and an
optional `MetadataManager`, and does all its work inline on the calling thread:

```python
class SyncCheckpointSaver(CheckpointSaver):
    def __init__(
        self,
        writer: CheckpointWriter,
        metadata_manager: MetadataManager | None = None,
    ): ...
```

`save` does three things:

1. `_prepare_checkpoint_write_info(checkpoint)` — calls `checkpoint.get_items()`,
   and if a `MetadataManager` is present, computes the distributed metadata once
   and pickles `metadata.distributed_metadata.to_dict()` into bytes. It returns a
   `CheckpointWriteInfo` via `CheckpointInfo(...).for_writes(serialized_metadata)`.
2. `self._writer.write(path, write_info)` — writes files to storage on the
   current thread.
3. Returns `None`.

There is no staging and no subprocess: the tensors are serialized straight from
wherever they live. This is the right choice when the trainer can afford to block
for the duration of the write.

## The asynchronous save pipeline

`AsyncCheckpointSaver` adds two things on top of the sync path: **staging** (copy
the state dict off the training path so the trainer can resume immediately) and a
**background write subprocess** (do the actual I/O in a separate process so it
does not contend with the training process for CPU or the GIL).

```python
class AsyncCheckpointSaver(CheckpointSaver):
    def __init__(
        self,
        checkpoint_stager: CheckpointStager,
        checkpoint_process: CheckpointProcess,
        metadata_manager: MetadataManager | None = None,
    ): ...

    def save(
        self,
        path: str,
        checkpoint: CheckpointBase,
        validate_state_dict: bool = False,
    ) -> tuple[Future, Future]: ...
```

`save` proceeds roughly as follows:

1. **Prepare write info.** `_prepare_checkpoint_write_info(checkpoint,
   validate=validate_state_dict)` computes metadata once (kicking off async
   serialization) and returns a `CheckpointWriteInfo`. From it, `save` reads
   `state_dict` and `layout_info_mappings`.
2. **Wait for the previous write.** If a prior `self._write_future` is
   outstanding, `save` blocks on `self._write_future.result()`. Only one
   asynchronous checkpoint is in flight at a time; a new `save` waits for the
   previous one to finish before staging.
3. **Select keys to stage.** Items whose `requires_copy` is `False` are collected
   into `keys_not_requiring_staging` and excluded from the staged subset. When
   there are such keys, only the remainder is staged and the results are merged
   back with `_merge_staged_with_full_state_dict`; otherwise the whole state dict
   is staged.
4. **Stage.** `self._checkpoint_stager.stage(state_dict=...)` returns either a
   staged dict or a `Future`, normalized with `ensure_future`. Subsequent
   transforms are chained onto it with `fut_then`.
5. **Build write info from staged tensors.** A continuation rebuilds a
   `dict[str, CheckpointItem]` from the staged values, attaching the recorded
   `layout` per key, attaches serialized metadata on the first write only, and
   returns a `CheckpointWriteInfo`.
6. **Hand off to the subprocess.** `self._checkpoint_process.write(
   checkpoint_write_info_fut, path)` submits the write and returns a `Future`,
   stored as `self._write_future`.
7. **Return** `(wrap_future(staging_fut), self._write_future)`.

The caller can wait on `stage_future` to know when its tensors are safe to
mutate again, and on `write_future` to know when the checkpoint is durable.

### `validate_state_dict` debug mode

`save` accepts `validate_state_dict: bool = False`. When `True`, before staging,
`save` writes a synchronous reference copy of the state dict to a temporary file
with `torch.save`. After staging, `_throw_if_not_equal_to_sync_save` reloads that
reference with `torch.load(..., mmap=True, weights_only=False)` and calls
`compare_state_dicts` against the staged copy. Any difference raises a
`ValueError` — this catches cases where the training loop mutated model state
during the forward/backward pass while staging was in flight. It also forces
metadata re-computation for consistency checking.

This is **extremely expensive** (it doubles the serialization work and reads the
whole checkpoint back) and is intended only for testing and debugging, not
production runs.

## Staging

Staging copies tensors out of the (typically GPU-resident) training state into
CPU storage so the training loop can proceed while the write happens in the
background. The interface is `CheckpointStager`:

```python
class CheckpointStager(abc.ABC):
    @abc.abstractmethod
    def stage(
        self,
        state_dict: STATE_DICT,
    ) -> STATE_DICT | Future[STATE_DICT]: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    def pinned_num_bytes(self) -> int:
        return 0
```

The default implementation is `DefaultStager`, configured by
`CheckpointStagerConfig`:

```python
@dataclass
class CheckpointStagerConfig:
    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = True
    thread_name: str = "ckpt-staging"
    metric_prefix: str = "train.checkpoint_write"
```

### The four flags

- **`use_pinned_memory`** — allocate the CPU staging buffers as pinned
  (page-locked) memory, which makes device-to-host copies faster. Passed through
  to the underlying `StateDictStager` as `pin_memory`.
- **`use_shared_memory`** — allocate staged storage in shared memory, so the
  background write subprocess can access the tensors without an extra copy across
  the process boundary. Passed through as `share_memory`.
- **`use_async_staging`** — run the staging copy on a background single-worker
  `ThreadPoolExecutor` so `stage` returns a `Future` immediately instead of
  blocking. Drives the executor setup in `DefaultStager.__init__`.
- **`use_non_blocking_copy`** — issue the device-to-host copies with
  `non_blocking=True` and rely on stream synchronization, letting CPU work
  continue while the transfer runs. Passed through to `StateDictStager` as
  `use_non_blocking_copy`.

### Accelerator gating

`DefaultStager` records `torch.accelerator.is_available()` as
`self._accelerator_available` and enforces the following at construction:

- `use_non_blocking_copy` **asserts** that an accelerator is available:

  ```python
  if self._config.use_non_blocking_copy:
      assert self._accelerator_available, (
          "Non-blocking copy requires that the current accelerator is available."
      )
  ```

- The staging `torch.Stream` is created only when `use_async_staging` is set
  **and** an accelerator is available.

At `stage` time, the accelerator path records a `torch.Event` on the current
stream and has the staging stream wait on it, so the copy does not begin until
the trainer's in-flight work (e.g. the optimizer step) has completed. When no
accelerator is available, the stager falls back to a plain synchronous copy and
measures latency with a wall timer instead of CUDA events.

`DefaultStager` also captures the main thread's CUDA device in
`self._cuda_device` and replays it on the staging worker via
`_init_worker_thread`. `torch.cuda.set_device` is thread-local; without this the
worker would default to device 0 and the pinning calls would create a stray
primary CUDA context there for every rank.

`pinned_num_bytes()` reports the total pinned host memory held by the staging
storage pool; the base class returns `0` for stagers that do not pin memory.

## Saver configuration

The saver configs live in `config.py` and simply compose the component configs:

```python
@dataclass(kw_only=True)
class CheckpointSaverConfig:
    wait_timeout_secs: int | None = 600


@dataclass
class SyncCheckpointSaverConfig(CheckpointSaverConfig):
    writer_config: CheckpointWriterConfig = field(
        default_factory=CheckpointWriterConfig
    )


@dataclass
class AsyncCheckpointSaverConfig(CheckpointSaverConfig):
    writer_config: CheckpointWriterConfig = field(
        default_factory=CheckpointWriterConfig
    )
    staging_config: CheckpointStagerConfig = field(
        default_factory=CheckpointStagerConfig
    )
    process_config: CheckpointProcessConfig = field(
        default_factory=CheckpointProcessConfig
    )
```

The base config carries the timeout used by `CheckpointManager` when waiting for
staging or writing to finish. The synchronous config adds only a `writer_config`,
because the sync path has neither a stager nor a subprocess. The asynchronous
config also adds `staging_config` (the `CheckpointStagerConfig` above) and
`process_config` (`CheckpointProcessConfig`, described next).

## The write subprocess

`CheckpointProcess` owns a single spawned worker process that performs the actual
file writes for the asynchronous saver, keeping I/O work out of the training
process. It is configured by:

```python
@dataclass
class CheckpointProcessConfig:
    subprocess_init_timeout_secs: int = 30
    subprocess_shutdown_timeout_secs: int = 60
    thread_name_prefix: str = "ckpt"
```

Its constructor takes the initialization hook and the writer args:

```python
class CheckpointProcess:
    def __init__(
        self,
        rank_info: RankInfo,
        config: CheckpointProcessConfig,
        subprocess_init_fn: Callable[[Any], None],
        subprocess_init_args: tuple[Any, ...],
        checkpoint_writer_args: CheckpointWriterArgs,
    ): ...
```

### `subprocess_init_fn` / `subprocess_init_args`

The child process is spawned (via `torch.multiprocessing`, spawn context) and
its first act is to call `subprocess_init_fn(*subprocess_init_args)`. This is the
hook for application-specific setup inside the write process — for example
importing modules that register custom tensor types, or configuring logging.

The default (`make_async_checkpoint_saver`) is `_default_subprocess_init`, a
module-level no-op. It must be a module-level function, **not** a lambda or
closure, because it is pickled to be sent to the spawned process. The init args
must likewise be picklable.

After `subprocess_init_fn` returns, the child builds its writer with
`checkpoint_writer_args.build()` and enters a request loop.

### Lifecycle and protocol

Construction submits `_create_subprocess` on a single-worker executor, so the
process spawns in the background while the trainer continues. The parent and
child communicate over a `multiprocessing` `Pipe` using pickled `WorkerRequest` /
`WorkerResponse` messages. Request types are:

```python
class RequestType(Enum):
    PING = "ping"
    WRITE_CHECKPOINT = "write_checkpoint"
    TERMINATE_PROCESS = "exit"
```

- **Startup handshake** — after spawning, the parent sends a `PING` and blocks up
  to `subprocess_init_timeout_secs` for the reply, raising `TimeoutError` on
  timeout. `wait_for_init()` blocks until the child is ready; `check_ok()` raises
  if the creation future failed or the child died.
- **Write** — `write(checkpoint_info, path)` accepts either a
  `CheckpointWriteInfo` or a `Future[CheckpointWriteInfo]`, submits the work on
  the comm executor, and returns a `Future[None]`. The comm thread resolves the
  future (waiting on staging if needed), sends a `WRITE_CHECKPOINT` request, and
  waits for the response. The child caches `serialized_distributed_metadata` from
  the first write that carries it and reuses it for subsequent writes, so
  metadata is serialized and sent only once.
- **Shutdown** — `close()` shuts down the executor, sends `TERMINATE_PROCESS`,
  and joins the child up to `subprocess_shutdown_timeout_secs`, killing it if it
  does not exit gracefully.

The checkpoint logging context (including the current `step`) is exported on
every request and re-imported in the child, and the comm thread snapshots
`contextvars` at submission time so the recorded step is the one set when `save`
was called, not whatever the trainer advanced to later.

## The writer

Both savers ultimately drive a `CheckpointWriter` (the sync saver directly, the
async saver inside the subprocess). `write(path, checkpoint_info)` iterates
`checkpoint_info.layout_info_mappings`, skipping keys absent from the state dict
and filling in `default_layout_info(key, global_rank)` where no layout was
specified. Files are written in parallel by a `ThreadPoolExecutor` sized by
`file_write_max_threads`, and `_write_key` dispatches on the layout's
serialization format:

- **`TorchSerialization`** — `torch.save` into a streamed write.
- **`JsonSerialization`** — `json.dumps` (sorted keys, indented), UTF-8 encoded.
- **`RawSerialization`** — writes `bytes` directly (raises if the value is not
  already bytes).
- **`SafetensorsSerialization`** — flattens tensors with
  `prepare_tensors_for_save`, then `safetensors.torch.save`.

Metadata is written by `role_rank == 0` only. When a barrier is configured, the
writer stages files under a `tmp_` directory, waits for all ranks at the barrier,
then renames to the final path for atomicity; `pre_finalize_callback` runs before
the barrier and `finalize_callback` after. With no barrier configured, the writer
writes directly to the final path and skips synchronization.

## The read pipeline

`CheckpointReader.read` reconstructs a state dict from storage:

```python
def read(
    self,
    path: str,
    checkpoint_info: CheckpointReadInfo,
    map_location: Any = None,
) -> tuple[STATE_DICT, list[str]]: ...
```

Only the keys present in `checkpoint_info` are loaded, and each file is read in
full. It returns the loaded state dict together with a list of missing keys.

### Resharding fast path vs. normal path

The reader inspects each item's `resharder`. If **no** item has a resharder, or
**all** resharders have `skip_resharding=True`, it takes a fast path
(`_read_without_resharding`) that skips metadata loading entirely and reads files
directly. Otherwise it loads the source `DistributedMetadata`, pivots it into
per-source-rank layout mappings, and — per item — calls
`Resharder.should_reshard(source_item_metadata, target_metadata)` to split items
into those that need resharding (routed through `_read_with_resharding`, which
delegates to each item's `Resharder.load`) and those that do not (routed through
`_read_without_resharding`).

### Dispatch on serialization format

For non-resharded items, `_load_full_file` resolves the per-key
`LayoutInfo` (falling back to `default_layout_info`) and dispatches on the
serialization format:

- **`TorchSerialization`** — by default routes through
  `load_torch_serialized_from_storage` (a single mmap-backed storage that reduces
  allocator fragmentation after load cleanup). Setting
  `disable_use_mmap_backed_storage_on_load=True` on the reader falls back to the
  `torch.load` path over a streamed read.
- **`JsonSerialization`** — reads bytes, `json.loads`, and rehydrates via
  `from_dict(serialization_format.cls, ...)`.
- **`RawSerialization`** — returns the raw bytes.
- **`SafetensorsSerialization`** — `safetensors.torch.load`; `map_location`, if
  given, must be a `str` or `torch.device` (callables and dict remappings, which
  `torch.load` accepts, are rejected loudly).

Any other format raises `ValueError`.

### Re-nesting to the target

safetensors only supports a flat `dict[str, Tensor]`, so the writer flattens
nested inputs using `.` separators. On load, when the format is
`SafetensorsSerialization`, the requested value is non-`None`, and the loaded
data is a dict, the reader re-nests with
`SafetensorsSerialization.unflatten_to_target(loaded_data, requested_value)`
before merging. This is required because `walk_checkpoint_structure` descends the
source and target in parallel — a flat source against a nested target would
misalign and silently drop everything.

Finally, `walk_checkpoint_structure` merges the loaded data into the requested
target. Where the target holds real objects (not `None`), mutable containers and
tensors are updated **in place** (tensors via `copy_`, preserving object
identity); immutable containers and non-tensor leaves are replaced. Where the
target is `None`, new objects are created from the loaded data. Keys present in
the request but absent from the file are reported in the returned missing-keys
list.
