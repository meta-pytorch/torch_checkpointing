# Troubleshooting & FAQ

Common issues and how to resolve them. Each entry lists the symptom, the cause,
and the fix. For the complete happy path, start with the
[training-loop tutorial](./tutorials.md).

## `save()` returned a single value (or `None`), not the `(stage, write)` tuple I expected

**Cause.** `CheckpointManager.save(checkpoint_id, {...})` returns the **write `Future`** for an async manager (`CheckpointManager.Config.with_async_save()`), or `None` for a sync one (`CheckpointManager.Config.with_sync_save()`). There is no `(stage_future, write_future)` tuple at this level — the two-future pair belongs to the [lower-level `AsyncCheckpointSaver`](./api_reference.md#lower-level-savers--loaders-advanced), which the manager wraps.

**Fix.** Treat the return as one value. Block on it only where you need the bytes durable in storage:

```python
from torch_checkpointing import CheckpointManager

manager = CheckpointManager(CheckpointManager.Config.with_async_save())
write_future = manager.save(checkpoint_id, {"model": model.state_dict(), "step": step})
# ... training continues ...
write_future.result()   # wait only where durability matters
```

The manager also serializes successive saves: the next `save()` waits on the previous write before staging again, so you rarely need to hold the future at all — `close()` blocks on the final write for you.

## Save fails on a CPU-only host (pinned memory / non-blocking copy)

**Cause.** Async staging defaults to pinned (page-locked) host memory and a non-blocking device-to-host copy, both of which currently require CUDA. On a CPU-only host they raise.

**Fix.** Gate pinned memory on CUDA availability when building the manager:

```python
import torch
from torch_checkpointing import CheckpointManager
from torch_checkpointing.config import AsyncCheckpointSaverConfig
from torch_checkpointing.staging import CheckpointStagerConfig

use_cuda = torch.cuda.is_available()
manager = CheckpointManager(
    CheckpointManager.Config(
        save=AsyncCheckpointSaverConfig(
            staging_config=CheckpointStagerConfig(
                use_pinned_memory=use_cuda,
                use_non_blocking_copy=use_cuda,
            )
        )
    )
)
```

The same flags apply if you construct a stager directly:

```python
from torch_checkpointing import CheckpointStagerConfig

staging = CheckpointStagerConfig(
    use_pinned_memory=torch.cuda.is_available(),
    use_non_blocking_copy=torch.cuda.is_available(),
)
```

## `cannot pickle ...` when saving asynchronously

**Cause.** An async manager runs the writer in a separate subprocess, so anything handed to that subprocess must be picklable. Lambdas and closures are not. This bites most often with the lower-level write callbacks (`pre_finalize_callback`, `finalize_callback`, `subprocess_init_fn`) and with values in your payload that capture unpicklable state.

**Fix.** Use module-level functions or `functools.partial` instead of lambdas/closures for any callable that reaches the subprocess (for example the callbacks accepted by `make_async_checkpoint_saver`), and keep every payload value picklable. This requirement also applies to values with `requires_copy=False`; bypassing the staging copy does not bypass subprocess serialization. A sync manager (`CheckpointManager.Config.with_sync_save()`) uses no subprocess, so it sidesteps the pickling requirement entirely.

## Async save crashes spawning its background process (re-runs your script, or `FileNotFoundError: <stdin>` / `Connection reset by peer`)

**Cause.** Async saves write from a background process started with multiprocessing `spawn`, which re-imports your entry-point module. If your script runs the checkpointing at top level (no `if __name__ == "__main__"` guard), the child re-executes it, spawning more processes and crashing.

**Fix.** Put your entry point behind the standard guard so `spawn` can import the module without re-running it:

```python
def main():
    manager = CheckpointManager(CheckpointManager.Config.with_async_save())
    ...


if __name__ == "__main__":
    main()
```

This is the standard requirement for any code using `multiprocessing` / `spawn` (including `torch.distributed`). A sync manager (`CheckpointManager.Config.with_sync_save()`) uses no subprocess and needs no guard.

## `load()` restored nothing (or only some keys)

**Cause.** Loading is template-driven. The `into=` mapping does double duty: it names which items to read, *and* its values are the templates the loader restores into. Filtering is recursive: at every nesting level, only mapping keys and sequence indices present in a structured template are loaded. A too-narrow (or empty, or omitted) `into=` therefore reads nothing back for the omitted state.

**Fix.** Pass an `into=` mapping whose templates declare the state you want back:

```python
restored = manager.load(checkpoint_id, into={
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": 0,
})
```

Tensors in the templates are restored in place; `load()` also returns the loaded mapping so you can read scalars (`restored["step"]`) and re-apply object state (`optimizer.load_state_dict(restored["optimizer"])`). Pass `strict=True` to raise on missing keys instead of skipping them silently. (`into=None` reads the items declared in the config as-is — handy for inspecting a checkpoint, but see the resharding entry below.)

## Resharding didn't run — I got the saved layout back unchanged

**Cause.** With `CheckpointManager` the metadata pipeline is wired for you on both save and load, so the classic "forgot to attach a `MetadataManager` on one side" footgun is gone. What remains is that resharding needs **live targets to reshard into** — it computes each rank's target shard from the objects you pass. `load(..., into=None)` has no targets, so it takes the direct-read path and returns the stored shards unchanged. (The same happens if the item simply has no `resharder` declared.)

**Fix.** Declare a resharder on the item in your config, and load with `into=` supplying the tensors built under the *current* layout:

```python
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.dtensor_resharder import DTensorResharder

manager = CheckpointManager(CheckpointManager.Config(
    items={"model": ItemSpec(resharder=DTensorResharder())},
))
manager.save(path, {"model": model.state_dict()})      # records source sharding metadata
manager.load(path, into={"model": model.state_dict()})  # reshards onto the new layout
```

The built-in `DTensorResharder` handles `DTensor` state; write a custom `Resharder` for other sharded types. See [Distributed and resharding](./distributed_and_resharding.md).

## `save()` / `load()` raises on a key like `"model.pt"` or `"optim/state"`

**Cause.** Each top-level payload key names one item and becomes part of its on-disk file name, so the checkpoint engine requires keys to match **`^[a-zA-Z0-9_-]+$`** — alphanumeric characters, hyphens, and underscores only. Dots, slashes, whitespace, and file extensions are rejected.

**Fix.** Use a bare name (`"model"`, `"optimizer"`, `"step"`) as the payload key. If you need a specific on-disk file name or extension, set it through the item's layout instead — `ItemSpec(layout=LayoutInfo("model_{rank}.pt", TorchSerialization()))` in `Config.items` — not in the payload key.

## Checkpoint captured a half-applied optimizer step

**Cause.** Async staging copies params off-device concurrently with training. If `optimizer.step()` mutates params while that copy is still in flight, the checkpoint captures a torn, half-updated state.

**Fix.** Wrap the in-place update in `manager.lock()` — it blocks until any in-flight staging copy has finished, so the step runs against a captured snapshot (staging still overlaps the rest of the step):

```python
with manager.lock():
    optimizer.step()
optimizer.zero_grad()
```

`lock()` is a no-op for sync managers, which stage inline.

## Checkpoint is missing or truncated after the process exits

**Cause.** An async `save()` returns as soon as staging is done; the bytes are written by a background subprocess afterward. If the process exits before that write finishes, the checkpoint can be missing or partial.

**Fix.** Call `manager.close()` before exit — it blocks until the last background write is durable and then releases resources. (Equivalently, block on the write `Future` returned by the final `save()`.) A sync manager has already written by the time `save()` returns, so it has nothing outstanding at exit.

## `RuntimeError: State dictionary has changed since last checkpoint`

**Cause.** The metadata pipeline caches sharding metadata across saves by default. It raises this when the payload's structure or sharding differs from the previous checkpoint, because the cached global view is no longer valid. You only hit this on the resharding path (an item with a `resharder`), where sharding metadata is computed.

**Fix.** Keep the structure and sharding of resharded items stable across checkpoints. If they genuinely change between saves and you need metadata caching off, drive the [lower-level metadata manager](./distributed_and_resharding.md) with caching disabled:

```python
from torch_checkpointing.metadata_manager import DefaultMetadataManager

manager = DefaultMetadataManager(rank_info=rank_info, should_cache_metadata=False)
```

## Warning about `O_DIRECT` / direct I/O

**Cause.** The local filesystem backend attempts `O_DIRECT` (to bypass the page cache during large I/O) and automatically falls back to buffered I/O when the filesystem doesn't support it, logging a warning. This is benign — the checkpoint still works.

**Fix.** Nothing required. To silence the warning on filesystems without direct-I/O support, disable it on the storage config and pass it to the manager:

```python
from torch_checkpointing import CheckpointManager
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig

manager = CheckpointManager(CheckpointManager.Config(
    storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
))
```
