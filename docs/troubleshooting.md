# Troubleshooting & FAQ

Common issues and how to resolve them. Each entry lists the symptom, the cause, and the fix.

## `RuntimeError: State dictionary has changed since last checkpoint`

**Cause.** `DefaultMetadataManager` caches sharding metadata across saves (`should_cache_metadata=True`, the default). It raises this when the state dict's structure or sharding differs from the previous checkpoint, because the cached global view is no longer valid.

**Fix.** Only rely on caching when your state dict's structure and sharding are stable across checkpoints. If they change between saves, construct the manager with caching off:

```python
from torch_checkpointing.metadata_manager import DefaultMetadataManager

manager = DefaultMetadataManager(rank_info=rank_info, should_cache_metadata=False)
```

## Save fails on a CPU-only host (pinned memory / non-blocking copy)

**Cause.** `CheckpointStagerConfig` defaults `use_pinned_memory` and `use_non_blocking_copy` to `True`, and those require an accelerator. On a CPU-only host they raise.

**Fix.** With `CheckpointManager`, gate pinned memory on accelerator availability:

```python
import torch
from torch_checkpointing import CheckpointManager

manager = CheckpointManager.Config.async_save(
    pinned_memory=torch.accelerator.is_available(),
).build()
```

If you configure a stager directly, gate both flags:

```python
from torch_checkpointing import CheckpointStagerConfig

staging = CheckpointStagerConfig(
    use_pinned_memory=torch.accelerator.is_available(),
    use_non_blocking_copy=torch.accelerator.is_available(),
)
```

## `cannot pickle ...` when saving asynchronously

**Cause.** Async saves run the writer — and your `pre_finalize_callback`, `finalize_callback`, and `subprocess_init_fn` — in a separate subprocess, so those callables must be picklable. Lambdas and closures are not.

**Fix.** Use module-level functions or `functools.partial` instead of lambdas/closures for any callback passed via `CheckpointManager.Config` (`pre_finalize_callback`, `finalize_callback`, `subprocess_init_fn`) or `make_async_checkpoint_saver`.

## Async save crashes spawning its background process (re-runs your script, or `FileNotFoundError: <stdin>` / `Connection reset by peer`)

**Cause.** Async saves write from a background process started with multiprocessing `spawn`, which re-imports your entry-point module. If your script runs the checkpointing at top level (no `if __name__ == "__main__"` guard), the child re-executes it, spawning more processes and crashing.

**Fix.** Put your entry point behind the standard guard so `spawn` can import the module without re-running it:

```python
def main():
    manager = CheckpointManager.Config.async_save().build()
    ...


if __name__ == "__main__":
    main()
```

This is the standard requirement for any code using `multiprocessing` / `spawn` (including `torch.distributed`). A sync manager (`CheckpointManager.Config.sync_save()`) uses no subprocess and needs no guard.

## Load restored nothing (or only some keys)

**Cause.** Loading is template-driven: the `value` you put in each `CheckpointItem` on load acts as a filter. Only keys/indices present in the template are loaded; a `None` value loads everything for that item.

**Fix.** Make sure your load-time template declares the keys you want back (see [Key concepts](./key_concepts.md)). Pass `strict=True` to `manager.load(...)` to raise on missing keys instead of skipping them silently.

## Resharding didn't run — I got the saved layout back unchanged

**Cause.** Resharding is opt-in on **both** ends. If you attach a resharder but don't give the loader a `MetadataManager` (or vice versa), the read silently falls back to a direct, non-resharded load.

**Fix.** Attach a resharder to each item you want remapped **and** give the manager a `MetadataManager` via `CheckpointManager.Config(checkpoint_metadata_manager=...)` (or pass one to the `CheckpointLoader` directly). See [Distributed and resharding](./distributed_and_resharding.md).

## Warning about `O_DIRECT` / direct I/O

**Cause.** The local filesystem backend attempts `O_DIRECT` (to bypass the page cache during large I/O) and automatically falls back to buffered I/O when the filesystem doesn't support it, logging a warning. This is benign — the checkpoint still works.

**Fix.** Nothing required. To silence the warning on filesystems without direct-I/O support, disable it:

```python
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig

storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
```
