# API reference

The public API is small: you interact with `CheckpointManager`, plus `ItemSpec`
when you need to override per-item behavior. Both import from the top level:

```python
from torch_checkpointing import CheckpointManager, ItemSpec
```

Everything else listed below is **advanced / lower-level**: still importable (from
the top-level package or the submodule noted), but not part of the surface a
typical user needs. The building blocks you plug into an `ItemSpec` — layouts,
serialization formats, resharders — live in their own submodules.

## High-level manager (start here)

| Symbol | Purpose |
| --- | --- |
| `CheckpointManager` | The entry point; drives both `save()` and `load()`. |
| `CheckpointManager.Config` | Manager configuration; pass to `CheckpointManager(...)` or call `.build()`. Presets: `Config.with_async_save()`, `Config.with_sync_save()`. Fields include `items`, `default`, `storage_config`. |
| `ItemSpec` | Per-item overrides in `Config.items`: `requires_copy`, `layout`, `resharder`, `required`. |

- `manager.save(checkpoint_id, checkpoint)` — `checkpoint` is a `Mapping[str, Any]`. Returns the write `Future` for async saves, or `None` for sync.
- `manager.load(checkpoint_id, into=None, *, map_location=None, strict=False)` — restores into the `into` templates in place and returns the loaded `Mapping`.
- `manager.lock()` — context manager; wrap `optimizer.step()` in it to wait for any in-flight async staging to finish before mutating params, so a checkpoint isn't staged mid-step. Waits only on the staging copy, not the write. No-op for sync.
- `manager.close()` — waits for the last write and releases resources.

`checkpoint_id` is a string interpreted by the configured storage backend. The
default local filesystem backend treats it as the path to a checkpoint
directory. `checkpoint` is the generic payload: its top-level values may be
tensors, nested state dictionaries, JSON-compatible values, or bytes, subject
to the selected serialization format.

## Configuring items — `ItemSpec` + building blocks

| Symbol | Module | Purpose |
| --- | --- | --- |
| `ItemSpec` | top-level | Per-item `requires_copy` / `layout` / `resharder` / `required`. |
| `LayoutInfo` | `.checkpoint_layout` | Where/how an item is written (`file_path`, `serialization_format`); `{rank}` in `file_path` is filled per rank. |
| `TorchSerialization` | `.checkpoint_layout` | `torch.save` format (the default). |
| `JsonSerialization(cls=None)` | `.checkpoint_layout` | JSON; `None` returns the raw JSON-decoded value. |
| `RawSerialization` | `.checkpoint_layout` | Raw bytes. |
| `SafetensorsSerialization` | `.checkpoint_layout` | safetensors format for tensors. |
| `DefaultResharder` | `.default_resharder` | Built-in resharder for `DTensor` state. |
| `Resharder` | `.resharding` | Base class for custom resharding. |

## Lower-level savers & loaders (advanced)

`CheckpointManager` wraps these — reach for them only when you need finer control
than the manager exposes (for example, driving a load-only eval job by hand).

| Symbol | Module | Purpose |
| --- | --- | --- |
| `CheckpointBase`, `CheckpointItem` | `.checkpoint_base` | Low-level item contract the manager builds internally. |
| `make_async_checkpoint_saver(...)`, `make_sync_checkpoint_saver(...)` | `.builder` | Build a saver directly (auto-detects rank). |
| `AsyncCheckpointSaver`, `SyncCheckpointSaver`, `CheckpointSaver` | `.checkpoint_saver` | Saver classes returned by the factories. |
| `CheckpointSaverConfig`, `AsyncCheckpointSaverConfig`, `SyncCheckpointSaverConfig` | `.config` | Save-side manager configuration, including `wait_timeout_secs`. |
| `CheckpointLoaderConfig` | `.config` | Load-side manager configuration (`use_mmap`). |
| `CheckpointStager`, `DefaultStager`, `CheckpointStagerConfig` | `.staging` | Async staging. |
| `CheckpointReader` | `.checkpoint_reader` | Reads bytes from storage into a state dict. |
| `CheckpointLoader` | `.checkpoint_loader` | Wraps a reader and applies a `load_state_dict`. |

## Storage — `torch_checkpointing.storage`

| Symbol | Module | Purpose |
| --- | --- | --- |
| `LocalFileSystemStorageConfig`, `LocalFileSystemStorage` | `.storage.filesystem` | Shipped local-filesystem backend (the default). |
| `Storage`, `StorageConfig`, `ReadArgs` | `.storage.base_storage` | Base classes for a custom backend. |

## Distributed & resharding (advanced)

| Symbol | Module | Purpose |
| --- | --- | --- |
| `RankInfo` | `.types` | Rank identity (auto-detected by the manager). |
| `MetadataManager`, `DefaultMetadataManager` | `.metadata_manager` | Sharding-metadata pipeline (auto-wired by the manager when an item has a resharder). |
| `Barrier`, `BarrierConfig` | `.barriers` | Cross-rank save coordination. |
| `TCPStoreBarrier`, `TCPStoreBarrierConfig` | `.barriers` | Barrier on a `TCPStore` of its own, served by rank 0. |
| `DefaultStoreBarrier`, `DefaultStoreBarrierConfig` | `.barriers` | Barrier on the store that already backs the default process group. |

See the [tutorial](./tutorials.md) for an end-to-end workflow,
[Key concepts](./key_concepts.md) for how these pieces fit together,
[Extensibility](./extensibility.md) for the extension points, and
[Distributed and resharding](./distributed_and_resharding.md) for the
distributed API in depth.
