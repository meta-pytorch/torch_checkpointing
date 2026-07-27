# API reference

The public API. Most symbols import directly from the top-level package:

```python
from torch_checkpointing import CheckpointManager, CheckpointBase, CheckpointItem
```

A few live in submodules (noted below): layouts and serialization formats, the built-in DTensor resharder, and storage backends.

## High-level manager (start here)

| Symbol | Purpose |
| --- | --- |
| `CheckpointManager` | The high-level entry point; drives both `save()` and `load()`. |
| `CheckpointManager.Config` | Manager configuration; call `.build()` to construct. Presets: `Config.with_async_save()`, `Config.with_sync_save()`. |
| `AsyncCheckpointSaverConfig`, `SyncCheckpointSaverConfig` | Save-side config on `Config.save`, including `wait_timeout_secs`. |
| `CheckpointLoaderConfig` | Load-side config on `Config.load` (`use_mmap`). |

`manager.save(checkpoint_id, checkpoint)` returns `(stage_future, write_future)` for async saves, or `None` for sync. `manager.load(checkpoint_id, checkpoint, *, map_location=None, strict=False)` loads in place. Call `manager.close()` when done.

## Defining a checkpoint

| Symbol | Purpose |
| --- | --- |
| `CheckpointBase` | Subclass and implement `get_items()` / `load_state_dict()`. |
| `CheckpointItem` | Per-item config: `value`, `requires_copy`, `layout`, `resharder`. |
| `STATE_DICT` | Type alias for a checkpoint state dict. |

## Lower-level savers & loaders

`CheckpointManager` wraps these — reach for them directly only when you need finer control than the manager exposes.

| Symbol | Purpose |
| --- | --- |
| `make_async_checkpoint_saver(...)`, `make_sync_checkpoint_saver(...)` | Build a saver directly (auto-detects rank). |
| `AsyncCheckpointSaver`, `SyncCheckpointSaver`, `CheckpointSaver` | Saver classes returned by the factories. |
| `AsyncCheckpointSaverConfig`, `SyncCheckpointSaverConfig` | Saver configuration. |
| `CheckpointStagerConfig`, `CheckpointStager`, `DefaultStager` | Async staging configuration and stager. |
| `CheckpointReader` | Reads bytes from storage into a state dict. |
| `CheckpointLoader` | Wraps a reader and calls your `load_state_dict()`. |

## Storage — `torch_checkpointing.storage`

| Symbol | Module | Purpose |
| --- | --- | --- |
| `LocalFileSystemStorageConfig`, `LocalFileSystemStorage` | `.storage.filesystem` | Shipped local-filesystem backend. |
| `Storage`, `StorageConfig`, `ReadArgs` | `.storage.base_storage` | Base classes for a custom backend. |

## Layout & serialization — `torch_checkpointing.checkpoint_layout`

| Symbol | Purpose |
| --- | --- |
| `LayoutInfo` | Where/how an item is written (`file_path`, `serialization_format`). |
| `TorchSerialization` | `torch.save` format (used by the default layout). |
| `JsonSerialization(cls)` | JSON, for human-readable metadata. |
| `RawSerialization` | Raw bytes. |
| `SafetensorsSerialization` | safetensors format for tensors. |

## Distributed & resharding

| Symbol | Module | Purpose |
| --- | --- | --- |
| `RankInfo` | top-level | Rank identity (auto-detected by the manager and the factories). |
| `Resharder` | top-level | Base class for custom resharding. |
| `DTensorResharder` | `.dtensor_resharder` | Built-in resharder for `DTensor` state. |
| `MetadataManager`, `DefaultMetadataManager` | top-level | Sharding-metadata pipeline (required for resharding). |
| `Barrier`, `TCPStoreBarrier`, `BarrierConfig`, `TCPStoreBarrierConfig` | top-level | Cross-rank save coordination. |

See [Key concepts](./key_concepts.md) for how these fit together, [Extensibility](./extensibility.md) for the extension points, and [Distributed and resharding](./distributed_and_resharding.md) for the distributed API in depth.
