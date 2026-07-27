# Extensibility — building bespoke components

`torch_checkpointing` is built from small, swappable components behind stable interfaces. Most users never touch them — the defaults (local filesystem storage, `torch.save` serialization, DTensor resharding) cover the common cases, and the [Key concepts](./key_concepts.md) guide is all you need.

Reach for this page when you need to integrate the library with **your own infrastructure**: a custom object store, a bespoke parallelism layout, custom cross-rank coordination, or a different on-disk format. For each extension point you implement one interface and plug it in; everything else keeps using the defaults.

This page is the map. Each section covers *what the extension point is for → the interface you implement → the built-in reference implementation to copy → where to go deep*.

## Storage backends — target your own filesystem or object store

Implement a backend when you want checkpoints on something other than a local filesystem (an object store, a distributed FS, an in-memory store for tests).

- **Interface:** `StorageConfig` / `Storage` (`torch_checkpointing.storage.base_storage`).
- **Implement:** `StorageConfig.create_storage()`, returning a `Storage` that implements `stream_read`, `stream_write`, `write`, `delete`, `mkdir`, `rmdir`, `rename`, `ls`, `exists`, `getsize`, `isdir`, and `remap_path`. (`read` and `glob` have working defaults.)
- **Built-in:** `LocalFileSystemStorage` / `LocalFileSystemStorageConfig` (`.storage.filesystem`).
- **Wire it in:** `CheckpointManager.Config(storage_config=YourStorageConfig(...))`.
- **Deep dive:** [Storage](./storage.md) — full method contracts, the O_DIRECT fallback, and a complete in-memory example.

## Resharders — load across a different parallelism layout

Implement a resharder when a checkpoint saved under one parallelism layout must load into another (different world size, mesh, or placement).

- **Interface:** `Resharder` (`torch_checkpointing.resharding`).
- **Implement:** `extract_sharding_metadata(item_key, item_value)` and `load(...)`; optionally override `should_reshard(...)` and the `skip_resharding` property.
- **Built-in:** `DTensorResharder` (`.dtensor_resharder`) — handles `DTensor` mesh/placement changes.
- **Wire it in:** attach per item via `CheckpointItem(resharder=YourResharder())`, and give the manager a `MetadataManager` (`CheckpointManager.Config(checkpoint_metadata_manager=...)`) so sharding metadata is available on load.
- **Deep dive:** [Distributed and resharding](./distributed_and_resharding.md).

## Metadata managers — customize cross-rank metadata aggregation

The metadata manager extracts each item's sharding metadata and aggregates it across ranks; resharding on load depends on it.

- **Interface:** `MetadataManager` — `compute_metadata`, `extract_object_metadata`, `close`.
- **Built-in:** `DefaultMetadataManager` (representative-rank dedup, `all_gather`, async serialization). Sharing one instance between save and load lets a load-then-save reuse the serialized bytes.
- **Wire it in:** `CheckpointManager.Config(checkpoint_metadata_manager=...)`.

## Serialization formats & layout — control the on-disk representation

Change how (and to which files) items are written — e.g. safetensors for tensors, JSON for human-readable metadata.

- **Interface:** `SerializationFormat` (`to_dict` / `from_dict`) plus `LayoutInfo(file_path, serialization_format)` (`.checkpoint_layout`).
- **Built-ins:** `TorchSerialization` (default), `SafetensorsSerialization`, `JsonSerialization(cls)`, `RawSerialization`.
- **Wire it in:** attach per item via `CheckpointItem(layout=LayoutInfo(...))`.
- **Deep dive:** [Configuring checkpoints](./configuring_checkpoints.md).

## Stagers — customize async host-side staging

The stager copies device state to host memory before the background write. Customize it to change buffer strategy or add instrumentation.

- **Interface:** `CheckpointStager` (`stage` / `close`) plus `CheckpointStagerConfig`.
- **Built-in:** `DefaultStager` (pinned / shared memory, non-blocking copy; pinned-memory support is currently CUDA-specific).

## Barriers — custom cross-rank coordination

Barriers coordinate ranks around a save (e.g. so a rename to the final path happens only after every rank has written).

- **Interface:** `BarrierConfig` / `Barrier` — `create_barrier(rank_info)`, `execute_barrier(timeout_secs)`.
- **Built-in:** `TCPStoreBarrier` / `TCPStoreBarrierConfig`.

## Going deeper

[Design & internals](./design.md) walks the save and load pipelines end to end — the async staging path, the background write subprocess and its protocol, and how these components fit together. It's the reference when you're extending behavior that crosses component boundaries.
