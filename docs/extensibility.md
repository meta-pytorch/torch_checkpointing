# Extensibility — building bespoke components

`torch_checkpointing` is built from small, swappable components behind stable interfaces. Most users never touch them — the defaults (local filesystem storage, `torch.save` serialization, DTensor resharding) cover the common cases, and the [Key concepts](./key_concepts.md) guide is all you need.

Reach for this page when you need to integrate the library with **your own infrastructure**: a custom object store, a bespoke parallelism layout, or custom cross-rank coordination. For each extension point you implement one interface and plug it into `CheckpointManager` — through an `ItemSpec` in `Config.items`, or a field on the `Config` itself; everything else keeps using the defaults.

This page is the map. Each section covers *what the extension point is for → the interface you implement → the built-in reference implementation to copy → where to go deep*.

## Storage backends — target your own filesystem or object store

Implement a backend when you want checkpoints on something other than a local filesystem (an object store, a distributed FS, an in-memory store for tests).

- **Interface:** `StorageConfig` / `Storage` (`torch_checkpointing.storage.base_storage`).
- **Implement:** `StorageConfig.create_storage()`, returning a `Storage` that implements `stream_read`, `stream_write`, `write`, `delete`, `mkdir`, `rmdir`, `rename`, `ls`, `exists`, `getsize`, `isdir`, and `remap_path`. (`read` and `glob` have working defaults.)
- **Built-in:** `LocalFileSystemStorage` / `LocalFileSystemStorageConfig` (`.storage.filesystem`).
- **Wire it in:** pass the config to the manager via `CheckpointManager.Config(storage_config=YourStorageConfig(...))`. The manager calls `create_storage()` internally.
- **Deep dive:** [Storage](./storage.md) — full method contracts, the O_DIRECT fallback, and a complete in-memory example.

## Resharders — load across a different parallelism layout

Implement a resharder when a checkpoint saved under one parallelism layout must load into another (different world size, mesh, or placement).

- **Interface:** `Resharder` (`torch_checkpointing.resharding`).
- **Implement:** `extract_sharding_metadata(item_key, item_value)` and `load(...)`; optionally override `should_reshard(...)` and the `skip_resharding` property.
- **Built-in:** `DefaultResharder` (`.default_resharder`) — handles `DTensor` mesh/placement changes.
- **Wire it in:** attach it per item via `ItemSpec(resharder=YourResharder())` in `CheckpointManager.Config(items={...})`. That is the whole contract: the manager **auto-wires** the sharding-metadata pipeline on both save and load — recording the source layout on save and computing the target layout on load — so a declared resharder actually runs. (Under the hood this is a `MetadataManager`; see the next section.)
- **Deep dive:** [Distributed and resharding](./distributed_and_resharding.md).

## Metadata managers — customize cross-rank metadata aggregation

The metadata manager extracts each item's sharding metadata and aggregates it across ranks; resharding on load depends on it. You do **not** wire one up by hand for the common case — `CheckpointManager` constructs and threads a `DefaultMetadataManager` through both save and load automatically whenever any item declares a resharder. Implement a custom one only to change the aggregation strategy itself.

- **Interface:** `MetadataManager` — `compute_metadata`, `extract_object_metadata`, `close`.
- **Built-in:** `DefaultMetadataManager` (representative-rank dedup, `all_gather`, async serialization). The manager reuses one instance across a load-then-save so the serialized bytes are computed once.
- **Wire it in:** the manager wires the default for you. To supply a custom implementation, drive the [lower-level savers and loaders](./api_reference.md#lower-level-savers--loaders-advanced) directly (`make_async_checkpoint_saver(checkpoint_metadata_manager=...)`, `CheckpointLoader(metadata_manager=...)`) rather than through `CheckpointManager`.

## Serialization formats & layout — select the on-disk representation

Choose how (and to which files) items are written — e.g. safetensors for
tensors or JSON for human-readable metadata.

- **Supported formats:** `TorchSerialization` (default),
  `SafetensorsSerialization`, `JsonSerialization(cls=None)`, and
  `RawSerialization`.
- **Wire it in:** attach it per item via `ItemSpec(layout=LayoutInfo(...))` in `CheckpointManager.Config(items={...})`.
- **Current limit:** the writer dispatches only these four built-in formats;
  defining another `SerializationFormat` subclass does not add a custom
  encoding.
- **Deep dive:** [Configuring checkpoints](./configuring_checkpoints.md).

## Stagers — customize async host-side staging

The stager copies device state to host memory before the background write. Customize it to change buffer strategy or add instrumentation.

- **Interface:** `CheckpointStager` (`stage` / `close`) plus `CheckpointStagerConfig`.
- **Built-in:** `DefaultStager` (pinned / shared memory, non-blocking copy; pinned-memory support is currently CUDA-specific).
- **Wire it in:** the async preset builds a `DefaultStager` with its defaults. Customize those options by passing `AsyncCheckpointSaverConfig(staging_config=...)` as `CheckpointManager.Config.save`. To swap in a custom stager implementation, drive the lower-level `AsyncCheckpointSaver` directly.

## Barriers — custom cross-rank coordination

Barriers coordinate ranks around a save (e.g. so a rename to the final path happens only after every rank has written).

- **Interface:** `BarrierConfig` / `Barrier` — `create_barrier(rank_info)`, `execute_barrier(timeout_secs)`.
- **Built-in:** `DefaultStoreBarrier` / `DefaultStoreBarrierConfig` reuse the store that already backs the default process group, so there is no extra port to manage; `TCPStoreBarrier` / `TCPStoreBarrierConfig` stand up a `TCPStore` of their own on a port you supply.
- **A group of one is a special case in `DefaultStoreBarrier`:** it has nobody to wait for, so it passes without touching a store, and therefore without needing a process group at all — which is what lets single-process training use it without `init_process_group`. Only a group of more than one rank requires one.
- **Barriers run in the write subprocess.** A `BarrierConfig` is serialized into it, so anything the barrier needs from the parent process — such as the address of a store that only the parent can look up — has to be captured before it crosses that boundary, not on arrival. `DefaultStoreBarrierConfig` does this in `__getstate__`, recording the absence of a process group rather than failing there, since whether that matters depends on a rank count only `create_barrier` is given.

## Going deeper

[Design & internals](./design.md) walks the save and load pipelines end to end — the async staging path, the background write subprocess and its protocol, and how these components fit together. It's the reference when you're extending behavior that crosses component boundaries. `CheckpointManager` is the deep module that composes all of the above (stager, subprocess writer, reader, metadata manager) behind `save()` / `load()`; extend a single component through the hooks above, or drop to the [lower-level savers and loaders](./api_reference.md#lower-level-savers--loaders-advanced) when you need to rewire how they fit together.
