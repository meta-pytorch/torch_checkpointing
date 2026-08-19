# Key concepts

This guide explains the mental models behind `torch_checkpointing` so the how-to
guides make sense. Read it once, top to bottom; each section builds on the last.

For a runnable end-to-end example, see the
[training-loop tutorial](./tutorials.md). For depth on any one topic, follow the
how-to links at the end of each section.

## 1. The `CheckpointManager` and the async save pipeline

`CheckpointManager` is the entry point: one object drives both save and load. You
build it from a config preset, then call `save()` and `load()` on it.

```python
from torch_checkpointing import CheckpointManager

manager = CheckpointManager(CheckpointManager.Config.with_async_save())   # or .with_sync_save()
```

The default async staging options currently require CUDA. See
[Troubleshooting](./troubleshooting.md) for the explicit CPU configuration.

An asynchronous save runs in two phases, and understanding the split is the key
to reasoning about performance and correctness.

**Stage (host-side copy).** The manager first copies your state off the training
device into host memory. This is a fast, bounded operation. Once it completes,
your training loop is free to keep mutating the original tensors — the staged
copy is an independent snapshot.

**Write (background subprocess).** The staged copy is then serialized and written
to storage by a *separate subprocess*, concurrently with training. This is the
slow, I/O-bound part, and it stays off the training loop's critical path.

`manager.save()` returns immediately with the **write `Future`**:

```python
write_future = manager.save(checkpoint_id, checkpoint)
# ... training continues here — the write does not block the loop ...
write_future.result()   # wait only where you need the bytes durable in storage
```

Wait on `write_future` only where you need a durability guarantee (for example,
before exiting the process). The manager also serializes successive saves: the
next `save()` first waits on the previous write before staging again, so a slow
write naturally back-pressures the loop instead of overlapping two writes.

A sync manager (`CheckpointManager.Config.with_sync_save()`) runs the same logical
steps inline and `save()` returns `None` — no future, no subprocess. Use it when
you want the simplest possible behavior and don't mind blocking the loop.

## 2. Payloads and template-driven loads

You describe *what* to checkpoint with a plain mapping — no base class to
subclass, no wrapper objects:

```python
manager.save(checkpoint_id, {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
})
```

The payload type is `Mapping[str, Any]`. Each top-level key names one item and
becomes part of its on-disk file name. The checkpoint engine therefore requires
keys to match **`^[a-zA-Z0-9_-]+$`** — alphanumeric characters, hyphens, and
underscores only (no dots, slashes, whitespace, or extensions). A value can be
a nested state dict, a single tensor, or a plain leaf (an `int`, a config dict,
`bytes`) — leaves are first-class top-level items, not something you must bury
inside a sub-dict.

**Loading is template-driven.** You pass `into=` — a mapping of the *live*
objects to restore into — and the loader merges the stored data into them:

```python
restored = manager.load(checkpoint_id, into={
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": 0,
})
```

The `into=` mapping does double duty: it names which items to read, and its
values are the templates the loader restores into. The merge rules:

- **Tensors are copied in place** via `copy_()` into the target tensor,
  preserving the target's identity (same object, new data). Because
  `model.state_dict()` returns references to the live parameters, loading into it
  updates the model in place.
- **Mutable containers** (`dict`, `list`, `deque`) are updated in place;
  **immutable containers** (`tuple`) and non-tensor leaves are replaced.
- **A structured template also filters recursively:** at every nesting level,
  only mapping keys and sequence indices present in the template are loaded.

`load()` also **returns** the loaded mapping, so you can read scalars back
(`restored["step"]`) and re-apply object state where an object owns richer state
than plain tensors (`optimizer.load_state_dict(restored["optimizer"])`). Pass
`strict=True` to raise on missing keys, and `map_location=` to relocate tensors
onto a target device.

> With `into=None`, the manager reads the items declared in the config as-is —
> useful for inspecting a checkpoint. Resharding needs live targets, so it
> requires `into=`.

## 3. Per-item control: `ItemSpec`

For the common case you pass bare values and the manager picks sensible defaults:
each item is copied during staging, written as one per-rank `torch.save` file,
and read back directly. When you need to override that for a specific item, bind
an `ItemSpec` to its key in the config — declared **once**, not rebuilt every
save:

```python
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.checkpoint_layout import LayoutInfo, SafetensorsSerialization
from torch_checkpointing.default_resharder import DefaultResharder

manager = CheckpointManager(CheckpointManager.Config(
    items={
        "model": ItemSpec(
            resharder=DefaultResharder(),                     # reshard on load
            layout=LayoutInfo("model_{rank}.safetensors",     # custom on-disk layout
                              SafetensorsSerialization()),
        ),
        "dataloader": ItemSpec(requires_copy=False),          # frozen snapshot: skip the staging copy
    },
))
```

An `ItemSpec` carries the per-item "how":

- **`requires_copy`** — whether staging deep-copies the value and moves device
  tensors to host memory. Keep the default (`True`) for state the training loop
  keeps mutating, so the background write can't race the loop. Set it `False`
  only for stable, already host-resident data that will not change through the
  write. Values sent to an async writer must be pickleable in either case.
- **`layout`** — a `LayoutInfo` (`file_path` + serialization format) for where and
  how the item is written. `None` uses the default per-rank `torch.save` layout. A
  `{rank}` placeholder in `file_path` is filled with the current rank.
- **`resharder`** — an optional `Resharder` for redistributing the item across a
  different distributed layout on load (see §5).
- **`required`** — whether the item must be present (fail fast if missing).

`Config.items` maps keys to their specs; `Config.default` is the spec applied to
any key **not** listed there (set `default=None` to make the config strict, so an
un-listed key raises instead of using defaults). You never touch the schema for
the common case — reach for `items` only to override a specific item.

> How-to: [Configuring checkpoints](./configuring_checkpoints.md).

## 4. Distributed: rank is auto-detected

Every save and load needs to know where the process sits in the job. The manager
**auto-detects** this from the default PyTorch process group: if
`torch.distributed` is initialized it reads `get_rank()` / `get_world_size()`;
otherwise it falls back to single-rank. The same code runs unchanged from a
notebook to a large distributed job — you don't construct or pass rank
information for the common case.

Under the hood this is a `RankInfo` (`global_rank`/`global_world_size` locate the
process in the whole job; `role_rank`/`role_world_size` within its role). You only
deal with it directly for multi-role topologies or when driving the low-level
savers by hand.

> How-to: [Distributed checkpointing and resharding](./distributed_and_resharding.md).

## 5. Resharding on load

A distributed checkpoint is *sharded*: each rank writes the slice of state it
owns under a particular layout (world size, device mesh, placement).
**Resharding** is loading a checkpoint whose saved layout differs from the one
you're loading into — redistributing the stored shards so each target rank ends
up with the slice it now owns. You need it whenever you resume on a different
world size, device mesh, or parallelism strategy.

You opt in per item by giving it a `resharder` in the config; the built-in
`DefaultResharder` handles `DTensor` state, and you can write your own for other
sharded types.

```python
manager = CheckpointManager(CheckpointManager.Config(
    items={"model": ItemSpec(resharder=DefaultResharder())},
))
manager.save(path, {"model": model.state_dict()})     # records sharding metadata
manager.load(path, into={"model": model.state_dict()})  # reshards onto the new layout
```

That is the whole contract: **declare a resharder and the manager does the
rest.** It automatically records the source sharding metadata on save and wires
the target metadata on load — the two-sided coordination that, with the
lower-level API, you had to set up by hand in three places. An item with no
resharder is read directly from the current rank's configured layout, with no
redistribution or metadata overhead; this does not imply that the item is
replicated.

> How-to: [Distributed checkpointing and resharding](./distributed_and_resharding.md).

## 6. The storage abstraction

All I/O goes through a small two-type abstraction so the same checkpoint code can
target a local disk, a network filesystem, or an object store.

- **`StorageConfig`** is an abstract dataclass describing *which* backend and
  *how* it's configured; its `create_storage() -> Storage` builds the live backend.
- **`Storage`** is the live backend — the primitive read/write/rename/list
  operations the library needs.

Pass a `StorageConfig` via `CheckpointManager.Config(storage_config=...)`; the
library calls `create_storage()` internally. A local filesystem backend ships in
the package and is the default:

```python
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig

manager = CheckpointManager(CheckpointManager.Config(
    storage_config=LocalFileSystemStorageConfig(),
))
```

To target a different backend, implement a `StorageConfig` / `Storage` pair and
pass your config. When none is supplied, the manager defaults to
`LocalFileSystemStorageConfig()`.

> How-to: [Storage](./storage.md).

## Where to go next

- [Tutorial](./tutorials.md) — a runnable checkpoint-and-resume training loop.
- [Configuring checkpoints](./configuring_checkpoints.md) — `ItemSpec` and the
  `layout` / `requires_copy` / `resharder` options in depth.
- [Distributed checkpointing and resharding](./distributed_and_resharding.md) —
  distributed coordination, sharding metadata, and writing a `Resharder`.
