# Key concepts

This guide explains the mental models behind `torch_checkpointing` so the how-to
guides make sense. Read it once, top to bottom; each section builds on the last.

For a runnable end-to-end example, see the [README](../README.md). For depth on
any one topic, follow the how-to links at the end of each section.

## 1. The `CheckpointManager` and the async save pipeline

`CheckpointManager` is the entry point: one object drives both save and load. You
build it from a config preset, then call `save()` and `load()` on it.

```python
from torch_checkpointing import CheckpointManager

manager = CheckpointManager.Config.async_save().build()   # or .sync_save()
```

An asynchronous save runs in two phases, and understanding the split is the key
to reasoning about performance and correctness.

**Stage (host-side copy).** The saver first copies your state dictionary off the
training device into host memory. This is a fast, bounded operation. Once it
completes, your training loop is free to keep mutating the original tensors —
the staged copy is an independent snapshot.

**Write (background subprocess).** The staged copy is then serialized and written
to storage by a *separate subprocess*, concurrently with training. This is the
slow, I/O-bound part, and it stays off the training loop's critical path.

`manager.save()` returns immediately with a tuple of two futures:

```python
stage_future, write_future = manager.save(checkpoint_id, checkpoint)
# ... training continues here — neither phase blocks the loop ...
stage_future.result()   # host-side staging complete; safe to mutate originals
write_future.result()   # bytes are durable in storage
```

The **`(stage_future, write_future)` contract** tells you exactly what has
finished:

- `stage_future` resolves once staging is done. After this point the source
  tensors are yours again — the training loop can overwrite them freely.
- `write_future` resolves once the background write has committed the checkpoint
  to storage. Wait on this only where you need durability guarantees (for
  example, before exiting the process).

The manager also serializes successive saves: the next `save()` call first waits
on the previous `write_future` before staging again, so a slow write naturally
back-pressures the loop instead of overlapping two writes.

A sync manager (`CheckpointManager.Config.sync_save()`) runs the same logical
steps inline and `save()` returns `None` — no futures, no subprocess. Use it when
you want the simplest possible behavior and don't mind blocking the loop.

> How-to: [Configuring checkpoints](./configuring_checkpoints.md).

## 2. The `CheckpointBase` / `CheckpointItem` contract

You describe *what* to checkpoint by subclassing `CheckpointBase` and
implementing two methods:

```python
class CheckpointBase(abc.ABC):
    @abc.abstractmethod
    def get_items(self) -> dict[str, CheckpointItem]: ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: STATE_DICT) -> None: ...
```

A representative implementation checkpoints the state a training job usually
carries — model, optimizer, dataloader, and some bookkeeping:

```python
from torch_checkpointing.dtensor_resharder import DTensorResharder


class TrainerCheckpoint(CheckpointBase):
    def __init__(self, model, optimizer, dataloader, step):
        self._model = model
        self._optimizer = optimizer
        self._dataloader = dataloader
        self._step = step

    def get_items(self) -> dict[str, CheckpointItem]:
        return {
            # DTensor-sharded tensors: copy during staging, reshard on load.
            "model": CheckpointItem(
                value=self._model.state_dict(),
                requires_copy=True,
                resharder=DTensorResharder(),
            ),
            "optimizer": CheckpointItem(
                value=self._optimizer.state_dict(),
                requires_copy=True,
                resharder=DTensorResharder(),
            ),
            # Dataloader progress: a snapshot (no staging copy) with a custom
            # resharder (see Configuring checkpoints for MyCustomDataloaderResharder).
            "dataloader": CheckpointItem(
                value=self._dataloader.state_dict(),
                requires_copy=False,
                resharder=MyCustomDataloaderResharder(),
            ),
            # Small replicated bookkeeping (step, config, RNG): no copy, no reshard.
            "misc_state": CheckpointItem(
                value={"step": self._step},
                requires_copy=False,
            ),
        }

    def load_state_dict(self, state_dict: STATE_DICT) -> None:
        self._model.load_state_dict(state_dict["model"])
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._step = state_dict["misc_state"]["step"]
```

**`get_items()`** returns a dict of `CheckpointItem` objects keyed by a string
identifier. Each key names one piece of state (for example `"model"`,
`"optimizer"`, `"step"`). Because keys become filename components, they must
match the rule **`^[a-zA-Z0-9_-]+$`** — alphanumeric characters, hyphens, and
underscores only. No dots, slashes, whitespace, or extensions. Keys must be
unique. Invalid keys raise `ValueError` when the checkpoint is saved or loaded.

`get_items()` may be specialized per rank — for example, rank 0 can add global
metadata that other ranks omit, or a rank can pick a rank-specific file name via
`layout`.

A `CheckpointItem` carries four fields:

```python
@dataclass
class CheckpointItem:
    value: Any = None
    requires_copy: bool = True
    layout: LayoutInfo | None = None
    resharder: Resharder | None = None
```

- **`value`** — the data to save. On *write*, this is the actual object (a module
  state dict, a tensor, an int). On *read*, it acts as a template (see section 3).
- **`requires_copy`** — whether the value must be copied during async staging.
  Keep this `True` for state the training loop keeps mutating, so the background
  write doesn't race the loop. Set it `False` for snapshots that won't change,
  to skip the staging copy.
- **`layout`** — where and how the item is written. `None` means the default:
  one `torch.save` file named after the key plus the rank id.
- **`resharder`** — an optional `Resharder` for redistributing data across a
  different parallelization layout on load (see sections 3 and 6).

**`load_state_dict(state_dict)`** is the mirror image. After a checkpoint is read
from storage, the loader calls this method with the loaded state so you can push
it back into your already-initialized model, optimizer, and other components:

```python
def load_state_dict(self, state_dict: STATE_DICT) -> None:
    self._model.load_state_dict(state_dict["model"])
    self._optimizer.load_state_dict(state_dict["optimizer"])
```

> How-to: [Configuring checkpoints](./configuring_checkpoints.md).

## 3. Template-driven loads

Loading is *template-driven*: the `value` you supply in each `CheckpointItem`
during a load describes the shape you want back, and the loader merges the stored
data into it. This is what lets loads preserve object identity instead of
allocating fresh objects.

The rules the loader (`CheckpointReader` / `CheckpointLoader`) applies:

- **Tensors are copied in place** via `copy_()` into the target tensor,
  preserving the target tensor's identity (same object, new data). This matters
  when other code already holds references to those tensors.
- **Mutable containers** (`dict`, `list`, `deque`) are updated in place.
- **Immutable containers** (`tuple`) and non-tensor leaf values are replaced —
  new objects are created.
- **When a `value` is `None`**, no template is provided, so the loader builds new
  objects from the stored data and returns everything for that item unfiltered.
- **When a `value` is a structure**, it also acts as a *filter*: only keys and
  indices present in the template are loaded.

You drive loads through the same `CheckpointManager`:

```python
manager.load(checkpoint_id, checkpoint)   # merges stored data into your checkpoint
```

`load()` accepts `strict=False` by default; pass `strict=True` to raise on
missing keys, and `map_location=` to relocate tensors onto a target device.

Under the hood two classes do the work — the manager wires them together, and you
can also use them directly for load-only workloads (eval, inference):

- **`CheckpointReader`** reads bytes from storage and produces the loaded state
  dict, honoring the template merge rules above. It takes a `RankInfo` and a
  `StorageConfig`, and `read()` returns `(state_dict, missing_keys)`.
- **`CheckpointLoader`** wraps a `CheckpointReader` for the common load flow: it
  calls `read()`, then feeds the result to your `checkpoint.load_state_dict()`.
  It is deliberately lightweight — no subprocess, staging, or barriers — which is
  why the manager can build one eagerly, and why it also suits eval- or
  inference-only use on its own:

```python
reader = CheckpointReader(rank_info=rank_info, storage_config=storage_config)
loader = CheckpointLoader(reader=reader)
loader.load(path, checkpoint)   # calls checkpoint.load_state_dict() for you
loader.close()
```

> How-to: [Configuring checkpoints](./configuring_checkpoints.md).

## 4. `RankInfo`: single-rank and distributed

Every saver, reader, and loader needs to know where it sits in the world.
`RankInfo` carries that:

```python
@dataclass
class RankInfo:
    global_rank: int
    global_world_size: int
    role_rank: int
    role_world_size: int
```

`global_rank` / `global_world_size` locate the process in the whole job;
`role_rank` / `role_world_size` locate it within its role (trainer, evaluator,
and so on).

`CheckpointManager` and the factory functions (`make_async_checkpoint_saver`,
`make_sync_checkpoint_saver`) **auto-detect** `RankInfo` when you don't pass one:

- If `torch.distributed` is initialized, they read the default process group —
  `global_rank`/`role_rank` from `get_rank()`, world sizes from
  `get_world_size()`.
- Otherwise they fall back to single-rank:
  `RankInfo(global_world_size=1, global_rank=0, role_rank=0, role_world_size=1)`.

For single-rank use (a notebook, a small eval job) you can construct it
explicitly and skip process-group setup entirely:

```python
rank_info = RankInfo(
    global_world_size=1, global_rank=0, role_rank=0, role_world_size=1
)
```

> How-to: [Distributed checkpointing and resharding](./distributed_and_resharding.md).

## 5. The storage abstraction

All I/O goes through a small two-type abstraction so the same checkpoint code can
target a local disk, a network filesystem, or an object store.

- **`StorageConfig`** is an abstract dataclass describing *which* backend and
  *how* it's configured. Its one abstract method, `create_storage() -> Storage`,
  builds the live backend.
- **`Storage`** is the live backend — an abstract class exposing the primitive
  operations the reader and writer need: `stream_read`, `stream_write`, `read`,
  `write`, `delete`, `mkdir`, `rmdir`, `rename`, `ls`, `exists`, `getsize`,
  `glob`, `isdir`, and `remap_path`.

You pass a `StorageConfig` to the `CheckpointManager` (via
`Config.storage_config`), the factories, savers, and `CheckpointReader`; the
library calls `create_storage()` internally. A local filesystem backend
ships in the package:

```python
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig

storage_config = LocalFileSystemStorageConfig()
```

To target a different backend, implement a `StorageConfig` / `Storage` pair and
pass your config wherever a `storage_config` is accepted. When no config is
supplied, the factories default to `LocalFileSystemStorageConfig()`.

## 6. What "resharding" means

A distributed checkpoint is *sharded*: each rank writes the slice of state it
owns, under a particular parallelization layout (data-parallel, tensor-parallel,
a specific device mesh and placement, a specific world size).

**Resharding** is loading a checkpoint whose saved layout differs from the layout
you're loading into — reassembling and redistributing the stored shards so each
target rank ends up with the slice *it* now owns. You need it whenever you resume
on a different world size, a different device mesh, or a different parallelism
strategy than you saved with.

The extensibility point is the abstract `Resharder`, attached per item via
`CheckpointItem.resharder`. Its `should_reshard(...)` compares source and target
sharding metadata to decide whether redistribution is needed for that item, and
its `load(...)` performs the redistributed read into the target. A built-in
`DTensorResharder` handles `DTensor` state; for state that isn't a plain
`DTensor` you attach your own `Resharder` subclass — see
[Extensibility](./extensibility.md) for the full list of extension points.

Attaching a resharder is necessary but not sufficient: the loader must also be
given a `MetadataManager` (for example `DefaultMetadataManager`), or the read
silently falls back to a direct, non-resharded load. See the
[distributed guide](./distributed_and_resharding.md) for the full save-and-load
wiring.

Items with no resharder are read back directly with no metadata overhead — the
common case for single-rank or unchanged-layout resumes. When `should_reshard`
determines the layouts match (or a resharder sets `skip_resharding = True`), the
reader also takes this fast, direct path.

> How-to: [Distributed checkpointing and resharding](./distributed_and_resharding.md).

## Where to go next

- [README](../README.md) — installation and a runnable quick start.
- [Configuring checkpoints](./configuring_checkpoints.md) — the `CheckpointItem`
  fields (`value`, `requires_copy`, `layout`, `resharder`) in depth.
- [Distributed checkpointing and resharding](./distributed_and_resharding.md) —
  `RankInfo`, distributed coordination, sharding metadata, and writing or
  attaching a `Resharder`.
