# Distributed Checkpointing and Resharding

This guide covers the distributed side of `torch_checkpointing`: how ranks
identify themselves, how they coordinate at save time, how sharding metadata is
collected and shared between save and load, and how a checkpoint saved
under one device mesh / placement can be loaded under a different one
(resharding).

See [./key_concepts.md](./key_concepts.md) for the core building blocks
(the `CheckpointManager`, the payload / `into=` model) and
[./tutorials.md](./tutorials.md) for a complete training loop.

> **Two tracks.** The common case at the top uses the high-level
> `CheckpointManager` — you declare a resharder per item and the manager wires
> the sharding-metadata pipeline on both save and load for you. The numbered
> reference sections below document the mechanisms it wires: rank identity,
> barriers, the metadata pipeline, and the resharding internals. They double as
> the guide for driving the low-level savers and loaders directly, and for
> writing a custom `Resharder`. See [Extensibility](./extensibility.md) for the
> full set of extension points.

> **Status: advanced / less battle-tested.** Single-rank and simple data-parallel
> saves and loads are the well-trodden path. The multi-rank metadata aggregation
> and DTensor resharding paths described below are newer and exercised by fewer
> workloads. Treat them as advanced features: validate end-to-end on your own
> configuration before relying on them, and expect rougher edges than the
> single-rank flow.

---

## The common case: save distributed, reshard on load

Most users come here for one task: *I saved a checkpoint on one set of ranks (or one device mesh) and want to load it onto a different one.* With the high-level `CheckpointManager` that is one declaration and two calls — you **declare a resharder per item** in the config, and the manager auto-wires the sharding-metadata pipeline on **both save and load**. Rank identity is auto-detected from the process group (see §1), and you pass plain `{name: value}` dicts. It runs under `torchrun` and assumes your model state is made of `DTensor`s, handled by the built-in `DefaultResharder`.

**1. Declare which items to reshard** — bind an `ItemSpec` with a `DefaultResharder` to each item, once, in the config:

```python
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.default_resharder import DefaultResharder

manager = CheckpointManager(CheckpointManager.Config(
    items={"model": ItemSpec(resharder=DefaultResharder())},
))
```

`Config.items` maps each key to its `ItemSpec`; use the `Config.with_async_save()` / `Config.with_sync_save()` presets to pick the async or sync pipeline. (For a multi-rank save to shared storage the manager coordinates the ranks with a barrier so all shards are present before the checkpoint is published — see §2.)

**2. Save on the original ranks** — the manager records the source sharding layout next to the shards (resharding on load depends on it):

```python
write_future = manager.save(path, {"model": model.state_dict()})
write_future.result()  # block only where you need the bytes durable
```

**3. Load on the new ranks / mesh** — pass the live model, built under the *new* layout, as the `into=` target; the resharder redistributes each stored shard onto the rank that now owns it:

```python
manager.load(path, into={"model": model.state_dict()})  # `model` is already built under the new layout
```

That is the whole contract: **declare a resharder on each item and the manager does the rest.** It records the source sharding metadata on save and wires the target metadata on load — the two-sided coordination automatically, on both ends. There is nothing to omit and no fast-path footgun to trip: the manager wires both sides for you, so a checkpoint with a declared resharder always reshards when the layout differs and always takes a plain direct read when it doesn't.

The reference sections below document the mechanisms the manager wires — rank identity, save-time coordination, the metadata pipeline, the resharding internals, and writing a resharder for non-DTensor state. Read them to understand what happens under the hood, or as the guide for driving the low-level savers and loaders directly.

---

## 1. Rank identity: `RankInfo`

`CheckpointManager` auto-detects rank identity from the default process group, so
most users never construct one of these. This section documents what it detects
(and what you would pass explicitly when driving the low-level savers directly or
running a multi-role topology).

Every distributed component is parameterized by a `RankInfo`
(`torch_checkpointing/types.py`), a plain dataclass:

```python
@dataclass
class RankInfo:
    global_rank: int
    global_world_size: int
    role_rank: int
    role_world_size: int
```

There are two distinct notions of rank:

- **Global rank / world size** — the rank's position across the entire job.
  Metadata aggregation and DTensor resharding key off the *global* rank.
- **Role rank / world size** — the rank's position within its role. Barrier
  coordination and metadata-file writing use the *role* rank (for example, only
  `role_rank == 0` writes the `metadata.pkl` file and hosts the barrier's
  TCPStore server).

In a homogeneous single-role job the two coincide. They diverge when a job runs
multiple roles (for example, trainer ranks alongside separate policy-model
ranks) and each role checkpoints independently.

### Auto-detection from the default process group

You rarely construct `RankInfo` by hand. The saver factories in
`torch_checkpointing/builder.py` fall back to `_get_default_rank_info()`, which
reads the default PyTorch process group:

```python
def _get_default_rank_info() -> RankInfo:
    if dist.is_initialized():
        return RankInfo(
            global_world_size=dist.get_world_size(),
            global_rank=dist.get_rank(),
            role_rank=dist.get_rank(),
            role_world_size=dist.get_world_size(),
        )
    else:
        return RankInfo(
            global_world_size=1,
            global_rank=0,
            role_rank=0,
            role_world_size=1,
        )
```

So if `torch.distributed` is initialized, both `make_sync_checkpoint_saver` and
`make_async_checkpoint_saver` pick up global rank and world size automatically,
and set role rank/world size equal to the global values. If it is not
initialized, you get the single-rank fallback. Pass an explicit `rank_info=` to
either factory when your role topology differs from the global one.

---

## 2. Cross-rank coordination: barriers

At save time, all ranks write their shards independently, then coordinate before
the checkpoint is renamed into its final location. That coordination is a
**barrier**, defined in `torch_checkpointing/barriers.py`.

### `BarrierConfig` and `TCPStoreBarrierConfig`

`BarrierConfig` is an abstract dataclass; concrete configs create concrete
barriers:

```python
@dataclass
class BarrierConfig(abc.ABC):
    timeout_barrier_init_sec: int

    @abc.abstractmethod
    def create_barrier(self, rank_info: RankInfo) -> "Barrier": ...


@dataclass
class TCPStoreBarrierConfig(BarrierConfig):
    use_checkpoint_barrier_tcpstore_libuv: bool
    tcpstore_port: int
    master_address: str

    def create_barrier(self, rank_info: RankInfo) -> "TCPStoreBarrier":
        return TCPStoreBarrier(config=self, rank_info=rank_info)
```

`TCPStoreBarrier` uses a PyTorch `TCPStore` for synchronization. The rank with
`role_rank == 0` is the master and eagerly starts the store server; other ranks
connect on first use. Each `execute_barrier(timeout_secs)` call sets the rank's
current sequence number in the store and waits (via
`torch.distributed.elastic.utils.store.barrier`) for all `role_world_size` ranks
to reach the same sequence number, then increments the sequence counter. Because
it uses a dedicated `TCPStore` rather than the collective process group,
checkpoint coordination stays independent of ongoing collective communication.

### Where the barrier fits in a save

The barrier is configured on `CheckpointWriterConfig`
(`torch_checkpointing/checkpoint_writer.py`):

```python
@dataclass
class CheckpointWriterConfig:
    checkpoint_write_barrier_timeout_sec: int = 600
    barrier_config: BarrierConfig | None = None
    file_write_max_threads: int = 1
```

Inside `CheckpointWriter.write()` the ordering is:

1. Write all shard files (and, on `role_rank == 0`, the metadata file) into a
   temporary directory.
2. If a barrier is configured, `execute_barrier(...)` — wait for every rank to
   finish writing.
3. Run `pre_finalize_callback` (see below) against the complete temporary
   checkpoint.
4. On `role_rank == 0`, atomically rename the temporary directory to the final
   path.
5. Run `finalize_callback`.

The temporary-directory-then-rename dance is what makes a checkpoint atomic:
readers never observe a half-written directory. Note that the writer only uses a
temp directory *when a barrier is configured* — without a barrier, files are
written directly to the final path and no rename occurs.

### Disabling the barrier

Set `barrier_config=None` (the default). With no barrier, the writer skips
synchronization and the rename step, writing directly to the final path. This is
correct for single-rank jobs and for storage backends that provide their own
coordination, but for multi-rank saves to shared storage a barrier is what
guarantees all shards are present before the checkpoint is published.

```python
from torch_checkpointing.config import SyncCheckpointSaverConfig
from torch_checkpointing.checkpoint_writer import CheckpointWriterConfig

# No barrier (single-rank or externally coordinated)
config = SyncCheckpointSaverConfig(
    writer_config=CheckpointWriterConfig(barrier_config=None),
)
```

To enable TCPStore-based coordination, supply a `TCPStoreBarrierConfig`:

```python
from torch_checkpointing.barriers import TCPStoreBarrierConfig

config = SyncCheckpointSaverConfig(
    writer_config=CheckpointWriterConfig(
        barrier_config=TCPStoreBarrierConfig(
            timeout_barrier_init_sec=300,
            use_checkpoint_barrier_tcpstore_libuv=True,
            tcpstore_port=29500,
            master_address="<rank-0 host>",
        ),
    ),
)
```

---

## 3. Finalize callbacks

The two finalize hooks let you run custom logic around the atomic-rename step.
They are passed to the saver **factory functions**, not stored on the
`CheckpointSaverConfig` objects, and flow through to `CheckpointWriterArgs`:

```python
@dataclass
class CheckpointWriterArgs:
    config: CheckpointWriterConfig
    rank_info: RankInfo
    storage_config: StorageConfig
    pre_finalize_callback: Callable[[str, EventLogger], None] | None = None
    finalize_callback: Callable[[str, EventLogger], None] | None = None
    metric_prefix: str = "train.checkpoint_write"
```

Both callbacks have the signature `(path: str, event_logger: EventLogger) -> None`:

- **`pre_finalize_callback`** runs after all files are written and the first
  barrier has completed, but before the atomic rename. It is invoked on every
  rank with the temporary checkpoint path. A distributed callback must ensure
  that rank zero does not return until work required before commit is complete.
  Without a configured barrier, it receives the final write path and no rename
  occurs.
- **`finalize_callback`** runs *after* the barrier and rename — it is invoked
  with the final path.

Pass them to either factory:

```python
from torch_checkpointing.builder import make_async_checkpoint_saver

saver = make_async_checkpoint_saver(
    pre_finalize_callback=lambda path, event_logger: validate_files(path),
    finalize_callback=lambda path, event_logger: logger.info(f"Checkpoint done: {path}"),
)
```

The same parameters exist on `make_sync_checkpoint_saver`. For asynchronous
saves the callbacks execute in the checkpoint subprocess, so they must be
picklable — prefer module-level functions or `functools.partial` over lambdas
and closures.

---

## 4. Sharding metadata: `MetadataManager`

To reshard a checkpoint on load, the loader needs to know how the *source*
checkpoint was sharded. That description is **sharding metadata**, produced and
aggregated by a `MetadataManager` (`torch_checkpointing/metadata_manager.py`).

`CheckpointManager` constructs and wires a `MetadataManager` on both save and
load whenever any item declares a resharder — you do not build one yourself. This
section describes the pipeline the manager runs (and what you would wire by hand
if driving the low-level savers and loaders directly).

### The abstract interface

```python
class MetadataManager(ABC):
    @abstractmethod
    def compute_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> CheckpointMetadata | None: ...

    @abstractmethod
    def extract_object_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> dict[str, dict[NestedPath, ShardingMetadata]]: ...

    @abstractmethod
    def close(self) -> None: ...
```

- `extract_object_metadata` walks the checkpoint items and, for each item that
  has a resharder attached, calls that resharder's `extract_sharding_metadata`
  to get per-path `ShardingMetadata`. Items without a resharder are skipped
  entirely.
- `compute_metadata` orchestrates the full pipeline: local extraction, cross-rank
  aggregation, and construction of the global view. It returns `None` when the
  result is unchanged from a cached previous computation (see caching below).
- `close` releases resources (a background serialization thread pool).

### `DefaultMetadataManager`

`DefaultMetadataManager` is the batteries-included implementation:

```python
def __init__(
    self,
    rank_info: RankInfo,
    process_group: ProcessGroup | None = None,
    should_cache_metadata: bool = True,
    enable_serialization: bool = True,
):
```

- `process_group=None` uses the default process group for its collectives.
- `should_cache_metadata=True` caches the extracted local metadata and, on
  subsequent `compute_metadata` calls, returns `None` if the extracted metadata
  is unchanged. If the local metadata *has* changed it raises `RuntimeError`
  ("State dictionary has changed since last checkpoint"), because a changed
  state dict invalidates the cached global view. Only enable caching when your
  state dict's structure and sharding are stable across checkpoints.
- `enable_serialization=True` kicks off async serialization of the aggregated
  metadata (in a single-worker `ThreadPoolExecutor`) after the first compute, so
  the bytes are ready to write without blocking the saver. Set it to `False` for
  load-only scenarios where no save will ever happen.

What it computes:

1. **Extract** local `ShardingMetadata` per item/path via
   `extract_object_metadata`.
2. **Aggregate** across ranks. For a single rank (`global_world_size == 1`, or
   when `torch.distributed` is not initialized) it builds the global view
   directly. For multi-rank it *compacts* metadata (for sharding types that
   report `equivalent_ranks`, only the representative rank — the minimum of the
   equivalent set — sends its entry, shrinking the collective payload at scale),
   `all_gather`s the pickled per-rank metadata, and merges entries that share
   the same `ShardingMetadata` into `GlobalObjectMetadata` groups.
3. **Validate** implicitly: the resulting `DistributedMetadata.__post_init__`
   checks that every rank appears in each item's layout info and that no two
   ranks map to the same file path.

The return value is a `CheckpointMetadata`
(`torch_checkpointing/distributed_metadata.py`) that bundles both views:

```python
@dataclass
class CheckpointMetadata:
    distributed_metadata: DistributedMetadata
    local_metadata: dict[str, dict[NestedPath, ShardingMetadata]]
```

`DistributedMetadata.metadata` maps each item key to a
`DistributedItemMetadata`, which stores `nested_path_to_metadata`
(path → list of `GlobalObjectMetadata`, each carrying the `ShardingMetadata` and
the tuple of `ranks` that share it) plus `rank_to_layout_info` for locating each
rank's file. `DistributedItemMetadata.get_file_path(rank, checkpoint_path,
item_key)` is what the resharder uses to find a source rank's shard file.

### Sharing a manager between saver and loader

The metadata computed on load can be reused by a subsequent save. Construct a
single `MetadataManager`, hand it to the `CheckpointLoader`, then hand the same
instance to `make_async_checkpoint_saver` via `checkpoint_metadata_manager`. The
loader triggers the compute (and its async serialization), and the saver reuses
the already-serialized bytes rather than recomputing:

```python
from torch_checkpointing.builder import make_async_checkpoint_saver
from torch_checkpointing.checkpoint_loader import CheckpointLoader
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.metadata_manager import DefaultMetadataManager

metadata_manager = DefaultMetadataManager(rank_info=rank_info)

reader = CheckpointReader(rank_info=rank_info, storage_config=storage_config)
loader = CheckpointLoader(reader=reader, metadata_manager=metadata_manager)
loader.load(path, checkpoint)
loader.close()

saver = make_async_checkpoint_saver(checkpoint_metadata_manager=metadata_manager)
saver.save(new_path, checkpoint)  # reuses serialized metadata from the load
```

---

## 5. Resharding: loading under a different mesh/placement

Resharding is loading a checkpoint whose shards were laid out for one device mesh
/ placement into tensors laid out for a *different* one — for example changing
the tensor-parallel degree, or moving from pure data parallelism to a 2-D mesh.

With `CheckpointManager` you opt in by declaring a resharder on the item's
`ItemSpec` (§"common case"); the manager then wires both halves of the mechanism
this section describes — the resharder on each item, and a `MetadataManager` on
save and load. What follows is what that wiring drives under the hood (and what
you assemble by hand when driving the low-level loader directly).

> **The direct-read fast path.** Resharding only engages when there is a
> resharder *and* target sharding metadata. Without both, `CheckpointReader.read`
> takes a direct-read fast path: each rank reads back exactly the file its layout
> points at, with no metadata loading and no shard remapping — correct precisely
> when the load-time layout matches the save-time layout. Through
> `CheckpointManager` this is an internal optimization, not something to
> configure: because the manager wires a resharder-declared item on both ends, it
> reshards when the layout differs and takes the direct read when it doesn't. The
> two ingredients below — (a) a resharder per item and (b) a `MetadataManager` on
> the loader — are exactly what the manager supplies; you assemble them yourself
> only when driving the low-level loader directly.

### 5a. Attach a resharder per item

Through `CheckpointManager` a resharder is declared on the item's `ItemSpec`
(see [./key_concepts.md](./key_concepts.md)); the low-level path carries it on the
`CheckpointItem`. During `extract_object_metadata`, only items whose `resharder is
not None` contribute sharding metadata:

```python
for item_key, checkpoint_item in checkpoint_info.checkpoint_items.items():
    if checkpoint_item.resharder is None:
        continue
    nested_path_to_metadata = checkpoint_item.resharder.extract_sharding_metadata(
        item_key, checkpoint_item.value
    )
```

For DTensor state dicts the built-in `DefaultResharder`
(`torch_checkpointing/default_resharder.py`) handles this.

### 5b. Wire a metadata manager into the loader

`CheckpointManager` does this step for you when an item declares a resharder. The
mechanism below is what it wires — and what you wire by hand when driving
`CheckpointLoader` directly.

`CheckpointLoader` takes an optional `MetadataManager`:

```python
class CheckpointLoader:
    def __init__(
        self,
        reader: CheckpointReader,
        metadata_manager: MetadataManager | None = None,
    ) -> None:
        self._reader = reader
        self._metadata_manager = metadata_manager
```

If `metadata_manager is None`, `_compute_metadata_once` returns `None`, so the
`CheckpointReadInfo` carries no target metadata. Even if items happen to have
resharders, `Resharder.should_reshard` then sees `target_metadata=None` and
returns `False`, so every item falls back to the direct-read path. Only when a
manager is present does the loader compute the *target* sharding metadata, pass
it through to the reader, and let the resharders run. (Independently, if *no*
item has a resharder — or every resharder sets `skip_resharding=True` — the
reader short-circuits to direct reads without even loading the source metadata
file.) For a load-only setup, construct the manager with
`enable_serialization=False` to skip the serialization overhead you would only
need for a save.

```python
from torch_checkpointing.checkpoint_loader import CheckpointLoader
from torch_checkpointing.metadata_manager import DefaultMetadataManager

metadata_manager = DefaultMetadataManager(
    rank_info=rank_info,
    enable_serialization=False,  # load-only
)
loader = CheckpointLoader(reader=reader, metadata_manager=metadata_manager)
loader.load(path, checkpoint)  # resharding runs for items with a resharder
loader.close()
```

### 5c. The built-in `DefaultResharder`

`DefaultResharder` uses DTensor's native placement APIs to compute shard geometry
and remap data. Its `extract_sharding_metadata` walks the item and produces a
`DTensorShardingMetadata` (from `torch_checkpointing/dtensor_metadata.py`) for
every `DTensor` it finds:

```python
result: dict[NestedPath, ShardingMetadata] = {}

def _collect(path: CheckpointPath, obj: Any, _: Any) -> None:
    if isinstance(obj, DTensor):
        result[path.nested_path] = DTensorShardingMetadata.from_dtensor(obj)
```

`DTensorShardingMetadata` records `global_shape`, `dtype`, `stride`, a
`mesh_spec` (a `DeviceMeshSpec` with device type, mesh shape, the flattened
global rank IDs, and optional dim names), and a tuple of `placements`
(`ShardSpec(dim=...)`, `StridedShardSpec(dim=..., split_factor=...)`, or
`ReplicateSpec()`). Its `equivalent_ranks` returns the mesh's flattened rank IDs
(`mesh_spec.mesh_data`), which is what enables the metadata compaction described
above. Resharding decomposes a strided placement into contiguous local-to-global
slices and emits one `LoadPlan` for each overlapping source and target region.
This preserves disjoint layouts such as a packed local tensor whose elements map
to nonadjacent global indices.

Its `load` computes, for the current rank's target shard, which source ranks'
shards overlap it, generates `LoadPlan`s for each overlapping chunk, reads the
relevant source files, and copies the intersecting slices into the target
tensor:

```python
def load(
    self,
    source_path: Path,
    item_key: str,
    target_metadata: dict[NestedPath, ShardingMetadata],
    source_metadata: DistributedItemMetadata,
    target: Any,
    storage: Storage,
) -> list[NestedPath]:
```

It returns the list of `NestedPath`s that could not be resharded (for example a
target path whose source metadata is missing, or a non-DTensor target).

### The direct-read fast path vs. `should_reshard`

Even with a resharder and a metadata manager wired up, resharding only *runs*
where it is actually needed. `Resharder.should_reshard(source_metadata,
target_metadata)` decides per item: the default returns `False` when either side
is missing metadata, and otherwise returns `True` only if some path's target
sharding does not match any source rank group. Separately,
`Resharder.skip_resharding` (default `False`) lets a resharder short-circuit
metadata loading entirely — use it for job-retry scenarios where the mesh is
provably identical between save and load, to avoid the metadata-loading overhead.

---

## 6. Writing a custom `Resharder`

To support a sharding scheme beyond DTensor, subclass `Resharder`
(`torch_checkpointing/resharding.py`). Two methods are abstract; two are
optional overrides.

### Required: `extract_sharding_metadata`

```python
@abc.abstractmethod
def extract_sharding_metadata(
    self,
    item_key: str,
    item_value: Any,
) -> dict[NestedPath, ShardingMetadata]:
    ...
```

Walk `item_value` and return a `ShardingMetadata` for each sharded object,
keyed by its `NestedPath` within the item. Return an empty dict if the item
holds no sharded objects. Your `ShardingMetadata` subclass must implement
`to_dict` / `from_dict` (it auto-registers by type name for polymorphic
deserialization) and the `equivalent_ranks` property.

### Required: `load`

```python
@abc.abstractmethod
def load(
    self,
    source_path: Path,
    item_key: str,
    target_metadata: dict[NestedPath, ShardingMetadata],
    source_metadata: DistributedItemMetadata,
    target: Any,
    storage: Storage,
) -> list[NestedPath]:
    ...
```

Read from the source checkpoint and populate `target` in place, returning any
`NestedPath`s you could not reshard. Use
`source_metadata.get_file_path(rank, source_path, item_key)` to locate a source
rank's file and `storage` to read it.

### Optional: `should_reshard`

```python
def should_reshard(
    self,
    source_metadata: DistributedItemMetadata | None,
    target_metadata: dict[NestedPath, ShardingMetadata] | None,
) -> bool:
```

Override to customize when resharding is triggered — for example to allow certain
layout differences, compare only mesh topology, or add performance heuristics.
The default returns `False` if either side is missing metadata, and otherwise
returns `True` when some path's target sharding does not match any source rank
group.

### Optional: `skip_resharding`

```python
@property
def skip_resharding(self) -> bool:
    return False
```

Override to return `True` to skip metadata loading and resharding checks
altogether (identical save/load configuration on retry).

---

## Summary

| Concern | Class / knob | File |
| --- | --- | --- |
| High-level entry point | `CheckpointManager`, `ItemSpec(resharder=...)` | `checkpoint_manager.py` |
| Rank identity | `RankInfo`, `_get_default_rank_info()` | `types.py`, `builder.py` |
| Cross-rank coordination | `BarrierConfig`, `TCPStoreBarrierConfig`, `TCPStoreBarrier` | `barriers.py` |
| Disable coordination | `CheckpointWriterConfig(barrier_config=None)` | `checkpoint_writer.py` |
| Finalize hooks | `pre_finalize_callback`, `finalize_callback` | `builder.py`, `checkpoint_writer.py` |
| Metadata pipeline | `MetadataManager`, `DefaultMetadataManager` | `metadata_manager.py` |
| Metadata payload | `CheckpointMetadata`, `DistributedMetadata`, `DistributedItemMetadata` | `distributed_metadata.py` |
| Resharding interface | `Resharder` | `resharding.py` |
| Built-in DTensor resharder | `DefaultResharder`, `DTensorShardingMetadata` | `default_resharder.py`, `dtensor_metadata.py` |
| Loader wiring | `CheckpointLoader(reader, metadata_manager)` | `checkpoint_loader.py` |

Through `CheckpointManager` resharding is a single declaration: give an item's
`ItemSpec` a resharder and the manager wires the source metadata on save and the
target metadata on load — both ends, automatically. Under the hood (and when
driving the low-level savers and loaders directly) that means a resharder on each
item plus a metadata manager on save and load; with either absent the loader
takes the direct-read fast path, which is correct exactly when the layout is
unchanged.
