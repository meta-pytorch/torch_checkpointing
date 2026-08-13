# Configuring checkpoints

Per-item configuration is expressed with an `ItemSpec`, bound **once** to a key in
`CheckpointManager.Config.items`. Each `ItemSpec` carries the per-item "how" —
whether the value is copied during async staging, where it lands on disk, and how
it is resharded on load — and is overlaid onto the value that arrives in the
payload at save/load time. Keys not listed in `items` fall back to
`Config.default`.

```python
from torch_checkpointing import CheckpointManager, ItemSpec

manager = CheckpointManager(CheckpointManager.Config(
    items={
        "dataloader": ItemSpec(requires_copy=False),   # override just this item
    },
    # default applies to every key not listed above (model, optimizer, step, ...)
))
```

For the common case you pass bare values in the payload and never touch the
schema; reach for `items` only to override a specific item. This guide covers the
four `ItemSpec` fields in depth. For the surrounding save/load flow see the
[tutorial](./tutorials.md); for the underlying model of items, layouts, and
resharders see [./key_concepts.md](./key_concepts.md).

## Which knob for my scenario?

| I want to… | Set |
| --- | --- |
| Save state the loop keeps mutating (model, optimizer) | `ItemSpec(requires_copy=True)` (default) |
| Save a frozen snapshot cheaply (e.g. dataloader progress) | `ItemSpec(requires_copy=False)` |
| Get per-rank files with no configuration | leave `layout=None` (default layout) |
| Put an item in a specific file / subdirectory | `ItemSpec(layout=LayoutInfo("state_{rank}.pt", TorchSerialization()))` |
| Store human-readable metadata (step, config) | `ItemSpec(layout=LayoutInfo(..., JsonSerialization()))` |
| Store already-serialized bytes | `ItemSpec(layout=LayoutInfo(..., RawSerialization()))` |
| Store tensors in safetensors format | `ItemSpec(layout=LayoutInfo(..., SafetensorsSerialization()))` |
| Resume on a different mesh / world size (DTensor) | `ItemSpec(resharder=DTensorResharder())` |
| Reshard non-DTensor sharded state | `ItemSpec(resharder=MyCustomResharder())` |
| Require an item to be present (fail fast if missing) | `ItemSpec(required=True)` (default) |
| Make an un-listed key raise instead of using defaults | `Config(default=None)` |

Each field and format is explained in full below.

## The `ItemSpec` fields

`ItemSpec` is a dataclass with four fields, all optional:

```python
@dataclass
class ItemSpec:
    requires_copy: bool = True
    layout: LayoutInfo | None = None
    resharder: Resharder | None = None
    required: bool = True
```

An `ItemSpec` describes only per-item behavior, bound once in `Config.items`.
The value itself comes from the payload you pass at call time —
`manager.save(id, {"model": ...})` on write and `manager.load(id, into={"model":
...})` on read. Keeping the schema (the specs) separate from the data (the
payload) lets you declare configuration once and pass plain dicts every save.

`Config.items` maps each top-level payload key to its spec; `Config.default` is
the spec applied to any key **not** listed there. Set `default=None` to make the
config strict — an un-listed key then raises instead of falling back to defaults.
The payload keys identify each item and become part of the on-disk file name.
The checkpoint engine requires only alphanumeric characters, hyphens, and
underscores (`^[a-zA-Z0-9_-]+$`). No dots, slashes, whitespace, special
characters, or extensions.

### `requires_copy`

Whether the value must be copied during async staging.

Async saves overlap the storage write with training: the value is first staged
(copied to host memory) and then written from a background subprocess while your
training loop continues. If the training loop mutates the value in place after
`save()` returns but before the background write reads it, the write can capture
inconsistent data — a race.

- **`requires_copy=True` (default):** staging deep-copies the value and moves
  device tensors to host memory, so the training loop can keep mutating the live
  object safely. Use this for model and optimizer state, which the optimizer
  step mutates every iteration.
- **`requires_copy=False`:** the value bypasses staging unchanged. Use it only
  for data that is already host-resident and remains stable until the background
  write completes. Setting it `False` avoids the copy cost but does not remove
  the async writer's multiprocessing requirement: the value must still be
  pickleable.

> Note: the staged object must be deep-copyable when `requires_copy=True`.
> Every value must be pickleable when using an async saver, regardless of this
> setting.

### `layout`

A `LayoutInfo | None` describing *where* and *how* the item is serialized.

If `layout` is `None`, the item uses the **default layout** (see below). Provide a
`LayoutInfo` to take full control of the file path and serialization format.

### `resharder`

An optional `Resharder` used on **load** to redistribute the item across a
different parallelization layout than the one it was saved with (for example,
resuming on a different device mesh or with different sharding). Items with no
resharder are read back directly.

The built-in `DTensorResharder` handles DTensor state. For state that is not a
plain DTensor (such as dataloader progress) you attach your own `Resharder`
subclass. See [Resharders](#resharders) below. When an item declares a resharder,
the manager auto-wires the sharding-metadata pipeline on both save and load — you
do not set up any metadata plumbing yourself.

### `required`

Whether the item must be present. With the default (`True`), a missing item fails
fast; set it `False` for optional items that may be absent from some checkpoints.

## Layouts

A layout maps a single top-level item key to a single file. Only top-level keys
are supported, and each key maps to its own unique file — there is no file sharing
between keys.

```python
@dataclass(frozen=True)
class LayoutInfo:
    file_path: str
    serialization_format: SerializationFormat
```

- **`file_path`** is relative to the checkpoint directory and gives you full
  control over naming and organization:
  - `"model_{rank}.pt"` -> a file per rank
  - `"rank_{rank}/model.pt"` -> a file inside a per-rank subdirectory
  - `"model.pt"` -> one fixed path, suitable for single-rank use
- **`serialization_format`** is one of the formats described in
  [Serialization formats](#serialization-formats).

You bind a `LayoutInfo` through the item's `ItemSpec`:

```python
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization

manager = CheckpointManager(CheckpointManager.Config(
    items={
        "model": ItemSpec(layout=LayoutInfo("model.pt", TorchSerialization())),
    },
))
```

### The `{rank}` placeholder

The schema is rank-agnostic: you write one `ItemSpec` that every rank shares, and
the manager fills in each process's identity for you. A `"{rank}"` placeholder in
`file_path` is substituted with the current global rank at save/load time, so a
single spec produces per-rank files:

```python
LayoutInfo("model_{rank}.pt", TorchSerialization())
# on global rank 3 -> checkpoint_dir/model_3.pt
```

You never construct or pass rank information yourself — leave `{rank}` in the path
and the manager expands it. `{rank}` is the only recognized placeholder;
`{key}` and other brace-delimited text are not substituted. In a distributed
save, every rank processes every configured layout, so a path without `{rank}`
causes the ranks to target the same file. Use fixed paths only for single-rank
checkpoints or when coordination is handled outside this API.

### The canonical FLAT layout

The canonical layout is *flat*: one `torch.save` file per item, named after the
item and serialized with `TorchSerialization`:

```python
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization

LayoutInfo("model_state_{rank}.pt", TorchSerialization())
```

This keeps each item in its own predictable file and is the recommended starting
point for tensor state.

### The default layout

When `layout` is `None`, the writer falls back to a per-rank default. Internally
this is:

```python
def default_layout_info(key: str, rank: int) -> LayoutInfo:
    return LayoutInfo(
        f"{key}_{rank}.pt",
        TorchSerialization(),
    )
```

So an item keyed `"model_state"` on global rank `3` is written to
`model_state_3.pt` with `torch.save`. The default layout is the flat layout with
the rank appended, giving you per-rank files without any explicit `LayoutInfo` —
exactly the `f"{item_key}_{rank}.pt"` shape you would otherwise construct for a
specific item key and rank. `{key}` is not a layout placeholder.

### Custom layouts

Because `file_path` is an arbitrary relative path, you can organize a checkpoint
with different components in separate files and subdirectories. Bind one
`LayoutInfo` per item in `Config.items`; include `{rank}` anywhere multiple ranks
share the config:

```python
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    TorchSerialization,
)

manager = CheckpointManager(CheckpointManager.Config(
    items={
        "model": ItemSpec(
            layout=LayoutInfo("model_{rank}.pt", TorchSerialization()),
        ),
        # Per-rank file: {rank} is expanded by the manager per process.
        "optimizer": ItemSpec(
            layout=LayoutInfo("optimizer_{rank}.pt", TorchSerialization()),
        ),
        # Per-rank JSON misc_state in a subdirectory.
        "misc_state": ItemSpec(
            requires_copy=False,
            layout=LayoutInfo("rank_{rank}/misc_state.json", JsonSerialization()),
        ),
    },
))
```

Because the schema is shared across ranks, `{rank}` lets every process use the
same config without writing to the same path.

## Serialization formats

`serialization_format` selects how the item's value is encoded. Four formats ship
with the library.

### `TorchSerialization` (used by the default and flat layouts)

```python
TorchSerialization()
```

Uses `torch.save` / `torch.load`. This is the general-purpose format and the one
used by the flat and default layouts. It handles arbitrary picklable Python
objects, including nested state dicts with tensors, DTensors, and non-tensor
values.

### `JsonSerialization(cls=None)`

```python
JsonSerialization(cls: type | None = None)
```

Serializes JSON-compatible data with `json.dump` / `json.load`. On write, the
value is pretty-printed with sorted keys and indentation. When `cls` is a type,
it records the Python type to reconstruct on read. When it is `None` (the
default), the reader returns the raw JSON-decoded dict, list, or scalar. Use it
for configuration, epoch/step counters, and other JSON-serializable metadata.

### `RawSerialization`

```python
RawSerialization()
```

Writes bytes as-is, with no encoding. The value **must** be `bytes` — passing
anything else raises a `ValueError` on write. Use it for pre-serialized payloads,
such as an already-encoded JSON string, or raw binary data.

### `SafetensorsSerialization(metadata)`

```python
SafetensorsSerialization(metadata: dict[str, str] | None = None)
```

Tensor-only format backed by the `safetensors` library.

- **Tensor-only:** non-tensor leaf values are not supported and raise a
  `ValueError` on write. Use `TorchSerialization` or `JsonSerialization` for
  non-tensor data.
- **Nested-dict flatten/re-nest:** on write, nested dicts (and lists/tuples) are
  flattened to dot-separated keys with tensor leaves. On read, when a target
  template is provided, the reader re-nests the flat keys back to the target's
  structure so the value round-trips. Leaves missing from the flat source are
  dropped from dict parents (and reported as missing) rather than silently
  misaligned; for list/tuple parents, missing leaves come back as `None`.
- **DTensor unwrap:** DTensor leaves are automatically unwrapped to their local
  tensor on save, and tensors are forced contiguous.
- **`metadata`:** optional user metadata embedded in the file. Both keys and
  values must be `str` (the safetensors contract). Treat the dict as read-only
  after construction — mutating its contents after building the format is not
  supported.

```python
from torch_checkpointing.checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
)

LayoutInfo(
    "model_state.safetensors",
    SafetensorsSerialization(metadata={"format_version": "1"}),
)
```

The writer recognizes the four formats documented in this section. Defining a
new `SerializationFormat` subclass is not currently sufficient to make the
writer support a custom encoding.

## Resharders

A resharder runs on **load** to map data saved under one distributed layout onto
this rank's target layout. Items with no resharder are read back directly.

- **`DTensorResharder()`** — reshards DTensor state across different
  Shard/Replicate placements, StridedShard layouts, and device mesh
  configurations. Disjoint rank-local layouts are represented as multiple
  contiguous load-plan slices. It takes no constructor arguments.
- **Custom `Resharder`** — for state that is not a plain DTensor (for example,
  dataloader progress), subclass `Resharder` and implement its two abstract
  methods, `extract_sharding_metadata()` and `load()`.

You attach a resharder through the item's `ItemSpec`:

```python
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.dtensor_resharder import DTensorResharder

manager = CheckpointManager(CheckpointManager.Config(
    items={"model": ItemSpec(resharder=DTensorResharder())},
))
manager.save(path, {"model": model.state_dict()})       # records sharding metadata
manager.load(path, into={"model": model.state_dict()})  # reshards onto the new layout
```

Declaring the resharder is the whole contract: the manager **auto-wires** the
sharding-metadata pipeline for you — recording the source metadata on save and the
target metadata on load — so there is no `MetadataManager` to construct or hand to
the loader. Resharding needs live targets, so always pass `into=` when you resume
onto a changed layout.

Import `DTensorResharder` from `torch_checkpointing.dtensor_resharder`; the
`Resharder` base class lives in `torch_checkpointing.resharding`. `Resharder`,
`LoadPlan`, and `ReshardingInfo` are also re-exported from the top-level
`torch_checkpointing` package.

## Full `CheckpointManager.Config` example

Putting it together with the state a training job usually checkpoints — `model`,
`optimizer`, `dataloader`, and `misc_state`. You declare each item's spec once in
`Config.items`, then pass plain payloads to `save()` and `load()`. Model and
optimizer are DTensor tensor state: copied during staging (`requires_copy=True`)
and resharded with the built-in `DTensorResharder`. The dataloader is an unmutated
snapshot, so it skips the staging copy (`requires_copy=False`) and uses a custom
`MyCustomDataloaderResharder`. `misc_state` is small per-rank bookkeeping,
written as human-readable JSON.

```python
from pathlib import Path
from typing import Any

from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.checkpoint_layout import JsonSerialization, LayoutInfo
from torch_checkpointing.dtensor_resharder import DTensorResharder
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    ShardingMetadata,
)
from torch_checkpointing.resharding import Resharder
from torch_checkpointing.storage.base_storage import Storage
from torch_checkpointing.types import NestedPath


class MyCustomDataloaderResharder(Resharder):
    """A resharder for non-DTensor dataloader progress state.

    Implement the two abstract methods to describe how the item's sharded
    objects map from the saved layout onto this rank's target layout.
    """

    def extract_sharding_metadata(
        self,
        item_key: str,
        item_value: Any,
    ) -> dict[NestedPath, ShardingMetadata]:
        ...

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


# Declare the per-item specs once. The manager auto-wires the sharding-metadata
# pipeline for the items that carry a resharder, on both save and load.
manager = CheckpointManager(CheckpointManager.Config(
    items={
        # model + optimizer: DTensor-sharded tensor state. Copy during staging
        # (the loop keeps mutating them) and reshard on load across mesh/placement
        # changes. No explicit `layout` -> the default per-rank file, which is
        # what sharded state wants.
        "model": ItemSpec(
            requires_copy=True,
            resharder=DTensorResharder(),
        ),
        "optimizer": ItemSpec(
            requires_copy=True,
            resharder=DTensorResharder(),
        ),
        # dataloader: not a plain DTensor, so a custom resharder. An unmutated
        # snapshot, so skip the staging copy.
        "dataloader": ItemSpec(
            requires_copy=False,
            resharder=MyCustomDataloaderResharder(),
        ),
        # misc_state: small per-rank bookkeeping (step, config, RNG). No copy or
        # resharder, and kept human-readable in a rank-specific JSON file.
        "misc_state": ItemSpec(
            requires_copy=False,
            layout=LayoutInfo("misc_state_{rank}.json", JsonSerialization()),
        ),
    },
))


# --- Save: pass the live values as a plain payload. ---
manager.save(
    checkpoint_id,
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "dataloader": dataloader.state_dict(),
        "misc_state": {"step": step},
    },
)

# --- Load: `into=` supplies the live targets to restore into and reshard onto. ---
restored = manager.load(
    checkpoint_id,
    into={
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "dataloader": dataloader.state_dict(),
        "misc_state": {"step": 0},
    },
)
optimizer.load_state_dict(restored["optimizer"])
dataloader.load_state_dict(restored["dataloader"])
step = restored["misc_state"]["step"]
```

With no explicit `layout`, `model`, `optimizer`, and `dataloader` are each written
as one per-rank `torch.save` file (`model_0.pt`, `optimizer_0.pt`, and
`dataloader_0.pt` on rank 0), and each rank writes its own
`misc_state_<rank>.json`. On load, the model and optimizer are resharded onto the
current mesh, the dataloader state is resharded by your custom `Resharder`, and
each rank directly reads its own `misc_state` copy. Tensors in the `into=`
templates are copied in place (identity preserved), so the model does not need
a second `load_state_dict()` call; the returned mapping lets you read scalars
back and re-apply object-owned state.

## See also

- [./tutorials.md](./tutorials.md) — end-to-end save/load training flow.
- [./key_concepts.md](./key_concepts.md) — items, layouts, and resharders overview.
- [./distributed_and_resharding.md](./distributed_and_resharding.md) — distributed
  metadata and resharding in depth.
