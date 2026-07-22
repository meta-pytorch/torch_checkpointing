# Configuring checkpoints

Every checkpoint is described by the `get_items()` method of your `CheckpointBase`
subclass. It returns a `dict[str, CheckpointItem]`, and each `CheckpointItem`
carries the per-item configuration that controls what is written, whether it is
copied during async staging, where it lands on disk, and how it is resharded on
load.

This guide covers the four `CheckpointItem` fields in depth. For the surrounding
save/load flow see the [README](../README.md); for the underlying model of items,
layouts, and resharders see [./key_concepts.md](./key_concepts.md).

## Which knob for my scenario?

| I want to… | Set |
| --- | --- |
| Save state the loop keeps mutating (model, optimizer) | `requires_copy=True` (default) |
| Save a frozen snapshot cheaply (e.g. dataloader progress) | `requires_copy=False` |
| Get per-rank files with no configuration | leave `layout=None` (default layout) |
| Put an item in a specific file / subdirectory | `layout=LayoutInfo("path.pt", TorchSerialization())` |
| Store human-readable metadata (step, config) | `JsonSerialization(dict)` |
| Store already-serialized bytes | `RawSerialization()` |
| Store tensors in safetensors format | `SafetensorsSerialization()` |
| Resume on a different mesh / world size (DTensor) | `resharder=DTensorResharder()` |
| Reshard non-DTensor sharded state | a custom `Resharder` subclass |

Each field and format is explained in full below.

## The `CheckpointItem` fields

`CheckpointItem` is a dataclass with four fields, all optional:

```python
@dataclass
class CheckpointItem:
    value: Any = None
    requires_copy: bool = True
    layout: LayoutInfo | None = None
    resharder: Resharder | None = None
```

The `get_items()` dict keys identify each item and become part of the on-disk
file name. Keys must be valid filename components — only alphanumeric characters,
hyphens, and underscores (`^[a-zA-Z0-9_-]+$`) — and must be unique. No dots,
slashes, whitespace, special characters, or extensions.

### `value`

The payload for the item.

- **On write:** the actual data to save — for example a module's `state_dict()`,
  an optimizer's `state_dict()`, or a plain dict of scalars.
- **On read:** `value` is a *template* that controls filtering. If it is `None`,
  all data is loaded from the file without filtering. If it is a dict/list
  structure, only the keys/indices present in the template are loaded. Tensors in
  the template are copied in place, preserving references.

### `requires_copy`

Whether the value must be copied during async staging.

Async saves overlap the storage write with training: the value is first staged
(copied to host memory) and then written from a background subprocess while your
training loop continues. If the training loop mutates the value in place after
`save()` returns but before the background write reads it, the write can capture
inconsistent data — a race.

- **`requires_copy=True` (default):** the staging step deep-copies the value, so
  the training loop can keep mutating the live object safely. Use this for model
  and optimizer state, which the optimizer step mutates every iteration.
- **`requires_copy=False`:** the value is written without a staging copy. This is
  only safe for state that is **not mutated** between `save()` and the background
  write — for example a snapshot you took yourself, or dataloader progress state
  that is frozen for the duration of the write. Setting it `False` avoids the copy
  cost.

> Note: the staged object must be deep-copyable when `requires_copy=True`.

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
subclass. See [Resharders](#resharders) below.

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
  - `"model.pt"` -> `checkpoint_dir/model.pt`
  - `"rank_0/model.pt"` -> `checkpoint_dir/rank_0/model.pt` (subdirectory)
  - `f"model_rank_{rank}.pt"` -> per-rank files
  - `"shared/config.json"` -> a global file in a subdirectory
- **`serialization_format`** is one of the formats described in
  [Serialization formats](#serialization-formats).

### The canonical FLAT layout

The canonical layout is *flat*: one `torch.save` file per item, named after the
item and serialized with `TorchSerialization`:

```python
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization

LayoutInfo("model_state.pt", TorchSerialization())
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
the rank appended, giving you per-rank files without any explicit `LayoutInfo`.

### Custom layouts

Because `file_path` is an arbitrary relative path, you can organize a checkpoint
however you like — different components in separate files, per-rank vs. global
files, and subdirectories:

```python
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    TorchSerialization,
)


def get_items(self) -> dict[str, CheckpointItem]:
    rank = self._rank_info.global_rank
    return {
        # Global file (same content across ranks) — rank 0 typically writes it.
        "model": CheckpointItem(
            value=self._model.state_dict(),
            layout=LayoutInfo("model.pt", TorchSerialization()),
        ),
        # Per-rank file: the rank is baked into the path.
        "optimizer": CheckpointItem(
            value=self._optimizer.state_dict(),
            layout=LayoutInfo(f"optimizer_{rank}.pt", TorchSerialization()),
        ),
        # Global JSON misc_state in a subdirectory.
        "misc_state": CheckpointItem(
            value={"step": self._step},
            requires_copy=False,
            layout=LayoutInfo("shared/config.json", JsonSerialization(dict)),
        ),
    }
```

You can specialize `get_items()` per rank — for example, only include global
metadata on rank 0, or vary `file_path` by rank for per-rank files.

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

### `JsonSerialization(cls)`

```python
JsonSerialization(cls: type)
```

Serializes JSON-compatible data with `json.dump` / `json.load`. On write, the
value is pretty-printed with sorted keys and indentation. The `cls` argument records
the Python type to deserialize the JSON back into on read. Use it for
configuration, epoch/step counters, and other JSON-serializable metadata.

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

## Resharders

A resharder runs on **load** to map data saved under one distributed layout onto
this rank's target layout. Items with no resharder are read back directly.

- **`DTensorResharder()`** — reshards DTensor state across different Shard/Replicate
  placements and device mesh configurations, using DTensor's native placement APIs
  to compute shard geometry. It takes no constructor arguments.
- **Custom `Resharder`** — for state that is not a plain DTensor (for example,
  dataloader progress), subclass `Resharder` and implement its two abstract
  methods, `extract_sharding_metadata()` and `load()`.

Import `DTensorResharder` from `torch_checkpointing.dtensor_resharder`; the
`Resharder` base class lives in `torch_checkpointing.resharding`. `Resharder`,
`LoadPlan`, and `ReshardingInfo` are also re-exported from the top-level
`torch_checkpointing` package.

## Full `get_items()` example

Putting it together with the state a training job usually checkpoints — `model`,
`optimizer`, `dataloader`, and `misc_state`. Model and optimizer are DTensor
tensor state: copied during staging (`requires_copy=True`) and resharded with the
built-in `DTensorResharder`. The dataloader is an unmutated snapshot, so it skips
the staging copy (`requires_copy=False`) and uses a custom
`MyCustomDataloaderResharder`. `misc_state` is small replicated bookkeeping,
written as a single human-readable JSON file.

```python
from pathlib import Path
from typing import Any

from torch_checkpointing import CheckpointBase, CheckpointItem
from torch_checkpointing.checkpoint_layout import JsonSerialization, LayoutInfo
from torch_checkpointing.dtensor_resharder import DTensorResharder
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    ShardingMetadata,
)
from torch_checkpointing.resharding import Resharder
from torch_checkpointing.storage.base_storage import Storage
from torch_checkpointing.types import NestedPath, STATE_DICT


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


class TrainingCheckpoint(CheckpointBase):
    def __init__(self, model, optimizer, dataloader, step):
        self._model = model
        self._optimizer = optimizer
        self._dataloader = dataloader
        self._step = step

    def get_items(self) -> dict[str, CheckpointItem]:
        return {
            # model + optimizer: DTensor-sharded tensor state. Copy during
            # staging (the loop keeps mutating them) and reshard on load across
            # mesh/placement changes. The default layout writes one per-rank file
            # per item, which is what sharded state wants — no explicit `layout`.
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
            # dataloader: not a plain DTensor, so a custom resharder. An
            # unmutated snapshot, so skip the staging copy.
            "dataloader": CheckpointItem(
                value=self._dataloader.state_dict(),
                requires_copy=False,
                resharder=MyCustomDataloaderResharder(),
            ),
            # misc_state: small replicated bookkeeping (step, config, RNG). Not
            # sharded and identical on every rank — no copy, no resharder — and
            # kept human-readable in one global JSON file.
            "misc_state": CheckpointItem(
                value={"step": self._step},
                requires_copy=False,
                layout=LayoutInfo("misc_state.json", JsonSerialization(dict)),
            ),
        }

    def load_state_dict(self, state_dict: STATE_DICT) -> None:
        self._model.load_state_dict(state_dict["model"])
        self._optimizer.load_state_dict(state_dict["optimizer"])
        self._dataloader.load_state_dict(state_dict["dataloader"])
        self._step = state_dict["misc_state"]["step"]
```

With the default layout, `model`, `optimizer`, and `dataloader` are each written
as one per-rank `torch.save` file (`model_{rank}.pt`, `optimizer_{rank}.pt`,
`dataloader_{rank}.pt`), and `misc_state` is a single global `misc_state.json`. On
load, the model and optimizer are resharded onto the current mesh, the dataloader
state is resharded by your custom `Resharder`, and `misc_state` is read back
directly.

## See also

- [../README.md](../README.md) — save/load flow and quick start.
- [./key_concepts.md](./key_concepts.md) — items, layouts, and resharders overview.
- [./distributed_and_resharding.md](./distributed_and_resharding.md) — distributed
  metadata and resharding in depth.
