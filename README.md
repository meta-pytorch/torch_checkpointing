# `torch_checkpointing`

High-performance **asynchronous** checkpointing for PyTorch. It takes checkpoint saving off your training loop's critical path:

- **Zero-overhead saves** — `save()` returns immediately; model state is staged off the training device and written by a background process while your training step keeps running.
- **Save and load through one API** — a single `CheckpointManager` drives both; you decide when to block on a save (for example, before exit).
- **Single-rank to distributed** — the same API scales from one process to large distributed jobs, with resharding across different parallelism layouts on load.

You subclass `CheckpointBase` to declare what to save, then drive save and load through a `CheckpointManager` over a pluggable storage backend. Power users can swap in bespoke components — storage backends, resharders, cross-rank coordination — through the [extension points](./docs/extensibility.md).

> **Experimental and pre-1.0.** The public API may still change.

## Installation

```bash
pip install torch_checkpointing
```

Requires Python >= 3.10 and torch >= 2.6.

## Quick start

Subclass `CheckpointBase` to declare what to checkpoint. In a training script you then **resume from a checkpoint at startup** and **save periodically during training** — both through one `CheckpointManager`. An async `save()` returns immediately with a `(stage_future, write_future)` pair, so the training loop keeps running while the checkpoint is staged off-device and written in the background.

```python
import os

import torch
from torch_checkpointing import (
    CheckpointBase,
    CheckpointItem,
    CheckpointManager,
    STATE_DICT,
)
from torch_checkpointing.config import AsyncCheckpointSaverConfig
from torch_checkpointing.staging import CheckpointStagerConfig


class TrainingState(CheckpointBase):
    def __init__(self, state_dict):
        self._state_dict = state_dict

    def get_items(self) -> dict[str, CheckpointItem]:
        # Keys become file names, so they must match [a-zA-Z0-9_-]+.
        return {k: CheckpointItem(value=v) for k, v in self._state_dict.items()}

    def load_state_dict(self, state_dict: STATE_DICT) -> None:
        self._state_dict.update(state_dict)


# One manager drives both save and load. This example gates CUDA-specific
# staging features so it also runs on CPU-only hosts.
use_cuda = torch.cuda.is_available()
manager = CheckpointManager.Config(
    save=AsyncCheckpointSaverConfig(
        staging_config=CheckpointStagerConfig(
            use_pinned_memory=use_cuda,
            use_non_blocking_copy=use_cuda,
        )
    )
).build()

state = TrainingState({"model": torch.nn.Linear(10, 5).state_dict(), "step": 0})
checkpoint_id = "/tmp/ckpt/step_1000"

# Resume: at startup, restore a previous checkpoint if this run is restarting.
# `state` is the template that declares which keys to load back; it is updated
# in place. (Nothing to restore on a fresh run.)
if os.path.isdir(checkpoint_id):
    manager.load(checkpoint_id, state)

# ... training updates `state` ...

# Save during training: async save() returns immediately and writes in the
# background, so the loop keeps running. Block on write_future only when you
# need the checkpoint durable (e.g. before exit).
stage_future, write_future = manager.save(checkpoint_id, state)
write_future.result()

manager.close()
```

Prefer a blocking save? Use `CheckpointManager.Config.with_sync_save()` — `save()` writes inline and returns `None`; `load()` is unchanged.

## Key features

- Non-blocking async saves overlapped with training (host-side staging + a background-process write).
- One high-level `CheckpointManager` for both save and load, with auto-detected rank, storage, and metadata.
- Template-driven loads: tensors restored in place (identity preserved), mutable containers (dict / list / deque) updated in place.
- Per-item control via `CheckpointItem` — `value`, `requires_copy`, `layout`, `resharder`.
- Pluggable storage behind the `Storage` / `StorageConfig` interface; a local filesystem backend ships in the package.
- Resharding on load across different distributed layouts (mesh / placement changes).

## Documentation

**Getting started**

- [Overview](./docs/index.md) — what the library does and how the pieces fit together.
- [Key concepts](./docs/key_concepts.md) — `CheckpointManager`, `CheckpointBase` / `CheckpointItem`, and how save and load work.
- [Configuring checkpoints](./docs/configuring_checkpoints.md) — per-item `layout`, `requires_copy`, and `resharder` options.
- [Troubleshooting & FAQ](./docs/troubleshooting.md) — common errors and how to fix them.
- [API reference](./docs/api_reference.md) — the public symbols at a glance.

**Building bespoke components (power users)**

- [Extensibility](./docs/extensibility.md) — the extension points, and how to plug in your own infrastructure.
- [Storage](./docs/storage.md) — the `Storage` / `StorageConfig` interface and writing a custom backend.
- [Distributed and resharding](./docs/distributed_and_resharding.md) — multi-rank saves and custom resharding across mesh/placement changes.
- [Design & internals](./docs/design.md) — the async staging and background-write architecture.

**Contributing**

- [Contributing](./CONTRIBUTING.md) — development setup, testing, and pull-request guidance.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
