# `torch_checkpointing`

High-performance **asynchronous** checkpointing for PyTorch. It takes checkpoint saving off your training loop's critical path:

- **Zero-overhead saves** — `save()` returns immediately; model state is staged off the training device and written by a background process while your training step keeps running.
- **Save and load through one API** — a single `CheckpointManager` drives both; you pass plain `{name: value}` dicts and decide when to block on a save (for example, before exit).
- **Single-rank to distributed** — the same API scales from one process to large distributed jobs, and reshards across different parallelism layouts on load.

You interact with one object, `CheckpointManager`: `save(checkpoint_id, {...})`
and `load(checkpoint_id, into={...})` over a pluggable storage backend. A
`checkpoint_id` is a string interpreted by that backend; the default local
filesystem backend treats it as the path to a checkpoint directory. Rank,
storage, sharding metadata, and per-item copy/reshard behavior are configured
for you. Power users can still swap in bespoke components — storage backends,
resharders, cross-rank coordination — through the
[extension points](./docs/extensibility.md).

> **Experimental and pre-1.0.** The public API may still change.

## Installation

```bash
pip install torch_checkpointing
```

Requires Python >= 3.10 and torch >= 2.6.

Saving is asynchronous by default. The optimized async staging defaults
currently require CUDA; CPU-only users should use the explicit configuration in
[Troubleshooting](./docs/troubleshooting.md).

## Key features

- Non-blocking async saves overlapped with training (host-side staging + a background-process write).
- One high-level `CheckpointManager` for both save and load, with auto-detected rank, storage, and metadata.
- Plain-dict payloads: `save(id, {...})` / `load(id, into={...})` — tensors restored in place (identity preserved), scalars and JSON/bytes are first-class top-level items.
- Resharding on load across different distributed layouts (mesh / placement changes), wired automatically when an item declares a resharder.
- Pluggable storage behind the `Storage` / `StorageConfig` interface; a local filesystem backend ships in the package.

## Documentation

**Getting started**

- [Tutorial](./docs/tutorials.md) — checkpoint and resume a complete training loop.
- [Overview](./docs/index.md) — what the library does and how the pieces fit together.
- [Key concepts](./docs/key_concepts.md) — the `CheckpointManager`, the payload/`into=` model, and how async save and load work.
- [Configuring checkpoints](./docs/configuring_checkpoints.md) — per-item `layout`, `requires_copy`, and `resharder` via `ItemSpec`.
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
