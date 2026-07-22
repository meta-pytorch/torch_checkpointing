# torch_checkpointing

`torch_checkpointing` is a library for scalable, **asynchronous** checkpointing of PyTorch models: a `CheckpointManager` stages model state and hands it to a background process that writes to storage, so training resumes almost immediately while the write completes out of band. The same API works single-rank and scales to large distributed jobs, with resharding across different parallelism layouts on load.

New here? Start with the [README quick start](../README.md), then read [Key concepts](./key_concepts.md).

## Getting started (most users)

- [Key concepts](./key_concepts.md) — the core building blocks: `CheckpointManager`, `CheckpointBase` / `CheckpointItem`, and how save and load work.
- [Configuring checkpoints](./configuring_checkpoints.md) — per-item options: `layout`, `requires_copy`, serialization formats, and `resharder`.
- [Troubleshooting & FAQ](./troubleshooting.md) — common errors and how to fix them.
- [API reference](./api_reference.md) — the public symbols at a glance.

## Building bespoke components (power users)

For integrating the library with your own infrastructure — a custom object store, a bespoke parallelism layout, custom cross-rank coordination. You implement one interface and plug it in; the defaults keep covering everything else.

- [Extensibility](./extensibility.md) — the extension points at a glance, and how to plug in your own components.
- [Storage](./storage.md) — the pluggable storage backend abstraction; write a backend for your filesystem or object store.
- [Distributed and resharding](./distributed_and_resharding.md) — checkpointing across ranks, and writing a custom `Resharder` to load into a different mesh or placement.
- [Design & internals](./design.md) — architecture and the design principles behind the library.

## Contributing

- [Contributing](../CONTRIBUTING.md) — development setup, running tests, and how to file issues.
