# Tutorial: checkpoint a training loop

This tutorial builds one complete workflow: resume the newest checkpoint, train
on CUDA, save without blocking each iteration, and wait for the final write at
shutdown. It uses the default local filesystem backend, so each `checkpoint_id`
is a string path naming a checkpoint directory.

The async preset currently requires CUDA for its default pinned-memory and
non-blocking staging options. For a CPU-only process, use the explicit stager
configuration in [Troubleshooting](./troubleshooting.md).

## Complete example

```python
import re
from pathlib import Path

import torch
from torch_checkpointing import CheckpointManager


STEP_DIRECTORY = re.compile(r"step_(\d+)")


def latest_checkpoint(root: Path) -> str | None:
    """Return the highest numbered `step_<n>` directory, if one exists."""
    if not root.is_dir():
        return None

    latest_step = -1
    latest_path: Path | None = None
    for candidate in root.iterdir():
        match = STEP_DIRECTORY.fullmatch(candidate.name)
        if not candidate.is_dir() or match is None:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_path = candidate
    return str(latest_path) if latest_path is not None else None


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This tutorial uses the CUDA async-save preset")

    device = torch.device("cuda")
    model = torch.nn.Linear(8, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_root = Path("/tmp/torch_checkpointing_tutorial")

    manager = CheckpointManager(CheckpointManager.Config.with_async_save())
    try:
        start_step = 0
        resume_from = latest_checkpoint(checkpoint_root)
        if resume_from is not None:
            restored = manager.load(
                resume_from,
                into={
                    "model": model.state_dict(),
                    # None loads the complete object-owned optimizer state.
                    "optimizer": None,
                    "step": 0,
                },
            )

            # Model tensors were copied into the live state_dict in place.
            # Optimizer state is owned by the optimizer and must be reapplied.
            optimizer.load_state_dict(restored["optimizer"])
            start_step = int(restored["step"]) + 1

        for step in range(start_step, 1_000):
            inputs = torch.randn(32, 8, device=device)
            loss = model(inputs).square().mean()
            loss.backward()

            # Do not mutate model or optimizer state while it is being staged.
            with manager.lock():
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                manager.save(
                    str(checkpoint_root / f"step_{step}"),
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                    },
                )
    finally:
        manager.close()


if __name__ == "__main__":
    main()
```

`save()` returns the background write `Future`; the example does not wait on it
inside the loop, so writing overlaps later training steps. A subsequent save
waits for the previous write before staging another checkpoint, and `close()`
waits for the final checkpoint to become durable. The main guard is required
because the async writer starts with multiprocessing `spawn`.

The `into=` mapping is recursive: only the nested keys and sequence indices in
the templates are loaded. Tensor values are copied into those templates in
place, which is why the model needs no second `load_state_dict()` call. A `None`
template loads the complete item instead, which lets the returned mapping carry
the optimizer's object-owned state for `optimizer.load_state_dict()`.

For a blocking write instead, construct the manager with
`CheckpointManager.Config.with_sync_save()`. The save/load payloads remain the
same, but `save()` writes inline and returns `None`.

Next, read [Configuring checkpoints](./configuring_checkpoints.md) for per-item
layout and copy controls, or
[Distributed checkpointing and resharding](./distributed_and_resharding.md) for
loading under a different device mesh.
