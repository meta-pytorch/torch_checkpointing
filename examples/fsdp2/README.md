# FSDP2

This example trains a small transformer that is sharded with FSDP2
([`fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)),
and checkpoints it with `torch_checkpointing`. It is adapted from the
[PyTorch FSDP2 example](https://github.com/pytorch/examples/tree/main/distributed/FSDP2).

It shows how to:

- save the sharded model and optimizer state, with each rank writing only the
  shards it owns and nothing gathered onto one rank;
- save asynchronously, so that training continues while the checkpoint is
  written;
- resume from the newest checkpoint;
- resume on a different number of ranks;
- run the same code on CPUs or GPUs.

## Requirements

- `torch_checkpointing` installed from this repository, as described in
  [Running the examples](../README.md#running-the-examples). Nothing else.
- Up to 4 processes. The example trains on GPUs if there are any, one per
  process, and on CPU otherwise. To train on CPU on a machine with GPUs, hide
  them: `CUDA_VISIBLE_DEVICES= torchrun ...`.

## Running it

From this directory, train for 60 steps on 2 ranks, saving every 20 steps:

```bash
torchrun --nproc-per-node 2 train.py --steps 60
```

Then run it again to resume from step 60 and train on to step 100 (the
default), this time on 4 ranks:

```bash
torchrun --nproc-per-node 4 train.py
```

The output looks like this (the loss depends on the device):

```
Training on 2 ranks (cuda)
step   10 | loss 1.3444
step   20 | loss 0.1174
Saving checkpoints/step_20: save() returned after 403.4 ms
step   30 | loss 0.0217
step   40 | loss 0.0089
Saving checkpoints/step_40: save() returned after 3.3 ms
step   50 | loss 0.0057
step   60 | loss 0.0044
Saving checkpoints/step_60: save() returned after 2.9 ms
Done
```

```
Training on 4 ranks (cuda)
Resumed from checkpoints/step_60
step   70 | loss 0.0037
step   80 | loss 0.0034
Saving checkpoints/step_80: save() returned after 235.2 ms
step   90 | loss 0.0030
step  100 | loss 0.0027
Saving checkpoints/step_100: save() returned after 3.1 ms
Done
```

Things to notice:

- **The resumed run picks up where the first one stopped.** A freshly
  initialized model starts with a loss of about 5.7. Every step trains on the
  same global batch whatever the number of ranks, so a resumed run follows the
  loss curve of an uninterrupted one.
- **Saving barely interrupts training.** `save()` returns as soon as it has
  started copying the state off the training device. The copy finishes in the
  background, and a background process then writes the checkpoint while
  training continues. Compare with `--sync-save`, which writes the checkpoint
  before `save()` returns. The first save in a run can take longer, mostly
  while it waits for the background writer process to finish starting. A save
  also waits for the previous checkpoint to be written, so saving more often
  than checkpoints can be written slows training down.

To start from scratch, delete the checkpoint directory.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--steps N` | 100 | Train until N optimizer steps are done, counting the steps of resumed runs. |
| `--save-every N` | 20 | Save every N steps. The last step is always saved. |
| `--checkpoint-dir DIR` | `checkpoints` | Where checkpoints are saved, and resumed from. |
| `--sync-save` | | Save synchronously: `save()` returns once the checkpoint is written. |
| `--mixed-precision` | | Compute in bfloat16. Parameters, and so checkpoints, stay in float32. |
| `--global-batch-size N` | 48 | Sequences per step across all ranks. It must divide by the number of ranks. |
| `--seed N` | 0 | Seed for initialization and data. |

## What gets saved

Each checkpoint is a directory. After a save on 2 ranks:

```
checkpoints/step_60/
├── metadata.pkl        # how each tensor was sharded, used to reshard on load
├── model_0.pt          # rank 0's shards of the parameters
├── model_1.pt
├── optimizer_0.pt      # rank 0's shards of the optimizer state
├── optimizer_1.pt
├── train_state_0.pt    # the step count
└── train_state_1.pt
```

The ranks write into `checkpoints/tmp_step_60/`, which is renamed to
`checkpoints/step_60/` once every rank has finished. A `step_<n>` directory is
therefore always complete, and the example resumes from the one with the
highest `n`.

## How it works

The checkpointing code is in `build_checkpoint_manager()`, `save_checkpoint()`,
`init_optimizer_state()` and `load_checkpoint()` in [`train.py`](./train.py).

**Configuring.** One `CheckpointManager` does both saving and loading. Each
item that holds sharded state declares a `DefaultResharder`, which records how
its DTensors were sharded when they are saved and redistributes them when they
are loaded under a different sharding:

```python
resharder = DefaultResharder()
manager = CheckpointManager(
    CheckpointManager.Config(
        items={
            "model": ItemSpec(resharder=resharder),
            "optimizer": ItemSpec(resharder=resharder),
            "train_state": ItemSpec(resharder=resharder),
        },
        save=AsyncCheckpointSaverConfig(),  # or SyncCheckpointSaverConfig()
    )
)
```

**Saving.** Under FSDP2, `model.state_dict()` holds this rank's shards of each
parameter, as DTensors, and so does the optimizer's state. An optimizer state
dict identifies each parameter by its position in the optimizer;
`optimizer_transform_pre()` identifies it by its name instead, so that the
state of each parameter can be found in the checkpoint by name. It reads the
names from the state dict, which records them only if the optimizer was built
from `model.named_parameters()`. Every rank passes its own state to `save()`:

```python
optimizer = torch.optim.AdamW(model.named_parameters(), lr=3e-3)
...
manager.save(
    "checkpoints/step_60",
    {
        "model": model.state_dict(),
        "optimizer": optimizer_transform_pre(optimizer.state_dict()),
        "train_state": {"step": torch.tensor(60)},
    },
)
```

An asynchronous `save()` returns once it has started copying the state off the
device. Until that copy finishes, the state must not change, so the optimizer
step is wrapped in `manager.lock()`, which waits for the copy if one is in
progress:

```python
with manager.lock():
    optimizer.step()
```

The next `save()`, or `manager.close()` at the end of training, waits for the
previous checkpoint to be written.

**Loading.** The model and optimizer are built for the current number of ranks
first. Their state dicts are the templates that `load()` fills in place: each
rank reads the parts of each tensor that it now owns, from whichever files hold
them. An optimizer creates its state in its first step, so a freshly built one
has none to load into; `init_optimizer_state()` takes a step with zero
gradients to create it. Without that step, `load()` would load no optimizer
state, and report no error. `optimizer_transform_post()` converts the names
back to the positions of the parameters in the optimizer it is given, and
`optimizer.load_state_dict()` applies the rest of the optimizer state, its
`param_groups`:

```python
init_optimizer_state(optimizer)
restored = manager.load(
    "checkpoints/step_60",
    into={
        "model": model.state_dict(),
        "optimizer": optimizer_transform_pre(optimizer.state_dict()),
        "train_state": {"step": torch.tensor(0)},
    },
    strict=True,
)
optimizer.load_state_dict(optimizer_transform_post(restored["optimizer"], optimizer))
```

## Differences from the PyTorch example

The [PyTorch example](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)
gathers the full model and optimizer state onto rank 0, which writes them while
the other ranks wait. This example has every rank write its own shards, in the
background. The example also drops the explicit prefetching option, which does
not affect checkpointing, and trains on data that it can learn, so that you can
see whether a resumed run continued where the previous run stopped.

## Limitations

- When a checkpoint is loaded onto a different number of ranks, only the
  tensors in resharded items are loaded. Other values in those items keep the
  values of the templates: the optimizer's hyperparameters in `param_groups`
  stay as the optimizer was constructed. The hyperparameters in this example
  never change, so it is not affected. If yours change during training, save
  them as tensors, as this example does with the step count.
- An item without a resharder is loaded from the loading rank's own file. A
  rank that was added since the checkpoint was saved has no file, and fails to
  load the item. That is why `train_state` declares a resharder: the step count
  is a tensor, which `DefaultResharder` treats as replicated across ranks, so a
  new rank reads it from a rank that saved it.
