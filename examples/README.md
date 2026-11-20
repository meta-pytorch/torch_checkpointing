# Examples

Runnable examples of checkpointing with `torch_checkpointing`. Each one is a
short training script that shows the recommended way to checkpoint in one
setting. They are also starting points for experiments, for developing new
features, and for testing changes to the library on your own machine.

| Example | What it shows |
| --- | --- |
| [FSDP2](./fsdp2/) | Asynchronous, sharded saves of a model and optimizer sharded with `fully_shard`, and resuming on a different number of ranks. |

## Running the examples

The examples follow the `main` branch, so run them against the library in this
repository rather than a release. With [uv](https://docs.astral.sh/uv/), from
the repository root:

```bash
uv sync
source .venv/bin/activate
```

or with pip, in an environment of your choice:

```bash
pip install -e .
```

Then follow the README of the example. For example:

```bash
cd examples/fsdp2
torchrun --nproc-per-node 2 train.py
```

The examples need nothing beyond the dependencies of `torch_checkpointing`.
They run on CPUs or GPUs, with up to 4 processes, in a few minutes at most.

## Adding an example

An example should:

- have a README that explains what it shows and how to run it;
- run on a single machine with at most 4 ranks;
- run on CPU as well as GPU, unless it demonstrates a GPU-only feature;
- need no dependencies beyond those of `torch_checkpointing`;
- finish in under 10 minutes.

Add it to the table above.
