# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train a small transformer with FSDP2 and checkpoint it with torch_checkpointing.

Each rank saves only its own shards, asynchronously, while training continues.
Rerunning the script resumes from the newest checkpoint, on any number of ranks.

    torchrun --nproc-per-node 2 train.py

See README.md for more ways to run it.
"""

import argparse
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from model import ModelArgs, Transformer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch_checkpointing import CheckpointManager, ItemSpec
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.state_transformations import (
    optimizer_transform_post,
    optimizer_transform_pre,
)

CHECKPOINT_NAME = re.compile(r"step_(\d+)")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="train until this many optimizer steps are done, counting resumed ones",
    )
    parser.add_argument(
        "--save-every",
        type=positive_int,
        default=20,
        help="save every N optimizer steps",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--sync-save",
        action="store_true",
        help="block the training loop on each save instead of saving asynchronously",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="compute in bfloat16; parameters and checkpoints stay float32",
    )
    parser.add_argument(
        "--global-batch-size",
        type=positive_int,
        default=48,
        help="sequences per step across all ranks; must divide by the number of ranks",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def log(message: str) -> None:
    if dist.get_rank() == 0:
        print(message, flush=True)


def init_distributed() -> torch.device:
    """Join the process group that torchrun set up; return this rank's device.

    Ranks train on GPUs if there are any, one per rank, and on CPU otherwise.
    """
    if not torch.cuda.is_available():
        dist.init_process_group("gloo")
        return torch.device("cpu")

    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if torch.cuda.device_count() < local_world_size:
        raise RuntimeError(
            f"{local_world_size} ranks per node but only {torch.cuda.device_count()} "
            "GPUs. Launch fewer ranks, or set CUDA_VISIBLE_DEVICES= to use the CPU."
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return device


def build_model(
    model_args: ModelArgs, device: torch.device, mixed_precision: bool
) -> Transformer:
    """Shard each transformer block, then the root, over all ranks.

    Parameters are allocated on the target device but left uninitialized.
    """
    with torch.device("meta"):
        model = Transformer(model_args)
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    mp_policy = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if mixed_precision
        else MixedPrecisionPolicy()
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    model.to_empty(device=device)
    return model


def get_batch(
    step: int, args: argparse.Namespace, model_args: ModelArgs, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this rank's slice of the global batch for ``step``.

    Each sequence counts up from a random token, so there is something to learn.
    The global batch depends only on the seed and the step, not on the number of
    ranks, so a resumed run, even on a different number of ranks, sees the same
    data as an uninterrupted one.
    """
    generator = torch.Generator().manual_seed(args.seed + step)
    starts = torch.randint(
        model_args.vocab_size, (args.global_batch_size, 1), generator=generator
    )
    tokens = (starts + torch.arange(model_args.max_seq_len + 1)) % model_args.vocab_size
    local_tokens = tokens.chunk(dist.get_world_size())[dist.get_rank()].to(device)
    return local_tokens[:, :-1], local_tokens[:, 1:]


# --- Checkpointing ---------------------------------------------------------


def build_checkpoint_manager(sync: bool) -> CheckpointManager:
    """Configure how the model, optimizer and training state are checkpointed.

    FSDP2 holds parameters and optimizer state as DTensors, and each rank saves
    only its local shards. A resharder records how they were sharded, so that a
    checkpoint can be loaded onto a different number of ranks.
    """
    resharder = DefaultResharder()
    return CheckpointManager(
        CheckpointManager.Config(
            items={
                "model": ItemSpec(resharder=resharder),
                "optimizer": ItemSpec(resharder=resharder),
                # An item without a resharder is read from the loading rank's
                # own file, which a rank added since the save does not have.
                # With a resharder, the step tensor is treated as replicated
                # and those ranks read it from a rank that saved it.
                "train_state": ItemSpec(resharder=resharder),
            },
            save=SyncCheckpointSaverConfig() if sync else AsyncCheckpointSaverConfig(),
        )
    )


def save_checkpoint(
    manager: CheckpointManager,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    """Save a checkpoint of the state after ``step`` optimizer steps.

    An async save returns once the state is being staged; a background process
    writes it while training continues. The next save, or ``manager.close()``,
    waits for the write to finish.
    """
    # FSDP2 state dicts hold this rank's DTensor shards. optimizer_transform_pre
    # keys the optimizer state by parameter name rather than by position.
    manager.save(
        str(path),
        {
            "model": model.state_dict(),
            "optimizer": optimizer_transform_pre(optimizer.state_dict()),
            "train_state": {"step": torch.tensor(step)},
        },
    )


def init_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Create the optimizer state, so that a checkpoint can be loaded into it.

    An optimizer creates its state, such as Adam's moments, in its first step.
    This takes a step with zero gradients. The step can still change the
    parameters, for example through weight decay, so load them afterwards.
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.requires_grad:
                param.grad = torch.zeros_like(param)
    optimizer.step()
    optimizer.zero_grad()


def load_checkpoint(
    manager: CheckpointManager,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load a checkpoint into the model and optimizer; return its step.

    The checkpoint may have been saved by a different number of ranks: each
    rank loads the parts of each tensor that it owns now.
    """
    # The load fills the live state in place. A freshly built optimizer has no
    # state to fill until it has taken a step.
    init_optimizer_state(optimizer)
    restored = manager.load(
        str(path),
        into={
            "model": model.state_dict(),
            "optimizer": optimizer_transform_pre(optimizer.state_dict()),
            "train_state": {"step": torch.tensor(0)},
        },
        strict=True,
    )
    # The tensors were loaded in place. This applies the rest of the optimizer
    # state, its param_groups, which hold the hyperparameters. Passing the
    # optimizer attaches the state to its parameters by name.
    optimizer.load_state_dict(
        optimizer_transform_post(restored["optimizer"], optimizer)
    )
    return int(restored["train_state"]["step"])


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the ``step_<n>`` directory with the highest ``n``, if any.

    A checkpoint is written to a temporary directory and renamed into place only
    once every rank has finished, so a ``step_<n>`` directory is complete.
    """
    if not checkpoint_dir.is_dir():
        return None
    steps = {}
    for path in checkpoint_dir.iterdir():
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match is not None and path.is_dir():
            steps[int(match.group(1))] = path
    return steps[max(steps)] if steps else None


# --- Training --------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = init_distributed()
    world_size = dist.get_world_size()
    if args.global_batch_size % world_size != 0:
        raise ValueError(
            f"--global-batch-size {args.global_batch_size} does not divide by "
            f"{world_size} ranks"
        )
    log(f"Training on {world_size} ranks ({device.type})")

    torch.manual_seed(args.seed)
    model_args = ModelArgs()
    model = build_model(model_args, device, args.mixed_precision)
    # Built from named parameters, so that its state dict records their names.
    optimizer = torch.optim.AdamW(model.named_parameters(), lr=3e-3)
    manager = build_checkpoint_manager(args.sync_save)

    try:
        # Rank 0 picks the checkpoint, so that every rank resumes from the same one.
        latest = [
            find_latest_checkpoint(args.checkpoint_dir)
            if dist.get_rank() == 0
            else None
        ]
        dist.broadcast_object_list(latest, src=0)
        resume_from = latest[0]
        if resume_from is None:
            model.reset_parameters()
            start_step = 0
        else:
            start_step = load_checkpoint(manager, resume_from, model, optimizer)
            log(f"Resumed from {resume_from}")

        for step in range(start_step, args.steps):
            inputs, labels = get_batch(step, args, model_args, device)
            loss = F.cross_entropy(model(inputs).flatten(0, 1), labels.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # An async save copies the state off the device in the background.
            # Wait for that copy to finish before updating the state in place.
            with manager.lock():
                optimizer.step()
            optimizer.zero_grad()

            steps_done = step + 1
            if steps_done % 10 == 0 or steps_done == args.steps:
                dist.all_reduce(loss)
                log(f"step {steps_done:4d} | loss {loss.item() / world_size:.4f}")

            if steps_done % args.save_every == 0 or steps_done == args.steps:
                path = args.checkpoint_dir / f"step_{steps_done}"
                start = time.perf_counter()
                save_checkpoint(manager, path, model, optimizer, steps_done)
                save_ms = (time.perf_counter() - start) * 1000
                log(f"Saving {path}: save() returned after {save_ms:.1f} ms")
    finally:
        # Waits for the last checkpoint to be written.
        manager.close()
    log("Done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
