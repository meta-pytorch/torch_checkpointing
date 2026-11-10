# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# WORKAROUND: copied from PyTorch's torch/distributed/tensor/_utils.py, where
# https://github.com/pytorch/pytorch/pull/203180 added it. No PyTorch release has
# it yet. Once every PyTorch release this package supports does, delete this
# module and import compute_local_shape_and_chunks from
# torch.distributed.tensor._utils instead. Until then, keep the code identical to
# upstream, except where PyTorch 2.11 lacks a helper that upstream uses:
# - _is_shard_like below stands in for the one in placement_types.
# - int(placement.split_factor) stands in for _StridedShard._split_factor_int().

import itertools
import math
from collections.abc import Sequence
from typing import TypeGuard

import torch
from torch._prims_common import ShapeType
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import _StridedShard, Placement, Shard
from torch.types import IntLikeType


def _is_shard_like(p: Placement) -> TypeGuard[Shard | _StridedShard]:
    """Check if a placement is Shard or _StridedShard."""
    return isinstance(p, (Shard, _StridedShard))


def compute_local_shape_and_chunks(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> tuple[
    tuple[IntLikeType, ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
]:
    """
    Compute the local tensor shape of a DTensor on its current global rank, and the
    chunks of the global tensor that the local tensor holds.

    :func:`compute_local_shape_and_global_offset` returns where the local tensor
    starts in the global tensor, which describes it only if it is one chunk of
    consecutive rows and columns. A ``_StridedShard`` can leave a rank holding
    several separate chunks. The chunks describe the local tensor either way. Each is
    a rectangle of consecutive indices along every dim, of the global tensor and of
    the local tensor: chunk ``i`` starts at ``global_offsets[i]`` in the global
    tensor and at ``local_offsets[i]`` in the local tensor, and has shape
    ``local_sizes[i]``. This is the layout of the fields of
    :class:`torch.distributed.checkpoint.protocol.CheckpointableTensor`::

        size = local_sizes[i]
        local_index = tuple(slice(o, o + n) for o, n in zip(local_offsets[i], size))
        global_index = tuple(slice(o, o + n) for o, n in zip(global_offsets[i], size))
        assert torch.equal(local_tensor[local_index], global_tensor[global_index])

    The number of chunks depends only on ``placements``: along each dim it is the
    product of the ``split_factor`` of the ``_StridedShard`` placements on that dim,
    and some chunks can be empty. Because it depends on neither the rank nor the
    sizes, every rank in the mesh returns the same structure, and this can be traced
    with a symbolic mesh coordinate, as under
    ``torch.compiler.config.compile_on_one_rank``. An empty chunk can have the same
    global offset as another chunk, and DCP looks up the chunks of a
    ``CheckpointableTensor`` by global offset, so drop the chunks with a zero in
    ``local_sizes`` before building one. Under ``LocalTensorMode``, do that in a
    function decorated with ``maybe_run_for_local_tensor`` that also uses the
    chunks, so that it sees each rank's sizes as ints. Outside one, a per-rank size
    equals 0 only if it is 0 on every rank, and ``<`` and ``>`` raise on it. The
    function can't return the filtered chunks, because their number can differ
    between ranks. With a symbolic mesh coordinate, the comparison can need a
    data-dependent guard.

    Example: a tensor of shape ``(8, 2)``, with its elements labelled ``a`` to ``p``,
    and placements ``[_StridedShard(0, split_factor=2), Replicate()]`` on a 2 x 2
    mesh::

        row 0  [ a, b ]
        row 1  [ c, d ]
        row 2  [ e, f ]
        row 3  [ g, h ]
        row 4  [ i, j ]
        row 5  [ k, l ]
        row 6  [ m, n ]
        row 7  [ o, p ]

                         mesh dim 1
                         0                          1
        mesh dim 0   0   rank 0: rows 0, 1, 4, 5    rank 1: rows 0, 1, 4, 5
                     1   rank 2: rows 2, 3, 6, 7    rank 3: rows 2, 3, 6, 7

    Rank 0 has local tensor ``[[a, b], [c, d], [i, j], [k, l]]``: global rows 0, 1
    at local rows 0, 1, and global rows 4, 5 at local rows 2, 3::

        local_shape = (4, 2)
        global_offsets = ((0, 0), (4, 0))
        local_offsets = ((0, 0), (2, 0))
        local_sizes = ((2, 2), (2, 2))

    With ``[_StridedShard(0, split_factor=2), Shard(0)]`` instead, rank 0 holds only
    rows 0, 1, and its second chunk is empty::

        local_shape = (2, 2)
        global_offsets = ((0, 0), (4, 0))
        local_offsets = ((0, 0), (2, 0))
        local_sizes = ((2, 2), (0, 2))

    Args:
        global_shape (ShapeType): The global shape of the DTensor.
        mesh (:class:`DeviceMesh`): The device mesh this DTensor is distributed on.
        placements (Sequence[:class:`Placement`]): The placements of the DTensor.

    Return:
        local_shape: the shape of the DTensor's _local_tensor on the current rank.
        global_offsets: for each chunk, where it starts in the global tensor.
        local_offsets: for each chunk, where it starts in the local tensor.
        local_sizes: for each chunk, its shape.
        Under ``LocalTensorMode``, values that differ between ranks are ``SymInt`` s
        holding each rank's value. If the current rank is not part of the mesh,
        ``local_shape`` is ``(0,)`` and there are no chunks.
    """
    if not mesh._is_current_rank_part_of_mesh():
        return (0,), (), (), ()
    coordinate = [
        mesh._sym_get_coordinate(mesh_dim) if _is_shard_like(placement) else None
        for mesh_dim, placement in enumerate(placements)
    ]
    return _compute_local_shape_and_chunks(
        global_shape, mesh.shape, coordinate, placements
    )


# accept 'plain data types' to enable simpler unit testing without creating device mesh
@maybe_run_for_local_tensor
def _compute_local_shape_and_chunks(
    global_shape: ShapeType,
    mesh_shape: ShapeType,
    my_coordinate: Sequence[IntLikeType | None],
    placements: Sequence[Placement],
) -> tuple[
    tuple[IntLikeType, ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
    tuple[tuple[IntLikeType, ...], ...],
]:
    """
    :func:`compute_local_shape_and_chunks` for the mesh coordinate ``my_coordinate``.
    Under ``LocalTensorMode`` this runs once per rank, and the ranks' results are
    combined value by value, which needs the same structure on every rank.
    """
    # For each tensor dim, the global indices the local tensor holds, in local
    # order, run-length encoded as (global_start, length). Sharding is applied
    # left to right.
    runs_per_dim: list[list[tuple[IntLikeType, IntLikeType]]] = [
        [(0, size)] for size in global_shape
    ]
    for mesh_dim, placement in enumerate(placements):
        if not _is_shard_like(placement):
            continue
        shard_dim = placement.dim
        if shard_dim >= len(global_shape):
            raise ValueError(
                f"Sharding dim {shard_dim} greater than tensor ndim {len(global_shape)}"
            )
        rank = my_coordinate[mesh_dim]
        if rank is None:
            raise AssertionError(f"No mesh coordinate for shard mesh dim {mesh_dim}")
        runs_per_dim[shard_dim] = _shard_runs(
            runs_per_dim[shard_dim], mesh_shape[mesh_dim], rank, placement
        )

    # A chunk is one run along each dim, as ((global_start, length), local_start).
    local_starts = [
        list(itertools.accumulate((length for _, length in r), initial=0))[:-1]
        for r in runs_per_dim
    ]
    chunks = list(
        itertools.product(*(zip(r, s) for r, s in zip(runs_per_dim, local_starts)))
    )
    num_chunks = math.prod(
        int(p.split_factor) for p in placements if isinstance(p, _StridedShard)
    )
    if len(chunks) != num_chunks:
        raise AssertionError(
            f"Expected {num_chunks} chunks for {placements}, got {len(chunks)}"
        )
    return (
        tuple(sum(length for _, length in r) for r in runs_per_dim),
        tuple(tuple(start for (start, _), _ in chunk) for chunk in chunks),
        tuple(tuple(local_start for _, local_start in chunk) for chunk in chunks),
        tuple(tuple(length for (_, length), _ in chunk) for chunk in chunks),
    )


def _shard_runs(
    shard_dim_runs: list[tuple[IntLikeType, IntLikeType]],
    mesh_dim_size: int,
    rank: IntLikeType,
    placement: Shard | _StridedShard,
) -> list[tuple[IntLikeType, IntLikeType]]:
    """
    The global indices left on ``rank`` along the tensor dim that ``placement``
    shards. ``shard_dim_runs`` is the global indices the local tensor holds
    along that dim so far, in local order, run-length encoded as
    ``(global_start, length)``: ``[(0, 2), (3, 2)]`` stands for 0, 1, 3, 4.
    Returns them encoded the same way.
    """
    size = sum(length for _, length in shard_dim_runs)
    split_factor = (
        int(placement.split_factor) if isinstance(placement, _StridedShard) else 1
    )
    selected: list[tuple[IntLikeType, IntLikeType]] = []
    # TODO: share this two-level split with _StridedShard.local_shard_size_and_offset.
    for group in range(split_factor):
        group_size, group_start = Shard.local_shard_size_and_offset(
            size, split_factor, group
        )
        piece_size, piece_start = Shard.local_shard_size_and_offset(
            group_size, mesh_dim_size, rank
        )
        start = group_start + piece_start
        selected.extend(_select_from_runs(shard_dim_runs, start, start + piece_size))
    return selected


def _select_from_runs(
    runs: list[tuple[IntLikeType, IntLikeType]],
    start: IntLikeType,
    stop: IntLikeType,
) -> list[tuple[IntLikeType, IntLikeType]]:
    """
    The part of ``runs`` at positions ``start`` to ``stop - 1`` of the local
    tensor, as runs of global indices: one for each of ``runs``, cut down to what
    falls in that range and possibly empty, so the structure doesn't depend on
    ``start`` or ``stop``, and nothing here branches on them.
    """
    selected: list[tuple[IntLikeType, IntLikeType]] = []
    position: IntLikeType = 0  # position of the run's first index in the local tensor
    for global_start, length in runs:
        first = _clamp(start - position, 0, length)
        last = _clamp(stop - position, 0, length)
        selected.append((global_start + first, last - first))
        position = position + length
    return selected


def _clamp(x: IntLikeType, lo: IntLikeType, hi: IntLikeType) -> IntLikeType:
    return torch.sym_min(torch.sym_max(x, lo), hi)
