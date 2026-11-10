# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Resharding for tensors described by ``CheckpointableTensor`` annotations.

``CheckpointableTensorResharder`` records every tensor it can reshard as
``CheckpointableTensorShardingMetadata``: the shards of a local tensor, each a
block of the logical global tensor. A tensor implementing
``CheckpointableTensor`` describes its shards itself; a ``DTensor``'s shards are
derived from its placements. Loading needs nothing else: each target shard is
filled from the saved shards that overlap it.

This resharder is experimental. For loads of ``DTensor`` state,
``DefaultResharder`` is still recommended.
"""

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from typing_extensions import override

from .checkpoint_layout import SafetensorsSerialization
from .checkpointable import CheckpointableTensor
from .checkpointable_tensor_metadata import CheckpointableTensorShardingMetadata
from .distributed_metadata import DistributedItemMetadata, ShardingMetadata
from .resharding import LoadPlan, Resharder, ReshardingInfo
from .resharding_utils import get_fqn_from_nested_path
from .types import CheckpointPath, NestedPath
from .walk_utils import walk_checkpoint_structure

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["CheckpointableTensorResharder"]

# A block of a global tensor: the indices [start, stop) along each dimension.
_Box = tuple[tuple[int, int], ...]
# A saved shard: the rank to read it from, its block of the global tensor, and
# where that block starts in the rank's saved local tensor.
_SourceShard = tuple[int, _Box, tuple[int, ...]]


def _shards(
    metadata: CheckpointableTensorShardingMetadata,
) -> list[tuple[_Box, tuple[int, ...]]]:
    """Each non-empty shard's block of the global tensor and its local offset."""
    return [
        (
            tuple((start, start + size) for start, size in zip(global_offset, sizes)),
            local_offset,
        )
        for global_offset, local_offset, sizes in zip(
            metadata.global_offsets, metadata.local_offsets, metadata.local_sizes
        )
        if all(sizes)
    ]


def _intersect(a: _Box, b: _Box) -> _Box | None:
    """The block that ``a`` and ``b`` share, or None if they do not overlap."""
    box = tuple((max(a0, b0), min(a1, b1)) for (a0, a1), (b0, b1) in zip(a, b))
    return box if all(start < stop for start, stop in box) else None


def _subtract(box: _Box, cut: _Box) -> list[_Box]:
    """Disjoint blocks covering the part of ``box`` outside ``cut``, which overlaps it."""
    pieces: list[_Box] = []
    rest = list(box)
    for dim, ((start, stop), (cut_start, cut_stop)) in enumerate(zip(box, cut)):
        if start < cut_start:
            pieces.append((*rest[:dim], (start, cut_start), *rest[dim + 1 :]))
        if cut_stop < stop:
            pieces.append((*rest[:dim], (cut_stop, stop), *rest[dim + 1 :]))
        rest[dim] = (max(start, cut_start), min(stop, cut_stop))
    return pieces


def _to_local(
    starts: tuple[int, ...], box: _Box, local_offset: tuple[int, ...]
) -> tuple[int, ...]:
    """Where global indices ``starts`` are in the local tensor of a shard.

    The shard holds ``box`` of the global tensor, from ``local_offset`` on.
    """
    return tuple(
        start - box_start + offset
        for start, (box_start, _), offset in zip(starts, box, local_offset)
    )


def _plan_shard(
    fqn: str,
    target_box: _Box,
    target_offset: tuple[int, ...],
    sources: list[_SourceShard],
) -> list[LoadPlan] | None:
    """Plans that fill one target shard, or None if part of it is in no source.

    Each part is read from the first source shard in ``sources`` that holds it:
    greedily, each overlap is planned and cut out of what remains. Every element
    is read once, but the plans need not be the fewest. When along each
    dimension any two saved shards are equal or disjoint, as for a ``DTensor``'s
    shards, the part of each saved shard inside the target falls within one
    remaining block, so each saved shard the target needs gets exactly one plan.
    Saved shards that overlap partly can split the remainder into more plans
    than needed: more, smaller reads of the same bytes. Finding the fewest plans
    in general is a set cover problem, not worth solving unless such layouts
    turn out to matter.
    """
    plans: list[LoadPlan] = []
    uncovered = [target_box]
    for rank, source_box, source_offset in sources:
        remaining: list[_Box] = []
        for box in uncovered:
            overlap = _intersect(box, source_box)
            if overlap is None:
                remaining.append(box)
                continue
            starts = tuple(start for start, _ in overlap)
            sizes = tuple(stop - start for start, stop in overlap)
            plans.append(
                LoadPlan(
                    offsets=_to_local(starts, target_box, target_offset),
                    sizes=sizes,
                    src_rank=rank,
                    src_fqn=fqn,
                    src_offsets=_to_local(starts, source_box, source_offset),
                    src_sizes=sizes,
                )
            )
            remaining.extend(_subtract(box, source_box))
        uncovered = remaining
        if not uncovered:
            return plans
    return None


class CheckpointableTensorResharder(Resharder):
    """Resharder that records tensors as ``CheckpointableTensorShardingMetadata``.

    Experimental: for state made of ``DTensor``\\ s, ``DefaultResharder`` is
    still preferred.

    It reshards two kinds of tensors:

    - Tensors implementing ``CheckpointableTensor``, whose annotations describe
      their shards.
    - ``DTensor``\\ s, whose annotations it derives from their placements.

    Any other tensor gets no metadata. Unlike ``DefaultResharder``, it does not
    assume such a tensor is replicated: a load that reshards the tensor's item
    reports it in ``ReshardingInfo.non_reshardable_paths`` instead of loading
    it. When nothing in an item needs resharding, the reader loads the whole
    item directly, these tensors included.

    Each part of a target shard is read from the lowest source rank that saved
    it, so replicated and overlapping saved shards are each read once. A path is
    planned only if the saved shards cover every one of its target shards; plans
    are plain copies, as ``DefaultLoadPlanExecutor`` requires.
    """

    @override
    def extract_sharding_metadata(
        self,
        item_key: str,
        item_value: Any,
    ) -> dict[NestedPath, ShardingMetadata]:
        """Record each ``DTensor`` and ``CheckpointableTensor`` leaf of the item.

        Args:
            item_key: The checkpoint item key (e.g., "model", "optimizer").
            item_value: The item's value (e.g., state_dict).

        Returns:
            ``CheckpointableTensorShardingMetadata`` for each such tensor, keyed
            by its NestedPath.
        """
        result: dict[NestedPath, ShardingMetadata] = {}
        unannotated_paths: list[NestedPath] = []

        def _collect(path: CheckpointPath, obj: Any, _: Any) -> None:
            if isinstance(obj, DTensor):
                result[path.nested_path] = (
                    CheckpointableTensorShardingMetadata.from_dtensor(obj)
                )
            elif isinstance(obj, torch.Tensor):
                if isinstance(obj, CheckpointableTensor):
                    result[path.nested_path] = (
                        CheckpointableTensorShardingMetadata.from_tensor(obj)
                    )
                else:
                    unannotated_paths.append(path.nested_path)

        walk_checkpoint_structure(
            item_key=item_key,
            source=item_value,
            target=None,
            leaf_fn=_collect,
        )
        if unannotated_paths:
            logger.warning(
                "Found %s tensors in checkpoint item %r that are neither DTensors "
                "nor CheckpointableTensors; they cannot be resharded. "
                "sample_paths=%s",
                len(unannotated_paths),
                item_key,
                unannotated_paths[:10],
            )
        return result

    @override
    def should_reshard(
        self,
        source_metadata: DistributedItemMetadata | None,
        target_metadata: dict[NestedPath, ShardingMetadata] | None,
    ) -> bool:
        """Reshard unless this rank saved each target tensor's current shards.

        The metadata describes one rank's local tensor, so a target matches only
        what this rank saved: another rank that saved equal metadata wrote a
        different file. Safetensors sources are always resharded.
        """
        if source_metadata is None or target_metadata is None:
            return False
        # Safetensors stores bare tensors, dropping a DTensor's wrapper, so a
        # direct read cannot place the local tensor; planned loads can.
        if any(
            layout is not None
            and isinstance(layout.serialization_format, SafetensorsSerialization)
            for layout in source_metadata.rank_to_layout_info.values()
        ):
            return True
        rank = dist.get_rank() if dist.is_initialized() else 0
        return any(
            nested_path in source_metadata.nested_path_to_metadata
            and source_metadata.get_metadata_for_path_and_rank(nested_path, rank)
            != target_sharding
            for nested_path, target_sharding in target_metadata.items()
        )

    @override
    def generate_load_plans(
        self,
        item_key: str,
        target: Any,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
    ) -> ReshardingInfo:
        """Plan which saved shards fill each target tensor, without reading data.

        Args:
            item_key: The checkpoint item key being loaded (e.g., "model").
            target: The item's live value, which the planned chunks are written into.
            target_metadata: This rank's target sharding from extract_sharding_metadata.
            source_metadata: Source checkpoint's distributed metadata for this item.

        Returns:
            ReshardingInfo binding each planned path to its target tensor. Tensor
            leaves without ``CheckpointableTensorShardingMetadata``, and paths
            the source cannot fill, are non-reshardable.
        """
        targets = self.resolve_targets(item_key, target)
        load_plans: dict[NestedPath, list[LoadPlan]] = {}
        non_reshardable_paths: list[NestedPath] = []
        for nested_path in dict.fromkeys([*targets, *target_metadata]):
            target_sharding = target_metadata.get(nested_path)
            plans = (
                self._plan_path(nested_path, target_sharding, source_metadata)
                if nested_path in targets
                and isinstance(target_sharding, CheckpointableTensorShardingMetadata)
                else None
            )
            if plans is None:
                non_reshardable_paths.append(nested_path)
            elif plans:
                load_plans[nested_path] = plans
        return ReshardingInfo(
            nested_path_to_load_plans=load_plans,
            non_reshardable_paths=non_reshardable_paths,
            nested_path_to_target_tensors={path: targets[path] for path in load_plans},
        )

    def _plan_path(
        self,
        nested_path: NestedPath,
        target_sharding: CheckpointableTensorShardingMetadata,
        source_metadata: DistributedItemMetadata,
    ) -> list[LoadPlan] | None:
        """Plans that fill the target at ``nested_path``, or None if none can."""
        groups = source_metadata.nested_path_to_metadata.get(nested_path)
        if groups is None:
            logger.warning("Missing source metadata for path: %s", nested_path)
            return None

        sources: list[_SourceShard] = []
        # Every rank of a group saved the same shards: read them from the lowest.
        for group in sorted(groups, key=lambda group: min(group.ranks)):
            saved = group.sharding_metadata
            if (
                not isinstance(saved, CheckpointableTensorShardingMetadata)
                or saved.global_shape != target_sharding.global_shape
            ):
                logger.warning(
                    "Source metadata %s for path %s cannot fill a target of "
                    "global shape %s",
                    type(saved).__name__,
                    nested_path,
                    target_sharding.global_shape,
                )
                return None
            rank = min(group.ranks)
            sources.extend((rank, box, offset) for box, offset in _shards(saved))

        fqn = get_fqn_from_nested_path(nested_path)
        plans: list[LoadPlan] = []
        for box, offset in _shards(target_sharding):
            shard_plans = _plan_shard(fqn, box, offset, sources)
            if shard_plans is None:
                logger.warning(
                    "Saved shards of path %s do not cover its target block %s",
                    nested_path,
                    box,
                )
                return None
            plans.extend(shard_plans)
        return plans
