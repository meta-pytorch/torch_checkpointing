# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The interface for executing load plans.

A ``Resharder`` decides what to load: which chunks of which saved tensors fill
each target tensor. A ``LoadPlanExecutor`` decides how to load them: how files
are read, in what order and with what concurrency. ``ReshardingInfo``, the load
plans of one item bound to the tensors they fill, is the boundary between them.
"""

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .distributed_metadata import DistributedItemMetadata
from .resharding import ReshardingInfo
from .storage.base_storage import Storage
from .types import ItemKey


@dataclass(frozen=True)
class LoadExecutionContext:
    """Where a load reads its planned source data from.

    Attributes:
        checkpoint_path: The checkpoint directory. File paths in the source
            metadata are relative to it.
        storage: The storage backend to read through, scoped to this load.
        source_metadata: The saved metadata of each item being loaded, keyed by
            item key. ``source_metadata[item_key].get_layout_info(src_rank,
            item_key)`` locates the file a load plan reads, and its
            serialization format.
    """

    checkpoint_path: Path
    storage: Storage
    source_metadata: Mapping[ItemKey, DistributedItemMetadata]


class LoadPlanExecutor(abc.ABC):
    """Reads the data that load plans describe into their target tensors.

    Implement this to change how a checkpoint is read, for example to prefetch,
    or to schedule and combine reads across items, independently of the
    resharders that plan what is read.
    """

    @abc.abstractmethod
    def execute(
        self,
        items: Mapping[ItemKey, ReshardingInfo],
        context: LoadExecutionContext,
    ) -> None:
        """Fill the target tensors of every item with its planned source data.

        Return only when no work started by this call can still write a target:
        CPU targets are filled, and copies into accelerator targets are queued on
        the caller's current stream. Raise on failure, after which targets may be
        partly written.

        Treat ``items`` as read-only. Different items may write the same
        storage, so do not write two items concurrently unless their targets
        are disjoint.

        The reader calls this once per read with the items that reshard on this
        rank, and not at all when none does, for example when every layout
        matches the checkpoint. Ranks can therefore differ in whether they call
        it: do not rely on every rank calling it.

        Args:
            items: The bound load plans of each item, keyed by item key, in the
                order the items should be loaded.
            context: Where to read the source data from.
        """
        ...
