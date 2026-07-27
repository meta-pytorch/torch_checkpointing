# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Asynchronous, distributed checkpointing for PyTorch.

``CheckpointManager`` is the entry point: it saves and loads plain
``{item_key: value}`` payloads, staging state off the training device and writing
it from a background process, and reshards on load across different distributed
layouts. See the ``docs/`` directory for the guide and API reference.
"""

from .barriers import (
    Barrier,
    BarrierConfig,
    TCPStoreBarrier,
    TCPStoreBarrierConfig,
)
from .builder import (
    make_async_checkpoint_saver,
    make_sync_checkpoint_saver,
)
from .checkpoint_base import CheckpointBase, CheckpointItem
from .checkpoint_loader import CheckpointLoader
from .checkpoint_manager import CheckpointManager
from .checkpoint_reader import CheckpointReader
from .checkpoint_saver import AsyncCheckpointSaver, CheckpointSaver, SyncCheckpointSaver
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig
from .config import (
    AsyncCheckpointSaverConfig,
    CheckpointLoaderConfig,
    CheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from .distributed_metadata import (
    CheckpointMetadata,
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
    ItemMetadata,
    ShardingMetadata,
)
from .dtensor_metadata import (
    _PlacementSpec,
    DeviceMeshSpec,
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
)
from .metadata_manager import DefaultMetadataManager, MetadataManager
from .resharding import LoadPlan, Resharder, ReshardingInfo
from .schema import ItemSpec
from .staging import CheckpointStager, CheckpointStagerConfig, DefaultStager
from .state_transformations import (
    optimizer_transform_post,
    optimizer_transform_pre,
)
from .types import RankInfo, STATE_DICT
from .utils import wrap_future
from .version import __version__, get_version, Version

# Public API. `CheckpointManager` -- plus `ItemSpec` for per-item overrides -- is
# the entire surface a typical user needs (see the tutorial in docs/). Every other
# symbol imported above stays importable from this package and from its own
# submodule for advanced use and backward compatibility, but is intentionally not
# advertised as part of the public API.
__all__ = [
    "CheckpointManager",
    "ItemSpec",
    "__version__",
    "get_version",
    "Version",
]
