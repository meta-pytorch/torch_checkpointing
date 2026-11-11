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

from .barriers import (  # noqa: F401
    Barrier,
    BarrierConfig,
    DefaultStoreBarrier,
    DefaultStoreBarrierConfig,
    TCPStoreBarrier,
    TCPStoreBarrierConfig,
)
from .builder import (  # noqa: F401
    make_async_checkpoint_saver,
    make_sync_checkpoint_saver,
)
from .checkpoint_base import CheckpointBase, CheckpointItem  # noqa: F401
from .checkpoint_loader import CheckpointLoader  # noqa: F401
from .checkpoint_manager import CheckpointManager
from .checkpoint_reader import CheckpointReader  # noqa: F401
from .checkpoint_saver import (  # noqa: F401
    AsyncCheckpointSaver,
    CheckpointSaver,
    SyncCheckpointSaver,
)
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig  # noqa: F401
from .checkpointable_tensor_metadata import (  # noqa: F401
    CheckpointableTensorShardingMetadata,
)
from .config import (  # noqa: F401
    AsyncCheckpointSaverConfig,
    CheckpointLoaderConfig,
    CheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from .distributed_metadata import (  # noqa: F401
    CheckpointMetadata,
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
    ItemMetadata,
    ShardingMetadata,
)
from .dtensor_metadata import (  # noqa: F401
    _PlacementSpec,
    DeviceMeshSpec,
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
    StridedShardSpec,
)
from .load_plan_executor import LoadExecutionContext, LoadPlanExecutor  # noqa: F401
from .metadata_manager import DefaultMetadataManager, MetadataManager  # noqa: F401
from .resharding import LoadPlan, Resharder, ReshardingInfo  # noqa: F401
from .schema import ItemSpec
from .staging import (  # noqa: F401
    CheckpointStager,
    CheckpointStagerConfig,
    DefaultStager,
)
from .state_transformations import (  # noqa: F401
    optimizer_transform_post,
    optimizer_transform_pre,
)
from .types import RankInfo, STATE_DICT  # noqa: F401
from .utils import wrap_future  # noqa: F401
from .version import __version__, get_version, Version

# Public API. `CheckpointManager` -- plus `ItemSpec` for per-item overrides -- is
# the entire surface a typical user needs (see the tutorial in docs/). Every other
# symbol imported above stays importable from this package and from its own
# submodule for advanced use and backward compatibility, but is intentionally not
# advertised as part of the public API.
__all__ = [
    "CheckpointManager",
    "ItemSpec",
    "StridedShardSpec",
    "__version__",
    "get_version",
    "Version",
]
