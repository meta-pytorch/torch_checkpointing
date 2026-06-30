"""
Checkpoint functionality for machine learning models.

This module provides classes for saving and loading model checkpoints in a distributed
training environment. It includes functionality for coordinating checkpoint operations
across multiple processes and customizing the checkpoint process through hooks.

Key components:
- CheckpointSaver: Main class for orchestrating checkpoint save operations
- CheckpointWriter: Handles writing state dictionaries to storage
- CheckpointReader: Handles reading state dictionaries from storage read
- Barrier: Synchronization mechanism for distributed checkpointing
- RankInfo: Information about the current rank in a distributed environment

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
from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter, CheckpointWriterConfig
from .checkpointer import AsyncCheckpointSaver, CheckpointSaver, SyncCheckpointSaver
from .config import (
    AsyncCheckpointSaverConfig,
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
from .staging import CheckpointStager, CheckpointStagerConfig, DefaultStager
from .state_transformations import (
    optimizer_transform_post,
    optimizer_transform_pre,
)
from .types import RankInfo, STATE_DICT
from .utils import wrap_future
from .version import __version__, get_version, Version

__all__ = [
    "Barrier",
    "TCPStoreBarrier",
    "CheckpointReader",
    "CheckpointWriter",
    "CheckpointWriterConfig",
    "CheckpointBase",
    "CheckpointItem",
    "CheckpointLoader",
    "CheckpointSaver",
    "SyncCheckpointSaver",
    "AsyncCheckpointSaver",
    "CheckpointSaverConfig",
    "SyncCheckpointSaverConfig",
    "AsyncCheckpointSaverConfig",
    "BarrierConfig",
    "TCPStoreBarrierConfig",
    "CheckpointStager",
    "CheckpointStagerConfig",
    "DefaultStager",
    "RankInfo",
    "STATE_DICT",
    "wrap_future",
    "make_sync_checkpoint_saver",
    "make_async_checkpoint_saver",
    # Distributed metadata components
    "CheckpointMetadata",
    "DistributedItemMetadata",
    "DistributedMetadata",
    "GlobalObjectMetadata",
    "ItemMetadata",
    "ShardingMetadata",
    "DTensorShardingMetadata",
    "DeviceMeshSpec",
    "get_device_mesh_spec",
    "_PlacementSpec",
    "ReplicateSpec",
    "ShardSpec",
    "MetadataManager",
    "DefaultMetadataManager",
    # Resharding
    "LoadPlan",
    "Resharder",
    "ReshardingInfo",
    # State transformations
    "optimizer_transform_pre",
    "optimizer_transform_post",
    # Versioning
    "__version__",
    "get_version",
    "Version",
]
