"""
Factory functions for creating checkpoint saver instances with sensible defaults.

This module provides high-level factory functions that simplify the creation
of checkpoint saver instances by automatically handling component initialization
and configuration with reasonable defaults.
"""

from typing import Any, Callable

import torch.distributed as dist

from .checkpoint_process import CheckpointProcess
from .checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterArgs,
)
from .checkpointer import AsyncCheckpointSaver, SyncCheckpointSaver
from .config import AsyncCheckpointSaverConfig, SyncCheckpointSaverConfig
from .logging_utils import EventLogger
from .metadata_manager import MetadataManager
from .staging import DefaultStager
from .storage.base_storage import StorageConfig
from .storage.filesystem import LocalFileSystemStorageConfig
from .types import RankInfo


def _get_default_rank_info() -> RankInfo:
    """
    Get default rank information from the current distributed environment.

    Returns:
        RankInfo: Rank information from the default process group if initialized,
                 otherwise single-rank fallback.
    """
    if dist.is_initialized():
        return RankInfo(
            global_world_size=dist.get_world_size(),
            global_rank=dist.get_rank(),
            role_rank=dist.get_rank(),
            role_world_size=dist.get_world_size(),
        )
    else:
        return RankInfo(
            global_world_size=1,
            global_rank=0,
            role_rank=0,
            role_world_size=1,
        )


def _default_subprocess_init(*args: Any) -> None:
    """Default subprocess initialization function that does nothing.

    This is a module-level function that can be pickled, unlike lambda functions.
    """
    pass


def make_sync_checkpoint_saver(
    config: SyncCheckpointSaverConfig | None = None,
    rank_info: RankInfo | None = None,
    pre_finalize_callback: Callable[[str, EventLogger], None] | None = None,
    finalize_callback: Callable[[str, EventLogger], None] | None = None,
    storage_config: StorageConfig | None = None,
    checkpoint_metadata_manager: MetadataManager | None = None,
) -> SyncCheckpointSaver:
    """
    Factory function to create a SyncCheckpointSaver instance with sensible defaults.

    This function creates a synchronous checkpoint saver with default components, automatically
    detecting rank information from the default process group if available, and using the
    provided component configurations.

    Args:
        config: SyncCheckpointSaverConfig containing writer configuration.
            Defaults to a new SyncCheckpointSaverConfig().
        rank_info: RankInfo for distributed training. Defaults to auto-detection from
                  the default PyTorch distributed process group if initialized, otherwise
                  falls back to single-rank (world_size=1, rank=0).
        pre_finalize_callback: Optional callback for custom actions before checkpoint finalization.
            Called after files are written but before barrier synchronization.
        finalize_callback: Optional callback for custom actions after checkpoint finalization.
            Called after barrier synchronization (all ranks coordinated).
        storage_config: StorageConfig for storage backend. If None, defaults to LocalFileSystemStorage.
        checkpoint_metadata_manager: Optional MetadataManager for managing distributed
            checkpoint metadata. Handles extraction, aggregation, and validation of sharding
            metadata across ranks for proper distributed checkpointing. If None, no metadata
            management steps are performed.

    Returns:
        SyncCheckpointSaver: A configured synchronous checkpoint saver instance.

    Examples:
        # Simplest usage - auto-detect rank, default config
        saver = make_sync_checkpoint_saver()

        # Explicit rank configuration
        saver = make_sync_checkpoint_saver(
            rank_info=RankInfo(global_world_size=4, global_rank=0, role_rank=0, role_world_size=4)
        )

        # Custom callbacks
        saver = make_sync_checkpoint_saver(
            pre_finalize_callback=lambda path, event_logger: validate_files(path),
            finalize_callback=lambda path, event_logger: logger.info(f"Checkpoint done: {path}"),
        )

        # Disable barrier
        config = SyncCheckpointSaverConfig(
            writer_config=CheckpointWriterConfig(
                barrier_config=BarrierConfig(barrier_type=None)
            )
        )
        saver = make_sync_checkpoint_saver(config=config)
    """
    if config is None:
        config = SyncCheckpointSaverConfig()

    if rank_info is None:
        rank_info = _get_default_rank_info()

    storage_config = storage_config or LocalFileSystemStorageConfig()

    args = CheckpointWriterArgs(
        config=config.writer_config,
        rank_info=rank_info,
        storage_config=storage_config,
        pre_finalize_callback=pre_finalize_callback,
        finalize_callback=finalize_callback,
    )
    writer = CheckpointWriter(args=args)

    return SyncCheckpointSaver(
        writer=writer,
        metadata_manager=checkpoint_metadata_manager,
    )


def make_async_checkpoint_saver(
    config: AsyncCheckpointSaverConfig | None = None,
    rank_info: RankInfo | None = None,
    pre_finalize_callback: Callable[[str, EventLogger], None] | None = None,
    finalize_callback: Callable[[str, EventLogger], None] | None = None,
    subprocess_init_fn: Callable[..., None] | None = None,
    subprocess_init_args: tuple[Any, ...] = (),
    storage_config: StorageConfig | None = None,
    checkpoint_metadata_manager: MetadataManager | None = None,
) -> AsyncCheckpointSaver:
    """
    Factory function to create an AsyncCheckpointSaver instance with sensible defaults.

    This function creates an asynchronous checkpoint saver using the provided configuration,
    automatically detecting rank information if not provided.

    Args:
        config: AsyncCheckpointSaverConfig containing all async-specific configurations
                (writer_config, staging_config, process_config).
                Defaults to a new AsyncCheckpointSaverConfig().
        rank_info: RankInfo for distributed training. Defaults to auto-detection from
                  the default PyTorch distributed process group if initialized, otherwise
                  falls back to single-rank (world_size=1, rank=0).
        pre_finalize_callback: Optional callback for custom actions before checkpoint finalization.
                              Called after files are written but before barrier synchronization.
        finalize_callback: Optional callback for custom actions after checkpoint finalization.
                          Called after barrier synchronization (all ranks coordinated).
        subprocess_init_fn: Function to initialize the subprocess. Defaults to no-op.
                           Must be picklable for subprocess communication.
        subprocess_init_args: Arguments to pass to subprocess_init_fn.
        storage_config: Storage backend configuration (defaults to local filesystem).
        checkpoint_metadata_manager: Optional MetadataManager instance for managing distributed
                                    checkpoint metadata. Handles extraction, aggregation, and
                                    validation of sharding metadata across ranks for proper
                                    distributed checkpointing. If None, no metadata management
                                    steps are performed. When shared with a CheckpointLoader,
                                    metadata computed during load will be reused by save.

    Returns:
        AsyncCheckpointSaver: A configured asynchronous checkpoint saver instance.

    Examples:
        # Create with default config
        saver = make_async_checkpoint_saver()

        # Create with custom callbacks
        saver = make_async_checkpoint_saver(
            pre_finalize_callback=lambda path, event_logger: validate_files(path),
            finalize_callback=lambda path, event_logger: logger.info(f"Checkpoint done: {path}"),
        )

        # Create with custom subprocess init
        saver = make_async_checkpoint_saver(
            subprocess_init_fn=my_subprocess_init_fn,
            subprocess_init_args=("my", "args"),
        )

        # Create with shared metadata manager for load-then-save workflows
        metadata_manager = CachingMetadataManager(...)
        reader = CheckpointReader(rank_info=rank_info, storage_config=storage_config)
        loader = CheckpointLoader(reader=reader, metadata_manager=metadata_manager)
        loader.load(path, checkpoint)
        loader.close()

        saver = make_async_checkpoint_saver(
            checkpoint_metadata_manager=metadata_manager,  # Shared with loader
        )
        saver.save(new_path, checkpoint)  # Reuses serialized metadata from load
    """
    if config is None:
        config = AsyncCheckpointSaverConfig()

    if rank_info is None:
        rank_info = _get_default_rank_info()

    storage_config = storage_config or LocalFileSystemStorageConfig()

    checkpoint_stager = DefaultStager(
        config=config.staging_config,
    )

    # Create checkpoint writer args
    checkpoint_writer_args = CheckpointWriterArgs(
        config=config.writer_config,
        rank_info=rank_info,
        storage_config=storage_config,
        pre_finalize_callback=pre_finalize_callback,
        finalize_callback=finalize_callback,
    )

    checkpoint_process = CheckpointProcess(
        rank_info=rank_info,
        config=config.process_config,
        subprocess_init_fn=subprocess_init_fn or _default_subprocess_init,
        subprocess_init_args=subprocess_init_args,
        checkpoint_writer_args=checkpoint_writer_args,
    )

    return AsyncCheckpointSaver(
        checkpoint_stager=checkpoint_stager,
        checkpoint_process=checkpoint_process,
        metadata_manager=checkpoint_metadata_manager,
    )
