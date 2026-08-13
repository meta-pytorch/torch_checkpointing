# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint writer functionality for machine learning models.

This module provides classes for writing checkpoints to storage, including
determining checkpoint layout, configuring the writer, and defining hooks
for custom actions during the checkpoint writing process.
"""

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .barriers import Barrier, BarrierConfig
from .checkpoint_base import CheckpointWriteInfo
from .checkpoint_layout import (
    default_layout_info,
    JsonSerialization,
    LayoutInfo,
    RawSerialization,
    SafetensorsSerialization,
    TorchSerialization,
)
from .distributed_metadata import METADATA_FILE_NAME
from .logging_utils import EventLogger, EventType, get_log_event_type_for_file_save
from .storage.base_storage import StorageConfig
from .types import RankInfo

logger = logging.getLogger(__name__)


@dataclass
class CheckpointWriterConfig:
    """
    Configuration options for the CheckpointWriter that application users likely will want to provide.

    Attributes:
        checkpoint_write_barrier_timeout_sec: Maximum time in seconds to wait for all ranks
            to reach the checkpoint barrier before timing out. Default is 600 seconds.
        barrier_config: Complete configuration for the synchronization barrier. If None,
            no barrier will be used. Must contain all necessary common and barrier-specific
            fields if provided.
        file_write_max_threads: Maximum workers for independent file writes and
            parent-directory creation.
    """

    checkpoint_write_barrier_timeout_sec: int = 600
    barrier_config: BarrierConfig | None = None
    file_write_max_threads: int = 1

    def __post_init__(self) -> None:
        if self.file_write_max_threads < 1:
            raise ValueError("file_write_max_threads must be positive")


@dataclass
class CheckpointWriterArgs:
    """All input arguments to checkpoint writer.

    Args:
        config: Configuration options for the checkpoint writer.
        rank_info: Information about the current rank in a distributed setting.
        pre_finalize_callback: Optional callback for custom actions before checkpoint finalization.
                With a barrier, it receives the complete temporary checkpoint path.
                Distributed callbacks must synchronize before rank zero returns.
        finalize_callback: Optional callback for custom actions after checkpoint finalization.
                Called after barrier synchronization (all ranks coordinated).
        storage_config: StorageConfig backend to use for I/O operations. If None, defaults to LocalFileSystemStorageConfig
                with directio enabled.
    """

    config: CheckpointWriterConfig
    rank_info: RankInfo
    storage_config: StorageConfig
    pre_finalize_callback: Callable[[str, EventLogger], None] | None = None
    finalize_callback: Callable[[str, EventLogger], None] | None = None
    metric_prefix: str = "train.checkpoint_write"

    def build(self) -> "CheckpointWriter":
        """Create the checkpoint writer for this args configuration.

        Subclasses can override this to return custom writers while reusing
        the subprocess infrastructure.
        """
        return CheckpointWriter(args=self)


class CheckpointWriter:
    """
    Handles writing state dictionaries to storage.

    This class is responsible for writing model state dictionaries to storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    TMP_PREFIX: str = "tmp_"

    def __init__(self, args: CheckpointWriterArgs):
        """
        Initialize a CheckpointWriter.

        """
        self._args = args
        self._metric_prefix = args.metric_prefix
        self._storage = args.storage_config.create_storage()
        event_logger = EventLogger()

        # Create barrier from config
        self._barrier: Barrier | None = None
        if args.config.barrier_config is not None:
            logger.info(
                f"Starting: initializing barrier of checkpointing sub-process for rank {args.rank_info.global_rank}/{args.rank_info.global_world_size} "
                f"with config: {args.config.barrier_config}",
                extra=event_logger(EventType.CHECKPOINT_SUBPROCESS_BARRIER_START),
            )

            # The barrier_config should already be complete, just use it directly
            self._barrier = args.config.barrier_config.create_barrier(args.rank_info)

            logger.info(
                f"Done: initialized barrier of checkpointing sub-process for rank {type(self._barrier).__name__} "
                f"for rank {args.rank_info.global_rank}",
                extra=event_logger(
                    EventType.CHECKPOINT_SUBPROCESS_BARRIER_END,
                    metric_name=f"{self._metric_prefix}.execute.subprocess_execute.initialize_barrier.latency_ms",
                    end_to_end=True,
                ),
            )

    def write(
        self,
        path: str,
        checkpoint_info: CheckpointWriteInfo,
    ) -> Future[None] | None:
        """
        Writes the checkpoint_info to storage.

        Args:
            path (str): The path to write the checkpoint to.
            checkpoint_info (CheckpointWriteInfo): Encapsulates state_dict, layout_info_mappings,
                and optional serialized_distributed_metadata.

        Returns:
            Optional[Future[None]]: A future for tracking the write operation, if applicable.
        """
        event_logger = EventLogger()
        logger.debug(
            f"Writing checkpoint to {path} for rank {self._args.rank_info.global_rank}"
        )

        final_path = Path(path)
        # tmp path is only safe to use if we have configured a barrier
        if self._barrier is None:
            tmp_dir_path = final_path
        else:
            tmp_dir_path = final_path.parent / f"{self.TMP_PREFIX}{final_path.name}"

        state_dict = checkpoint_info.state_dict
        save_items: list[tuple[str, LayoutInfo, Path]] = []
        for key, layout_info in checkpoint_info.layout_info_mappings.items():
            # Skip keys that are in layout but not in state_dict
            if key not in state_dict:
                continue

            if layout_info is None:
                layout_info = default_layout_info(key, self._args.rank_info.global_rank)
            save_items.append((key, layout_info, tmp_dir_path / layout_info.file_path))

        self._write_keys(save_items, state_dict)

        # Write metadata directly from serialized string if available (rank0 only)
        if self._args.rank_info.role_rank == 0:
            logger.info(
                "Writing metadata to checkpoint directory",
            )
            self._write_metadata(
                tmp_dir_path,
                checkpoint_info.serialized_distributed_metadata,
            )
            logger.info(
                "Done writing metadata to checkpoint directory",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"{self._metric_prefix}.execute.filesystem.metadata.write.latency_ms",
                ),
            )

        logger.info(
            "Finished all filesystem writes",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.execute.filesystem.save.latency_ms",
                end_to_end=True,
            ),
        )

        logger.info(
            "Successfully saved checkpoint",
            extra=event_logger(EventType.CHECKPOINT_SAVED_TMP),
        )

        # Wait for all ranks to finish writing if barrier is available
        if self._barrier is not None:
            logger.info(
                f"Waiting for all ranks at barrier with timeout {self._args.config.checkpoint_write_barrier_timeout_sec}s",
                extra=event_logger(EventType.CHECKPOINT_BARRIER_START),
            )
            self._barrier.execute_barrier(
                self._args.config.checkpoint_write_barrier_timeout_sec,
            )
            logger.info(
                "All ranks passed barrier",
                extra=event_logger(
                    EventType.CHECKPOINT_BARRIER_END,
                    metric_name=f"{self._metric_prefix}.execute.final_barrier.latency_ms",
                ),
            )
        else:
            logger.info("No barrier configured, skipping synchronization")

        if self._args.pre_finalize_callback is not None:
            logger.debug(f"Executing pre-finalize callback for {tmp_dir_path}")
            self._args.pre_finalize_callback(str(tmp_dir_path), event_logger)

        # rename back to original path for atomicity. It's important this is done
        # immediately after the barrier to ensure all ranks have finished writing
        if (
            self._barrier is not None
            and self._storage.exists(tmp_dir_path)
            and self._args.rank_info.role_rank == 0
        ):
            logger.info("Moving checkpoint to final directory")
            self._storage.rename(tmp_dir_path, final_path, is_directory=True)
            logger.info(
                "Finished moving checkpoint to final directory",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"{self._metric_prefix}.execute.rename.latency_ms",
                ),
            )

        # Execute finalize callback if available
        if self._args.finalize_callback is not None:
            logger.debug(f"Executing finalize callback for {path}")
            self._args.finalize_callback(str(final_path), event_logger)

        logger.info(
            f"Successfully wrote checkpoint to {final_path} for rank {self._args.rank_info.global_rank}"
        )

        return None

    def _write_keys(
        self,
        save_items: list[tuple[str, LayoutInfo, Path]],
        state_dict: dict[str, Any],
    ) -> None:
        if not save_items:
            return

        write_threads = min(self._args.config.file_write_max_threads, len(save_items))
        assert write_threads >= 1

        with ThreadPoolExecutor(
            max_workers=write_threads,
            thread_name_prefix="ckpt-write",
        ) as pool:
            self._prepare_save_dirs(
                sorted({full_path.parent for _, _, full_path in save_items}),
                pool,
            )

            file_writes_event_logger = EventLogger()
            futures = [
                pool.submit(
                    self._write_key,
                    key,
                    layout_info,
                    full_path,
                    state_dict[key],
                )
                for key, layout_info, full_path in save_items
            ]
            for future in futures:
                future.result()

        logger.info(
            "Finished checkpoint file writes. "
            f"num_threads={write_threads} num_files={len(save_items)}",
            extra=file_writes_event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.execute.storage.file_writes.e2e.latency_ms",
                end_to_end=True,
            ),
        )

    def _prepare_save_dirs(
        self,
        parent_dirs: list[Path],
        pool: ThreadPoolExecutor,
    ) -> None:
        if not parent_dirs:
            return

        mkdir_event_logger = EventLogger()
        futures = [
            pool.submit(self._storage.mkdir, parent_dir) for parent_dir in parent_dirs
        ]
        for future in futures:
            future.result()
        logger.info(
            f"Prepared checkpoint folders. num_parent_dirs={len(parent_dirs)}",
            extra=mkdir_event_logger(
                EventType.LOG_METRIC,
                metric_name=f"{self._metric_prefix}.execute.storage.mkdir.latency_ms",
                end_to_end=True,
            ),
        )

    def _write_key(
        self,
        key: str,
        layout_info: LayoutInfo,
        full_path: Path,
        data_to_serialize: Any,
    ) -> None:
        event_logger = EventLogger()

        logger.info(
            f"Saving {key} to {full_path}.",
            extra=event_logger(get_log_event_type_for_file_save(key, True)),
        )
        if isinstance(layout_info.serialization_format, TorchSerialization):
            with self._storage.stream_write(full_path) as f:
                torch.save(data_to_serialize, f)  # type: ignore[arg-type]
        elif isinstance(layout_info.serialization_format, JsonSerialization):
            # For JSON, we need to serialize to string first, then encode to bytes
            json_str = json.dumps(
                data_to_serialize,
                sort_keys=True,
                indent=4,
            )
            self._storage.write(full_path, json_str.encode("utf-8"))
        elif isinstance(layout_info.serialization_format, RawSerialization):
            if not isinstance(data_to_serialize, bytes):
                raise ValueError(
                    f"RawSerialization requires bytes, but key '{key}' has type "
                    f"'{type(data_to_serialize).__name__}'. Please pass the serialized "
                    f"bytes directly to the checkpoint."
                )
            self._storage.write(full_path, data_to_serialize)
        elif isinstance(layout_info.serialization_format, SafetensorsSerialization):
            from safetensors.torch import save as safetensors_save

            flat_tensors = SafetensorsSerialization.prepare_tensors_for_save(
                data_to_serialize
            )
            sf_bytes = safetensors_save(
                flat_tensors,
                metadata=layout_info.serialization_format.metadata,
            )
            self._storage.write(full_path, sf_bytes)
        else:
            raise ValueError(
                f"Unsupported serialization format: {layout_info.serialization_format}"
            )

        logger.info(
            f"Saved {key}.",
            extra=event_logger(
                get_log_event_type_for_file_save(key, False),
                metric_name=f"{self._metric_prefix}.execute.storage.{key}.save_task.e2e.latency_ms",
                end_to_end=True,
            ),
        )

    def close(self) -> None:
        """
        Close the writer and release any resources.

        This is a no-op for the base CheckpointWriter but may be overridden
        by subclasses that need to perform cleanup.
        """
        logger.debug("Closing checkpoint writer")

    def _write_metadata(
        self, dir_path: Path, serialized_distributed_metadata: bytes | None = None
    ) -> None:
        """
        Write metadata to the checkpoint directory.

        Args:
            dir_path (Path): The path to the checkpoint directory.
            serialized_distributed_metadata (bytes | None): The pre-serialized distributed
                metadata pickle bytes to write directly to file.
        """
        if serialized_distributed_metadata:
            metadata_path = dir_path / METADATA_FILE_NAME
            self._storage.write(metadata_path, serialized_distributed_metadata)
