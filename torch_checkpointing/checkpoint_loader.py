# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight checkpoint loader for load-only scenarios.

This module provides CheckpointLoader, a standalone loader class that handles
checkpoint loading without initializing save-specific infrastructure
(subprocess, staging, barriers). This enables eval-only workloads to avoid
the overhead of save infrastructure.
"""

import logging
from typing import Any, TypeVar

from .checkpoint_base import CheckpointBase, CheckpointInfo, CheckpointReadInfo
from .checkpoint_reader import CheckpointReader
from .config import CheckpointLoaderConfig
from .distributed_metadata import CheckpointMetadata
from .logging_utils import EventLogger, EventType
from .metadata_manager import MetadataManager

logger = logging.getLogger(__name__)

CheckpointT = TypeVar("CheckpointT", bound=CheckpointBase)


class CheckpointLoader:
    """
    Lightweight checkpoint loader for load-only scenarios.

    This class provides checkpoint loading without initializing save-specific
    infrastructure (subprocess, staging, barriers). It's suitable for:
    - Evaluation workloads that only need to load checkpoints
    - Inference pipelines that don't need save capabilities
    - Any scenario where save overhead should be avoided
    - Resharding: loading checkpoints saved with a different parallelism config

    Example:
        # Eval-only (no save infrastructure)
        reader = CheckpointReader(rank_info=rank_info, storage_config=storage_config)
        loader = CheckpointLoader(reader=reader)
        loader.load(path, checkpoint)
        loader.close()
    """

    def __init__(
        self,
        reader: CheckpointReader,
        metadata_manager: MetadataManager | None = None,
        config: CheckpointLoaderConfig | None = None,
    ) -> None:
        """
        Initialize a CheckpointLoader.

        Args:
            reader: CheckpointReader for reading from storage.
            metadata_manager: Optional MetadataManager for resharding support.
                If None, resharding is disabled.
            config: Optional CheckpointLoaderConfig. If None, a default
                config is used (verify_integrity=False).
        """
        self._reader = reader
        self._metadata_manager = metadata_manager
        self._config = config or CheckpointLoaderConfig()

    def _compute_metadata_once(
        self,
        checkpoint: CheckpointBase,
    ) -> CheckpointMetadata | None:
        """
        Compute CheckpointMetadata. Only computes once per metadata_manager.

        Uses compute_metadata() result as the guard - it returns None when cache is valid.
        The metadata_manager automatically kicks off async serialization after first compute.

        Returns:
            CheckpointMetadata if newly computed, None if already computed or no metadata_manager.
        """
        if self._metadata_manager is None:
            return None

        items = checkpoint.get_items()
        checkpoint_info = CheckpointInfo(checkpoint_items=items)

        # compute_metadata returns None if cache is valid (already computed)
        # It also kicks off async serialization automatically after first compute
        return self._metadata_manager.compute_metadata(checkpoint_info)

    def _prepare_checkpoint_read_info(
        self,
        checkpoint: CheckpointBase,
        checkpoint_metadata: CheckpointMetadata | None,
    ) -> CheckpointReadInfo:
        """
        Prepare CheckpointReadInfo for CheckpointReader.read().
        """
        items = checkpoint.get_items()
        checkpoint_info = CheckpointInfo(checkpoint_items=items)
        return checkpoint_info.for_reads(checkpoint_metadata)

    def load(
        self,
        path: str,
        checkpoint: CheckpointT,
        default_map_location: Any = None,
        strict: bool = False,
    ) -> None:
        """
        Load a checkpoint from storage.

        In-place modification behavior:
            Loaded data is merged into the checkpoint object's values. The following
            are modified IN-PLACE:
            - Mutable containers (dict, list, deque): updated in the existing objects
            - Tensors: data is copied via copy_() into the target tensors, preserving
              the target tensor's identity (same object, updated data)

            The following are NOT modified in-place:
            - Immutable containers (tuple): new containers are created
            - Non-tensor leaf values: source value replaces target value

            When checkpoint item values are None, new objects are created from the
            loaded checkpoint data.

        Args:
            path: The path from which to load the checkpoint.
            checkpoint: CheckpointBase object to update with loaded values.
                Only keys checkpoint.get_items() returns will be loaded.
            default_map_location: Device mapping function or device name for
                relocating tensors.
            strict: If True, raises an error when there are missing keys in the
                checkpoint.

        Raises:
            RuntimeError: If strict=True and there are missing keys in the checkpoint.
            FileNotFoundError: If the checkpoint file is not found.
        """
        event_logger = EventLogger()
        logger.info("Loading checkpoint from %s", path)

        # Verify checkpoint integrity before any weight is materialised.
        # If the manifest is absent (old checkpoint saved without
        # verify_integrity), the load proceeds silently.
        if self._config.verify_integrity:
            from .integrity import verify_manifest

            logger.debug("Verifying checkpoint integrity for %s", path)
            try:
                verify_manifest(path)
                logger.info("Checkpoint integrity verified for %s", path)
            except FileNotFoundError:
                logger.debug(
                    "No integrity manifest at %s, skipping verification", path
                )

        logger.info(
            "Preparing checkpoint read info",
        )
        # Prepare CheckpointReadInfo with metadata for resharding
        # This also kicks off async serialization in the metadata_manager
        checkpoint_metadata = self._compute_metadata_once(checkpoint)
        checkpoint_read_info = self._prepare_checkpoint_read_info(
            checkpoint, checkpoint_metadata
        )
        logger.info(
            "Finished preparing checkpoint read info",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.submit.prepare_read_info.latency_ms",
            ),
        )

        # Read the checkpoint from storage
        loaded_state_dict, missing_keys = self._reader.read(
            path=path,
            checkpoint_info=checkpoint_read_info,
            map_location=default_map_location,
        )
        if strict and missing_keys is not None and missing_keys != []:
            raise RuntimeError(f"Checkpoint at {path} is missing keys: {missing_keys}")

        logger.info(
            f"Checkpoint loaded from {path}",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.submit.reader.latency_ms",
            ),
        )

        # Load the state into the checkpoint object
        checkpoint.load_state_dict(loaded_state_dict)

        logger.info(
            "Load State Dict complete",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.submit.load_state_dict.latency_ms",
            ),
        )

        logger.info(
            f"Checkpoint loaded from {path}",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.submit.checkpointer.latency_ms",
                end_to_end=True,
            ),
        )

        return None

    def close(self) -> None:
        """Release any resources."""
        logger.info("CheckpointLoader closed")
