# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import tempfile
from concurrent.futures import Future
from typing import Any, TypeVar

import torch

from .checkpoint_base import (
    CheckpointBase,
    CheckpointInfo,
    CheckpointItem,
    CheckpointWriteInfo,
)
from .checkpoint_process import CheckpointProcess
from .checkpoint_writer import CheckpointWriter
from .lock import RWLock
from .logging_utils import EventLogger, EventType
from .metadata_manager import MetadataManager
from .staging import CheckpointStager
from .types import STATE_DICT
from .utils import compare_state_dicts, ensure_future, fut_then, wrap_future

logger = logging.getLogger(__name__)

LOG_INTERVAL = 60
T = TypeVar("T")
CheckpointT = TypeVar("CheckpointT", bound=CheckpointBase)
CATCH_ALL_FILE_NAME = "other_{rank}.pt"


def _throw_if_not_equal_to_sync_save(
    staged_sd: STATE_DICT, sync_save_file: tempfile._TemporaryFileWrapper, path: str
) -> STATE_DICT:
    logger.info("Checking if staged state_dict is equal to sync copy")
    sync_sd = torch.load(
        sync_save_file.name, map_location="cpu", mmap=True, weights_only=False
    )
    diffs = compare_state_dicts(sync_sd, staged_sd)
    if diffs:
        formatted_diff = "\n".join(
            # Use "/" as path separator because tensor names contain "."
            "/".join(str(elem) for elem in k) + ": " + v.value
            for k, v in diffs.items()
        )
        raise ValueError(
            f"State dict changed during async staging! This means that some portion of state has gotten mutated by the training loop during your forward or backward pass.\nCheckpoint path: {path}\nList of everything that changed:\n{formatted_diff}"
        )
    logger.info(
        f"Staged state_dict is equal to sync copy, everything is good. Checkpoint path: {path}"
    )
    return staged_sd  # To make it chainable


def _get_serialized_metadata(
    metadata_manager: MetadataManager,
    checkpoint_info: CheckpointInfo,
    validate: bool = False,
) -> bytes:
    """
    Return the serialized distributed metadata for a save, computing it (with
    collectives) only when the manager has none cached.

    With ``validate``, raise if this rank's items no longer match the cached
    metadata instead of silently writing stale metadata.
    """
    if not metadata_manager.has_computed_metadata():
        metadata_manager.compute_distributed_metadata(checkpoint_info)
    elif validate and not metadata_manager.computed_metadata_matches(checkpoint_info):
        raise RuntimeError(
            "Checkpoint structure, sharding or layout has changed since its "
            "distributed metadata was computed. Call "
            "invalidate_computed_metadata() on every rank after changing it."
        )
    return metadata_manager.get_serialized_computed_metadata()


class CheckpointSaver(abc.ABC):
    """
    Abstract base class that defines the API for saving checkpoints.

    This class defines the interface for coordinating the writing of model
    state dictionaries to storage. It provides abstract methods to save model states
    with support for both synchronous and asynchronous operations.

    For loading checkpoints, use CheckpointLoader instead.

    Concrete implementations of this class must implement all the abstract methods.
    """

    @abc.abstractmethod
    def save(
        self,
        path: str,
        checkpoint: CheckpointBase,
    ) -> tuple[Future, Future] | None:
        """
        Save a state dictionary to storage.

        Args:
            path: The path where the checkpoint should be saved.
            state_dict: The state dictionary to save.

        Returns:
            For synchronous implementations: None
            For asynchronous implementations: tuple of (stage_future, write_future)
                                              representing the staging and writing operations.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the checkpoint saver and release any resources.

        This method should be called when the checkpoint saver is no longer needed to ensure
        proper cleanup of resources.
        """

    @property
    def stager(self) -> CheckpointStager | None:
        """The staging cache, or None for savers that don't use one
        (e.g. savers that write synchronously instead of pinning to host
        memory). Used to pre-allocate the staging pool before the first
        real save, and by `pinned_num_bytes` reporting.
        """
        return None


class SyncCheckpointSaver(CheckpointSaver):
    """
    Synchronous implementation of CheckpointSaver.

    This class coordinates the writing of model state dictionaries to storage
    using only synchronous operations. It provides a simple, efficient interface for checkpoint
    operations without async overhead.

    For loading checkpoints, use CheckpointLoader instead.

    Attributes:
        _writer: CheckpointWriter for writing state dictionaries to storage.
        _metadata_manager: MetadataManager for aggregating and storing global metadata

    Example:
        saver = SyncCheckpointSaver(
            writer=writer,
            metadata_manager=DefaultMetadataManager(rank_info),
        )
        checkpoint = SimpleCheckpoint(state_dict)
        saver.save(checkpoint, path)

    """

    def __init__(
        self,
        writer: CheckpointWriter,
        metadata_manager: MetadataManager | None = None,
    ):
        """
        Initialize a synchronous checkpoint saver.

        Args:
            writer: CheckpointWriter for writing checkpoints to storage.
            metadata_manager: Optional MetadataManager for distributed metadata.
                Every save raises if its items no longer match the cached
                metadata; invalidate it on every rank after changing them.
        """
        self._writer = writer
        self._metadata_manager = metadata_manager

    def _prepare_checkpoint_write_info(
        self,
        checkpoint: CheckpointBase,
    ) -> CheckpointWriteInfo:
        """
        Prepare CheckpointWriteInfo for CheckpointWriter.write().
        """
        checkpoint_info = CheckpointInfo(checkpoint_items=checkpoint.get_items())
        serialized_metadata: bytes | None = None
        if self._metadata_manager is not None:
            # The local check costs little next to a blocking write.
            serialized_metadata = _get_serialized_metadata(
                self._metadata_manager, checkpoint_info, validate=True
            )
        return checkpoint_info.for_writes(serialized_metadata)

    def save(
        self,
        path: str,
        checkpoint: CheckpointBase,
    ) -> tuple[Future, Future] | None:
        """
        Save a state dictionary to storage synchronously.

        Args:
            path: The path where the checkpoint should be saved.
            checkpoint: The checkpoint to save.

        Returns:
            Always returns None as operations are synchronous.
        """
        logger.debug("Saving checkpoint synchronously to %s", path)

        # Build CheckpointWriteInfo for the writer
        write_info = self._prepare_checkpoint_write_info(checkpoint)

        self._writer.write(
            path,
            write_info,
        )
        return None

    def close(self) -> None:
        """
        Close the checkpoint saver and release any resources.

        This method should be called when the checkpoint saver is no longer needed to ensure
        proper cleanup of resources.
        """
        self._writer.close()
        logger.info("SyncCheckpointSaver closed")


class AsyncCheckpointSaver(CheckpointSaver):
    """
    Asynchronous implementation of CheckpointSaver.

    This class coordinates the writing of model state dictionaries to storage
    using asynchronous operations for saving. It provides efficient async checkpoint operations
    with staging and background writing capabilities.

    The staging buffer is guarded by an ``RWLock`` (exposed via the ``lock``
    property) that tracks access to the shared staged state:

    - ``save()`` acquires the lock for write before staging begins, so no other
      writer can touch the buffer while it is being filled.
    - Once staging finishes, the lock is downgraded to read. The buffer now holds
      a consistent snapshot, so the background writer -- and any other reader --
      can read it concurrently, while the next save's write acquire is held off.
    - Once writing finishes, the read lock is released.

    For loading checkpoints, use CheckpointLoader instead.

    Attributes:
        _checkpoint_stager: Stager for async operations.
        _checkpoint_process: Process for async operations.
        _write_future: Future representing the ongoing async write operation.
        _metadata_manager: MetadataManager for aggregating and storing global metadata
        _lock: RWLock guarding the staging buffer.

    Example:
        saver = AsyncCheckpointSaver(
            checkpoint_stager=stager,
            checkpoint_process=process,
            metadata_manager = DefaultMetadataManager(rank_info),
        )
        stage_future, write_future = saver.save(state_dict, path)
        # ... do other work ...
        write_future.result()  # Wait for completion
    """

    def __init__(
        self,
        checkpoint_stager: CheckpointStager,
        checkpoint_process: CheckpointProcess,
        metadata_manager: MetadataManager | None = None,
        lock_acquire_timeout: float = 600.0,
    ):
        """
        Initialize an asynchronous checkpoint saver.

        Args:
            checkpoint_stager: Stager for async operations.
            checkpoint_process: Process for async operations.
            metadata_manager: Optional MetadataManager for distributed metadata.
            lock_acquire_timeout: Seconds to wait when acquiring the staging-buffer
                lock before raising (a timeout here almost always means a deadlock).
        """
        self._checkpoint_stager = checkpoint_stager
        self._checkpoint_process = checkpoint_process
        self._write_future: Future[Any] | None = None
        self._metadata_manager = metadata_manager
        # The write subprocess reuses the last metadata it was sent, so only send
        # metadata that differs from this.
        self._sent_serialized_metadata: bytes | None = None
        self._staging_lock = RWLock(acquire_timeout=lock_acquire_timeout)

    @property
    def stager(self) -> CheckpointStager:
        return self._checkpoint_stager

    @property
    def staging_lock(self) -> RWLock:
        """The lock guarding the staging buffer (see class docstring)."""
        return self._staging_lock

    def _prepare_checkpoint_write_info(
        self,
        checkpoint: CheckpointBase,
        validate: bool = False,
    ) -> CheckpointWriteInfo:
        """
        Prepare CheckpointWriteInfo for CheckpointProcess.write().

        Fetches the metadata on the calling thread, so a later invalidation
        cannot change what this save writes. The write info carries it only if
        the write subprocess doesn't already hold it.

        Args:
            checkpoint: The checkpoint object
            validate: If True, raise if the items no longer match the cached
                metadata.
        """
        checkpoint_info = CheckpointInfo(checkpoint_items=checkpoint.get_items())
        if self._metadata_manager is None:
            return checkpoint_info.for_writes(None)

        serialized_metadata = _get_serialized_metadata(
            self._metadata_manager, checkpoint_info, validate=validate
        )
        if serialized_metadata is self._sent_serialized_metadata:
            return checkpoint_info.for_writes(None)
        self._sent_serialized_metadata = serialized_metadata
        return checkpoint_info.for_writes(serialized_metadata)

    def _stage(
        self,
        checkpoint: CheckpointBase,
    ) -> Future[STATE_DICT]:
        """
        Stage a state dictionary for async writing. Non-public API and not thread-safe.
        """
        checkpoint_items = checkpoint.get_items()
        state_dict = {k: item.value for k, item in checkpoint_items.items()}
        keys_not_requiring_copy = {
            k for k, item in checkpoint_items.items() if not item.requires_copy
        }
        return ensure_future(
            self._checkpoint_stager.stage(state_dict, keys_not_requiring_copy)
        )

    def save(
        self,
        path: str,
        checkpoint: CheckpointBase,
        validate_state_dict: bool = False,
    ) -> tuple[Future, Future]:
        """
        Save a state dictionary to storage asynchronously.

        Args:
            path: The path where the checkpoint should be saved.
            checkpoint: The checkpoint object to save.
            validate_state_dict: If True, perform overall validation of the state_dict
              pre/post staging to verify nothing has been mutated by the training loop.
              Also validates consistency with computed metadata. This is extremely
              expensive and should only be used for testing/debugging.

        Returns:
            A tuple of (stage_future, write_future) representing the staging and writing operations.

        Example:
            stage_future, write_future = checkpointer.save("/path/to/checkpoint", state_dict)
            # ... do other work ...
            write_future.result()  # Wait for completion
        """
        event_logger = EventLogger()
        logger.info(
            f"Initiating checkpoint save to {path}. Will wait for prev checkpoints to complete.",
        )

        # Acquire the staging buffer exclusively before any staging work begins,
        # so no other writer can mutate it while we fill it. It is held (later
        # downgraded to read) until the background write finishes; see the class
        # docstring for the full lifecycle.
        self._staging_lock.write.acquire_or_raise()
        logger.info(
            "Finished waiting for previous checkpointing to finish",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_write.submit.wait_for_prev.latency_ms",
            ),
        )
        try:
            # The write acquire above blocks until any previous checkpoint's write
            # has finished (the lock is released only then), so this result()
            # returns immediately -- it just surfaces a failed prior write.
            if self._write_future is not None:
                self._write_future.result()
                self._write_future = None

            logger.info(
                "Preparing checkpoint write info",
            )
            checkpoint_write_info = self._prepare_checkpoint_write_info(
                checkpoint, validate=validate_state_dict
            )
            logger.info(
                "Finished preparing checkpoint write info",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name="train.checkpoint_write.submit.prepare_write_info.latency_ms",
                ),
            )

            state_dict = checkpoint_write_info.state_dict
            layout_info_mappings = checkpoint_write_info.layout_info_mappings
            serialized_metadata = checkpoint_write_info.serialized_distributed_metadata

            if validate_state_dict:
                # Create a sync copy of the state_dict to compare against later. Write it
                # to disk to avoid carrying around extra GPU copies. The temp file will get
                # deleted when sync_save_file gets garbage collected.
                sync_save_file = tempfile.NamedTemporaryFile()
                torch.save(state_dict, sync_save_file.name, pickle_protocol=-1)
            else:
                sync_save_file = None

            logger.debug("Starting state dictionary staging")

            staging_fut = self._stage(checkpoint)
            if validate_state_dict:
                assert sync_save_file is not None
                staging_fut = fut_then(
                    staging_fut,
                    lambda sd: _throw_if_not_equal_to_sync_save(
                        sd, sync_save_file, path
                    ),
                )

            logger.debug("Starting checkpoint write to %s", path)

            def build_checkpoint_write_info_from_staged(
                staged_sd: STATE_DICT,
            ) -> CheckpointWriteInfo:
                # Build CheckpointWriteInfo from the merged state_dict with layout info
                rebuilt_items = {
                    key: CheckpointItem(
                        value=staged_sd[key],
                        layout=layout_info_mappings.get(key),
                    )
                    for key in staged_sd.keys()
                }

                write_info = CheckpointWriteInfo(checkpoint_items=rebuilt_items)
                return write_info.for_writes(serialized_metadata)

            # Transform the staging future to produce CheckpointWriteInfo
            checkpoint_write_info_fut = fut_then(
                staging_fut, build_checkpoint_write_info_from_staged
            )

            write_future = self._checkpoint_process.write(
                checkpoint_write_info_fut,
                path,
            )
        except BaseException:
            # Futures failed to launch; release the lock
            self._staging_lock.write.release()
            # The write subprocess may not have received the metadata.
            self._sent_serialized_metadata = None
            raise

        # Futures have launched; release the lock in callbacks. On staging
        # completion: downgrade write->read (the buffer now holds a consistent
        # snapshot readable concurrently), then arm the read release on write
        # completion. Registering the release only AFTER the downgrade has run
        # guarantees it is never scheduled before the downgrade.
        def _downgrade_then_release_after_write(_staging_fut: Future) -> None:
            self._staging_lock.downgrade()
            write_future.add_done_callback(
                lambda _write_fut: self._staging_lock.read.release()
            )

        staging_fut.add_done_callback(_downgrade_then_release_after_write)
        self._write_future = write_future

        logger.info(
            f"Checkpoint save to {path} initiated",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_write.submit.checkpointer.latency_ms",
                end_to_end=True,
            ),
        )

        # Return futures for the staging and writing operations
        return wrap_future(staging_fut), self._write_future

    def stage(self, checkpoint: CheckpointBase) -> Future[STATE_DICT]:
        """
        Stage a checkpoint into the buffer without writing it to storage. Acquires
        the lock for write, and releases it once staging completes.
        """
        self._staging_lock.write.acquire_or_raise()
        try:
            staging_fut = self._stage(checkpoint)
        except BaseException:
            self._staging_lock.write.release()
            raise
        staging_fut.add_done_callback(
            lambda _staging: self._staging_lock.write.release()
        )
        return staging_fut

    def close(self) -> None:
        """
        Close the checkpoint saver and release any resources.

        This method should be called when the checkpoint saver is no longer needed to ensure
        proper cleanup of async resources.
        """
        self._checkpoint_stager.close()
        self._checkpoint_process.close()
        logger.info("AsyncCheckpointSaver closed")
