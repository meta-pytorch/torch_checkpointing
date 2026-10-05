# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Discover and load distributed checkpoint metadata."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from .distributed_metadata import DistributedMetadata, METADATA_FILE_NAME
from .storage.base_storage import Storage


class DistributedMetadataFormat(ABC):
    """An on-disk representation that can produce ``DistributedMetadata``."""

    @classmethod
    @abstractmethod
    def maybe_load(
        cls,
        checkpoint_dir: Path,
        storage: Storage,
    ) -> DistributedMetadata | None:
        """Load this format, or return ``None`` when its marker is absent.

        A present but malformed metadata artifact must raise rather than look
        absent and allow discovery to fall through to another format.
        """
        raise NotImplementedError


def load_distributed_metadata(
    checkpoint_dir: str | Path,
    storage: Storage,
    *,
    formats: Sequence[type[DistributedMetadataFormat]],
) -> DistributedMetadata | None:
    """Load the first recognized metadata format in ``checkpoint_dir``.

    Args:
        checkpoint_dir: Checkpoint directory whose metadata should be loaded.
        storage: Storage backend containing the checkpoint.
        formats: Formats to try in precedence order. An empty sequence
            intentionally disables metadata discovery.

    Returns:
        The discovered distributed metadata, or ``None`` when no format is
        present.
    """
    checkpoint_dir = Path(checkpoint_dir)
    for format_type in formats:
        metadata = format_type.maybe_load(checkpoint_dir, storage)
        if metadata is not None:
            return metadata

    return None


class TorchDistributedMetadataFormat(DistributedMetadataFormat):
    """The native trusted-pickle ``metadata.pkl`` representation."""

    @classmethod
    def maybe_load(
        cls,
        checkpoint_dir: Path,
        storage: Storage,
    ) -> DistributedMetadata | None:
        """Load trusted native metadata when ``metadata.pkl`` is present.

        ``metadata.pkl`` uses pickle and may execute arbitrary code. Only load
        checkpoints from trusted sources that have not been tampered with.
        """
        metadata_path = checkpoint_dir / METADATA_FILE_NAME
        if not storage.exists(metadata_path):
            return None
        return DistributedMetadata.from_dict(pickle.loads(storage.read(metadata_path)))
