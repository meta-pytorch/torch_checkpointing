# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint integrity verification via manifests.

On save, every file (or tensor, in future granularities) in the checkpoint
directory is hashed and the hashes are written to
``_integrity_manifest.json``.  On load, the manifest is read and each
entry is re-hashed; any mismatch indicates silent corruption (bit rot,
truncated writes, transfer errors) and the load is rejected.

Design choices:
  - **Algorithm-pluggable**: ``write_manifest`` and ``verify_manifest``
    accept an ``algorithm`` parameter (default ``"sha256"``).  Currently
    only SHA-256 is implemented; adding a new algorithm is a single
    function + one branch in ``_compute_hash_dispatch``.
  - **Granularity-pluggable**: a ``granularity`` parameter (default
    ``"file"``) controls the hash scope.  Only ``"file"`` is implemented
    now; ``"tensor"`` is reserved for future per-FQN hashing.
  - **Manifest is forward-compatible**: it records ``version``,
    ``granularity`` and ``algorithm`` so that a future version can still
    verify old manifests (the verifier reads the algorithm from the
    manifest, not the current config).
  - **Distributed-safe**: only rank 0 computes / verifies hashes and
    the verdict is broadcast so all ranks raise in lock-step or proceed
    together.
  - **The manifest itself is excluded from hashing** (chicken-and-egg).

Policies (hardcoded, no per-run knobs):
  - mismatch  -> always raise.
  - missing manifest -> raises FileNotFoundError (caller silently skips).
  - extra-file detection -> disabled (hardcoded False).
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "_integrity_manifest.json"
"""Fixed manifest filename.  Prefixed with ``_`` so DCP / safetensors
loaders that glob for known extensions simply ignore it."""

_READ_CHUNK_SIZE = 1 << 20  # 1 MiB — same as Megatron-LM.
"""Stream-read chunk size for large-file hashing.  Keeps memory usage
constant regardless of file size."""

# ---------------------------------------------------------------------------
# Algorithm dispatch
# ---------------------------------------------------------------------------

Algorithm = Literal["sha256"]
"""Supported hash algorithms.  Currently only ``"sha256"``; extending
the Literal and adding a branch to :func:`_compute_hash_dispatch` is
all that's needed to support e.g. ``"blake3"`` or ``"crc32"``."""


def _compute_hash_sha256(filepath: str | os.PathLike) -> str:
    """Stream-hash a single file with SHA-256, returning a hex digest."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(_READ_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_hash_dispatch(
    filepath: str | os.PathLike,
    algorithm: str,
) -> str:
    """Dispatch to the appropriate hash function by algorithm name.

    To add a new algorithm: implement a ``_compute_hash_<name>`` function
    and add a branch here.
    """
    if algorithm == "sha256":
        return _compute_hash_sha256(filepath)
    raise NotImplementedError(
        f"Hash algorithm '{algorithm}' is not implemented. "
        f"Supported: 'sha256'."
    )


# ---------------------------------------------------------------------------
# Granularity: file-level (v1)
# ---------------------------------------------------------------------------

Granularity = Literal["file"]
"""Supported hash granularities.  Currently only ``"file"``; ``"tensor"``
is reserved for future per-FQN hashing (see :func:`_collect_hashes_file`
for the file-level implementation)."""


def _collect_hashes_file(
    directory: str | os.PathLike,
    algorithm: str,
) -> dict[str, str]:
    """Hash every file in *directory* (non-recursive, sorted by name).

    The manifest file itself is excluded from hashing.

    To add a ``"tensor"`` granularity: implement a
    ``_collect_hashes_tensor`` function that walks the state dict and
    hashes per-FQN, then add a branch in :func:`_collect_hashes_dispatch`.
    """
    dir_path = Path(directory)
    entries: list[tuple[str, str]] = []
    for entry in sorted(dir_path.iterdir(), key=lambda p: p.name):
        if entry.is_file() and entry.name != MANIFEST_FILENAME:
            entries.append((entry.name, _compute_hash_dispatch(entry, algorithm)))
    return dict(entries)


def _collect_hashes_dispatch(
    directory: str | os.PathLike,
    algorithm: str,
    granularity: str,
) -> dict[str, str]:
    """Dispatch to the appropriate hash-collection function by granularity."""
    if granularity == "file":
        return _collect_hashes_file(directory, algorithm)
    raise NotImplementedError(
        f"Granularity '{granularity}' is not implemented. "
        f"Supported: 'file'."
    )


# ---------------------------------------------------------------------------
# Manifest write / read
# ---------------------------------------------------------------------------

def write_manifest(
    directory: str | os.PathLike,
    *,
    algorithm: Algorithm = "sha256",
    granularity: Granularity = "file",
) -> None:
    """Compute hashes for every entry in *directory* and write the manifest.

    Args:
        directory: checkpoint directory to hash.
        algorithm: hash algorithm to use (default ``"sha256"``).
        granularity: hash granularity to use (default ``"file"``).

    Should be called **after** all checkpoint files (including
    ``.metadata``) have been written to disk.
    """
    files = _collect_hashes_dispatch(directory, algorithm, granularity)
    payload: dict[str, Any] = {
        "version": 1,
        "granularity": granularity,
        "algorithm": algorithm,
        "files": files,
    }
    manifest_path = os.path.join(str(directory), MANIFEST_FILENAME)
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.debug(
        "Wrote integrity manifest to %s (%d files, algorithm=%s, granularity=%s)",
        manifest_path, len(files), algorithm, granularity,
    )


def _load_manifest(directory: str | os.PathLike) -> dict[str, Any]:
    """Load and parse the manifest for *directory*.

    Raises ``FileNotFoundError`` if the manifest is absent.
    Raises ``ValueError`` if the manifest is incompatible.
    """
    manifest_path = os.path.join(str(directory), MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Integrity manifest not found at {manifest_path}. "
            f"Checkpoint was likely saved without verify_integrity=True."
        )
    with open(manifest_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest at {manifest_path} is not a JSON object")
    return payload


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class CheckpointingException(Exception):
    """Raised when checkpoint hashes do not match the manifest."""
    pass


def _verify_manifest_impl(
    directory: str | os.PathLike,
) -> None:
    """Re-hash every entry listed in the manifest and compare.

    The algorithm and granularity are read **from the manifest** (not
    from the current config), so that a manifest produced by one
    algorithm can still be verified even if the default has since
    changed.

    Collects **all** mismatches before raising, so the error message
    lists every bad entry rather than just the first one.
    """
    manifest_data = _load_manifest(directory)
    algorithm = manifest_data.get("algorithm", "sha256")
    granularity = manifest_data.get("granularity", "file")
    expected: dict[str, str] = manifest_data.get("files", {})

    mismatches: list[str] = []
    for name, expected_hash in expected.items():
        full_path = os.path.join(str(directory), name)
        try:
            actual_hash = _compute_hash_dispatch(full_path, algorithm)
        except (FileNotFoundError, OSError) as exc:
            mismatches.append(f"  {name}: file missing or unreadable ({exc})")
            continue
        if actual_hash != expected_hash:
            mismatches.append(
                f"  {name}: hash mismatch "
                f"(expected {expected_hash[:16]}..., got {actual_hash[:16]}...)"
            )

    if mismatches:
        raise CheckpointingException(
            f"Checkpoint integrity verification failed for {directory} "
            f"(algorithm={algorithm}, granularity={granularity}):\n"
            + "\n".join(mismatches)
        )


def verify_manifest(directory: str | os.PathLike) -> None:
    """Verify checkpoint integrity by re-hashing every entry listed in
    the manifest and comparing.

    The algorithm and granularity are read from the manifest itself, so
    verification always matches what was written at save time.

    Only rank 0 does the actual file I/O; the verdict is broadcast to
    all ranks so every rank raises in lock-step or proceeds together.
    Falls back to a single-process path when ``torch.distributed`` is
    not initialised.

    Raises ``CheckpointingException`` on any mismatch.
    Raises ``FileNotFoundError`` if the manifest is absent (the caller
    is expected to catch this and silently skip for old checkpoints).
    """
    import torch.distributed as dist

    dist_available = (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )

    if dist_available:
        # error_payload is None on success, or a (exc_type_name, exc_msg)
        # tuple on failure.  We preserve the exception type so that
        # FileNotFoundError (manifest absent) can be caught separately
        # from CheckpointingException (hash mismatch) by callers.
        error_payload: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                _verify_manifest_impl(directory)
            except FileNotFoundError as exc:
                error_payload = [("FileNotFoundError", str(exc))]
            except CheckpointingException as exc:
                error_payload = [("CheckpointingException", str(exc))]
            except Exception as exc:
                error_payload = [("CheckpointingException", str(exc))]
        dist.broadcast_object_list(error_payload, src=0)
        if error_payload[0] is not None:
            exc_type, exc_msg = error_payload[0]
            if exc_type == "FileNotFoundError":
                raise FileNotFoundError(exc_msg)
            raise CheckpointingException(exc_msg)
    else:
        _verify_manifest_impl(directory)
