# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Version file for torch_checkpointing
"""

import importlib.metadata
from typing import Optional, Tuple

UNKNOWN = "Unknown"

try:
    __version__: str = importlib.metadata.version("torch_checkpointing")
except Exception:
    __version__ = UNKNOWN


class Version:
    """Wrapper class for version information that provides utility methods."""

    def __init__(self, full_version_string: str):
        """Initialize Version with a version string like '0.1.0+eb82421ca0c1'."""
        self._full_version_string = full_version_string
        self._version: str
        self._sha: Optional[str]
        (self._version, self._sha) = self._parse_version(self._full_version_string)

    def _parse_version(self, version_string: str) -> Tuple[str, Optional[str]]:
        """Parse the version string into components."""
        # Split by '+' to separate version and SHA
        parts = version_string.split("+", 1)
        version = parts[0]
        sha = parts[1] if len(parts) > 1 else None
        return (version, sha)

    def get_version(self) -> str:
        """Return the official version part (e.g., '0.1.0' from '0.1.0+eb82421ca0c1')."""
        return self._version

    def get_full_version(self) -> str:
        """Return the full version string (e.g., '0.1.0+eb82421ca0c1')."""
        return self._full_version_string

    def get_sha(self) -> Optional[str]:
        """Return the SHA part (e.g., 'eb82421ca0c1' from '0.1.0+eb82421ca0c1')."""
        return self._sha

    def __str__(self) -> str:
        """String representation returns the full version."""
        return self._full_version_string

    def __repr__(self) -> str:
        """Representation of the Version object."""
        return f"Version('{self._full_version_string}')"


def get_version() -> Optional[Version]:
    """Return a Version instance for the current package version, or None if version is unknown."""
    if __version__ == UNKNOWN:
        return None
    return Version(__version__)
