# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from unittest.mock import patch


@patch("importlib.metadata.version")
def test_version_class_with_sha(mock_version):
    """Test Version class with version string containing SHA."""
    # Mock the importlib.metadata.version to return a controlled version string
    mock_version.return_value = "1.2.3+abc123def456"

    # Re-import the version module to get the mocked version
    import torch_checkpointing.version

    importlib.reload(torch_checkpointing.version)

    version = torch_checkpointing.version.get_version()

    # Assert version is not None (it should be a Version object for valid versions)
    assert version is not None

    # Test get_version
    assert version.get_version() == "1.2.3"

    # Test get_full_version
    assert version.get_full_version() == "1.2.3+abc123def456"

    # Test get_sha
    assert version.get_sha() == "abc123def456"

    # Test string representation
    assert str(version) == "1.2.3+abc123def456"

    # Test repr
    assert repr(version) == "Version('1.2.3+abc123def456')"

    # Verify mock was called with correct package name
    mock_version.assert_called_with("torch_checkpointing")


@patch("importlib.metadata.version")
def test_version_class_without_sha(mock_version):
    """Test Version class with version string without SHA."""
    # Mock the importlib.metadata.version to return a controlled version string
    mock_version.return_value = "2.0.1"

    # Re-import the version module to get the mocked version
    import torch_checkpointing.version

    importlib.reload(torch_checkpointing.version)

    version = torch_checkpointing.version.get_version()

    # Assert version is not None (it should be a Version object for valid versions)
    assert version is not None

    # Test get_version
    assert version.get_version() == "2.0.1"

    # Test get_full_version
    assert version.get_full_version() == "2.0.1"

    # Test get_sha (should be None)
    assert version.get_sha() is None

    # Test string representation
    assert str(version) == "2.0.1"

    # Test repr
    assert repr(version) == "Version('2.0.1')"

    # Verify mock was called with correct package name
    mock_version.assert_called_with("torch_checkpointing")


@patch("importlib.metadata.version")
def test_version_class_multiple_plus_signs(mock_version):
    """Test Version class with multiple plus signs (should split on first one only)."""
    # Mock the importlib.metadata.version to return a controlled version string
    mock_version.return_value = "1.0.0+sha+extra"

    # Re-import the version module to get the mocked version
    import torch_checkpointing.version

    importlib.reload(torch_checkpointing.version)

    version = torch_checkpointing.version.get_version()

    # Assert version is not None (it should be a Version object for valid versions)
    assert version is not None

    assert version.get_version() == "1.0.0"
    assert version.get_full_version() == "1.0.0+sha+extra"
    assert version.get_sha() == "sha+extra"

    # Verify mock was called with correct package name
    mock_version.assert_called_with("torch_checkpointing")


@patch("importlib.metadata.version")
def test_version_class_edge_cases(mock_version):
    """Test Version class with edge cases."""
    # Mock the importlib.metadata.version to return a controlled version string
    mock_version.return_value = "1.0.0+"

    # Re-import the version module to get the mocked version
    import torch_checkpointing.version

    importlib.reload(torch_checkpointing.version)

    # Empty SHA after plus
    version = torch_checkpointing.version.get_version()

    # Assert version is not None (it should be a Version object for valid versions)
    assert version is not None

    assert version.get_version() == "1.0.0"
    assert version.get_full_version() == "1.0.0+"
    assert version.get_sha() == ""

    # Verify mock was called with correct package name
    mock_version.assert_called_with("torch_checkpointing")


@patch("importlib.metadata.version")
def test_version_class_parsing_with_raw_version(mock_version):
    """Test Version class with a mocked package version."""
    # Mock the importlib.metadata.version to return a controlled version string
    mock_version.return_value = "2.0.1+def789abc123"

    # Re-import the version module to get the mocked version
    import torch_checkpointing.version

    importlib.reload(torch_checkpointing.version)

    version = torch_checkpointing.version.Version(
        torch_checkpointing.version.__version__
    )

    # Should properly parse the mocked version
    assert isinstance(version.get_version(), str)
    assert isinstance(version.get_full_version(), str)
    assert version.get_full_version() == "2.0.1+def789abc123"

    # Version contains SHA, test that it's properly extracted
    expected_official = "2.0.1"
    expected_sha = "def789abc123"
    assert version.get_version() == expected_official
    assert version.get_sha() == expected_sha

    # Verify mock was called with correct package name
    mock_version.assert_called_with("torch_checkpointing")


@patch("importlib.metadata.version")
def test_version_package_not_found_exception(mock_version):
    """Test version fallback behavior when package is not found."""
    # Mock importlib.metadata.version to raise PackageNotFoundError
    from importlib.metadata import PackageNotFoundError

    mock_version.side_effect = PackageNotFoundError("torch_checkpointing")

    # Re-import the version module to get the mocked exception behavior
    import torch_checkpointing.version

    importlib.reload(torch_checkpointing.version)

    # Should fall back to "Unknown" when package is not found
    assert torch_checkpointing.version.__version__ == "Unknown"

    # get_version() should return None when version is "Unknown"
    version = torch_checkpointing.version.get_version()
    assert version is None
