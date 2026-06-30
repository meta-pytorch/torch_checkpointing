# Owner(s): ["oncall: distributed checkpointing"]

import array
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig


class TestStorage(TestCase):
    def setUp(self):
        # Set up the base directory for the test. The temporary testing directory will be set up inside this.
        # As these are made and deleted using os.makedirs and shutil.rmtree, this assumes the directories are local.
        # Change if dealing with remote directories that are not mounted locally.
        self.base_dir = "/tmp"
        self.temp_dir = os.path.join(
            self.base_dir, f"test_storage_{random.randint(100000, 999999)}"
        )
        os.makedirs(self.temp_dir, exist_ok=True)

        # Set up the storage backend here and initialize with the desired settings.
        self.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
        self.storage = self.storage_config.create_storage()

        # Prepare test data of different types
        self.bytes_data = b"Hello bytes! "
        self.bytearray_data = bytearray(b"Hello bytearray! ")
        self.numpy_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.torch_tensor = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        self.python_array = array.array("i", [100, 200, 300, 400, 500])

        # Calculate expected content for testing on read.
        self.expected_content = (
            self.bytes_data
            + bytes(self.bytearray_data)
            + self.numpy_array.tobytes()
            + self.torch_tensor.numpy().tobytes()
            + self.python_array.tobytes()
        )

    def tearDown(self):
        # Clean up temp directory after tests
        shutil.rmtree(self.temp_dir)

    def write_data_using_write(self, file_path):
        """Write test data using the write method."""
        self.storage.write(Path(file_path), self.expected_content)

    def write_data_using_stream_write(self, file_path):
        """Write test data using the stream_write method."""
        # stream_write supports numpy arrays (and torch tensors.numpy()),and raw python arrays directly as the input data without needing a copy.
        stream = self.storage.stream_write(Path(file_path))
        with stream:
            stream.write(self.bytes_data)
            stream.write(self.bytearray_data)
            stream.write(self.numpy_array)
            stream.write(self.torch_tensor.numpy())
            stream.write(self.python_array)

    def return_ls_check(self, paths_with_expected):
        """
        Compares the ls result for each of the paths with the expected results.

        Args:
            paths_with_expected: List of tuples (path, expected_contents) where
                                expected_contents is a list of expected file/dir names
        """
        for path, expected in paths_with_expected:
            result = self.storage.ls(Path(path))
            self.assertEqual(
                sorted(result),
                sorted(expected),
                f"ls result for {path} does not match expected. Got {sorted(result)}, expected {sorted(expected)}",
            )

    def test_local_filesystem(self):
        """Test functionality of the local file system."""
        # Create dir_level1/dir_level2 to test mkdir.
        dir_level1_path = os.path.join(self.temp_dir, "dir_level1")
        dir_level2_path = os.path.join(dir_level1_path, "dir_level2")

        self.storage.mkdir(Path(dir_level1_path), recursive=True)
        self.storage.mkdir(Path(dir_level2_path), recursive=True)

        # Verify directories were created
        self.assertTrue(os.path.exists(dir_level1_path))
        self.assertTrue(os.path.exists(dir_level2_path))

        # Test write functionality by writing into file.
        # File is write_output.bin in dir_level1
        write_output_file = os.path.join(dir_level1_path, "write_output.bin")
        self.write_data_using_write(write_output_file)

        # Verify file was created and has correct content
        self.assertTrue(os.path.exists(write_output_file))
        with open(write_output_file, "rb") as f:
            content = f.read()
        self.assertEqual(content, self.expected_content)

        # Test stream_write functionality
        # File is stream_write_output.bin in dir_level2
        stream_write_output_file = os.path.join(
            dir_level2_path, "stream_write_output.bin"
        )
        self.write_data_using_stream_write(stream_write_output_file)

        # Verify file was created and has correct content
        self.assertTrue(os.path.exists(stream_write_output_file))
        with open(stream_write_output_file, "rb") as f:
            content = f.read()
        self.assertEqual(content, self.expected_content)

        # List the files and directories to test ls. Test ls at self.temp_dir, self.temp_dir/dir_level1, and self.temp_dir/dir_level1/dir_level2
        self.return_ls_check(
            [
                (self.temp_dir, ["dir_level1"]),
                (dir_level1_path, ["dir_level2", "write_output.bin"]),
                (dir_level2_path, ["stream_write_output.bin"]),
            ]
        )

        # Test rename by renaming stream_write_output.bin to renamed_stream_write.bin
        renamed_file = os.path.join(dir_level2_path, "renamed_stream_write.bin")
        self.storage.rename(Path(stream_write_output_file), Path(renamed_file))

        # Verify rename worked
        self.assertFalse(os.path.exists(stream_write_output_file))
        self.assertTrue(os.path.exists(renamed_file))

        # Verify ls shows the renamed file
        self.return_ls_check(
            [
                (dir_level2_path, ["renamed_stream_write.bin"]),
            ]
        )

        # Test delete by deleting the two files.
        self.storage.delete(Path(write_output_file))
        self.storage.delete(Path(renamed_file))

        # Verify files were deleted
        self.assertFalse(os.path.exists(write_output_file))
        self.assertFalse(os.path.exists(renamed_file))

        # Verify directories are now empty
        self.return_ls_check(
            [
                (dir_level1_path, ["dir_level2"]),
                (dir_level2_path, []),
            ]
        )

        # Test rmdir by deleting all the directories created for the test.
        self.storage.rmdir(Path(dir_level1_path))

        # Verify entire directory tree was removed
        self.assertFalse(os.path.exists(dir_level1_path))
        self.assertFalse(os.path.exists(dir_level2_path))

    def test_write_fails_if_parent_dir_not_created(self):
        """Test that write fails with FileNotFoundError if parent directory doesn't exist."""
        # Create a path to a file in a non-existent directory
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent_dir")
        test_file = os.path.join(nonexistent_dir, "test_file.bin")

        # Verify the parent directory doesn't exist
        self.assertFalse(os.path.exists(nonexistent_dir))

        # Test that write() fails with FileNotFoundError
        with self.assertRaises(RuntimeError) as context:
            self.storage.write(Path(test_file), b"test data")

        # Test that stream_write() also fails with FileNotFoundError
        with self.assertRaises(RuntimeError) as context:
            stream = self.storage.stream_write(Path(test_file))
            with stream:
                stream.write(b"test data")

        # Verify the file was not created
        self.assertFalse(os.path.exists(test_file))

        # Now create the parent directory and verify writing works
        self.storage.mkdir(Path(nonexistent_dir))
        self.assertTrue(os.path.exists(nonexistent_dir))

        # Write should now succeed
        self.storage.write(Path(test_file), b"test data")
        self.assertTrue(os.path.exists(test_file))

        # Verify the content
        with open(test_file, "rb") as f:
            content = f.read()
        self.assertEqual(content, b"test data")
