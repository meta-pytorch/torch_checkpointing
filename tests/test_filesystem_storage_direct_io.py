# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import tempfile
from pathlib import Path
from unittest import mock

import torch_checkpointing.storage.filesystem as filesystem_module
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_checkpointing.storage.base_storage import ReadArgs
from torch_checkpointing.storage.filesystem import (
    LocalFileSystemStorageConfig,
    WriteStream,
)


class TestDirectIOWithoutOSSupport(TestCase):
    """Platforms without os.O_DIRECT (macOS, Windows) must use buffered I/O."""

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "payload.bin"
        patcher = mock.patch.object(filesystem_module, "_O_DIRECT", None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_write_stream_falls_back_to_buffered_io(self) -> None:
        with WriteStream(self.path, use_direct_io=True) as stream:
            stream.write(b"x" * 16)
            self.assertFalse(stream.using_direct_io)
        self.assertEqual(self.path.read_bytes(), b"x" * 16)

    def test_storage_round_trip_never_tries_direct_io(self) -> None:
        storage = LocalFileSystemStorageConfig().create_storage()
        self.assertFalse(storage._try_direct_io_reads)
        storage.write(self.path, b"z" * 32)
        self.assertEqual(storage.read(self.path), b"z" * 32)
        # Asking for direct I/O still opens a plain buffered file.
        read_args = ReadArgs(pre_read_full_file=False, direct_io=True)
        with storage.stream_read(self.path, read_args) as f:
            self.assertIsInstance(f, io.FileIO)
            self.assertEqual(f.read(), b"z" * 32)


if __name__ == "__main__":
    run_tests()
