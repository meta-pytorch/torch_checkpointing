# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the integrity verification module."""

import hashlib
import json
import os
import shutil
import tempfile

from torch.testing._internal.common_utils import run_tests, TestCase

from torch_checkpointing.integrity import (
    CheckpointingException,
    MANIFEST_FILENAME,
    _compute_hash_dispatch,
    _compute_hash_sha256,
    verify_manifest,
    write_manifest,
)


class TestComputeHash(TestCase):

    def test_sha256_known_value(self) -> None:
        data = b"hello world"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        try:
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(_compute_hash_sha256(path), expected)
        finally:
            os.unlink(path)

    def test_large_file_streaming(self) -> None:
        data = b"x" * (4 << 20)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        try:
            h = _compute_hash_sha256(path)
            self.assertEqual(len(h), 64)
            self.assertEqual(h, hashlib.sha256(data).hexdigest())
        finally:
            os.unlink(path)

    def test_dispatch_sha256(self) -> None:
        data = b"test"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        try:
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(_compute_hash_dispatch(path, "sha256"), expected)
        finally:
            os.unlink(path)

    def test_dispatch_unsupported_algorithm(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            with self.assertRaises(NotImplementedError):
                _compute_hash_dispatch(path, "blake3")
        finally:
            os.unlink(path)


class TestWriteManifest(TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_ckpt(self, files: dict[str, bytes]) -> str:
        for name, content in files.items():
            with open(os.path.join(self.tmpdir, name), "wb") as f:
                f.write(content)
        return self.tmpdir

    def test_write_and_load_manifest(self) -> None:
        self._create_ckpt({
            "model.pt": b"model data",
            "metadata.json": b'{"key": "val"}',
        })
        write_manifest(self.tmpdir)

        with open(os.path.join(self.tmpdir, MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["granularity"], "file")
        self.assertEqual(manifest["algorithm"], "sha256")
        self.assertIn("model.pt", manifest["files"])
        self.assertEqual(len(manifest["files"]["model.pt"]), 64)

    def test_manifest_excludes_itself(self) -> None:
        self._create_ckpt({"model.pt": b"data"})
        write_manifest(self.tmpdir)
        with open(os.path.join(self.tmpdir, MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
        self.assertNotIn(MANIFEST_FILENAME, manifest["files"])

    def test_manifest_sorted(self) -> None:
        self._create_ckpt({
            "c_file.pt": b"c",
            "a_file.pt": b"a",
            "b_file.pt": b"b",
        })
        write_manifest(self.tmpdir)
        with open(os.path.join(self.tmpdir, MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
        keys = list(manifest["files"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_write_with_explicit_algorithm(self) -> None:
        self._create_ckpt({"model.pt": b"data"})
        write_manifest(self.tmpdir, algorithm="sha256")
        with open(os.path.join(self.tmpdir, MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["algorithm"], "sha256")

    def test_write_with_unsupported_algorithm(self) -> None:
        self._create_ckpt({"model.pt": b"data"})
        with self.assertRaises(NotImplementedError):
            write_manifest(self.tmpdir, algorithm="blake3")

    def test_write_with_unsupported_granularity(self) -> None:
        self._create_ckpt({"model.pt": b"data"})
        with self.assertRaises(NotImplementedError):
            write_manifest(self.tmpdir, granularity="tensor")


class TestVerifyManifest(TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_verify_passes(self) -> None:
        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"model data")
        write_manifest(self.tmpdir)
        verify_manifest(self.tmpdir)

    def test_verify_detects_corruption(self) -> None:
        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"original data")
        write_manifest(self.tmpdir)
        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"corrupted data")
        with self.assertRaises(CheckpointingException):
            verify_manifest(self.tmpdir)

    def test_verify_detects_missing_file(self) -> None:
        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"data")
        with open(os.path.join(self.tmpdir, "extra.pt"), "wb") as f:
            f.write(b"extra")
        write_manifest(self.tmpdir)
        os.unlink(os.path.join(self.tmpdir, "extra.pt"))
        with self.assertRaises(CheckpointingException):
            verify_manifest(self.tmpdir)

    def test_verify_collects_all_mismatches(self) -> None:
        for name, content in {"a.pt": b"a original", "b.pt": b"b original"}.items():
            with open(os.path.join(self.tmpdir, name), "wb") as f:
                f.write(content)
        write_manifest(self.tmpdir)
        for name, content in {"a.pt": b"a corrupted", "b.pt": b"b corrupted"}.items():
            with open(os.path.join(self.tmpdir, name), "wb") as f:
                f.write(content)
        try:
            verify_manifest(self.tmpdir)
            self.fail("Should have raised")
        except CheckpointingException as exc:
            msg = str(exc)
            self.assertIn("a.pt", msg)
            self.assertIn("b.pt", msg)

    def test_verify_missing_manifest_raises_file_not_found(self) -> None:
        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"data")
        with self.assertRaises(FileNotFoundError):
            verify_manifest(self.tmpdir)

    def test_verify_reads_algorithm_from_manifest(self) -> None:
        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"data")
        write_manifest(self.tmpdir, algorithm="sha256")
        verify_manifest(self.tmpdir)


class TestEndToEnd(TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_corrupt_restore_cycle(self) -> None:
        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"original model")
        with open(os.path.join(self.tmpdir, ".metadata"), "wb") as f:
            f.write(b'{"version": 1}')

        write_manifest(self.tmpdir)
        verify_manifest(self.tmpdir)

        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"corrupted model")
        with self.assertRaises(CheckpointingException):
            verify_manifest(self.tmpdir)

        with open(os.path.join(self.tmpdir, "model.pt"), "wb") as f:
            f.write(b"original model")
        verify_manifest(self.tmpdir)


if __name__ == "__main__":
    run_tests()
