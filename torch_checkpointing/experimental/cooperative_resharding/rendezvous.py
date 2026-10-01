# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small, namespaced rendezvous primitives for cooperative checkpoint loads."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

DEFAULT_MAX_BLOB_BYTES: int = 16 * 1024 * 1024
DEFAULT_MAX_ERROR_BYTES: int = 64 * 1024


@dataclass(frozen=True)
class RendezvousNamespace:
    """Uniquely identifies one cooperative-load protocol instance."""

    protocol_version: int
    job_id: str
    load_token: str

    def __post_init__(self) -> None:
        if self.protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.load_token:
            raise ValueError("load_token must not be empty")

    def key(self, kind: str, name: str = "") -> str:
        if not kind:
            raise ValueError("kind must not be empty")
        components = (
            "python-cooperative-loader",
            f"v{self.protocol_version}",
            quote(self.job_id, safe=""),
            quote(self.load_token, safe=""),
            quote(kind, safe=""),
        )
        prefix = "/".join(components)
        return f"{prefix}/{quote(name, safe='')}" if name else prefix


class Rendezvous(Protocol):
    """Facade for small control-plane values; not for tensor payloads."""

    def put_blob(
        self, namespace: RendezvousNamespace, name: str, value: bytes
    ) -> None: ...

    def get_blob(
        self,
        namespace: RendezvousNamespace,
        name: str,
        *,
        timeout: float | None = None,
    ) -> bytes | None: ...

    def publish_error(self, namespace: RendezvousNamespace, message: str) -> bool: ...

    def get_error(
        self,
        namespace: RendezvousNamespace,
        *,
        timeout: float | None = None,
    ) -> str | None: ...


class InMemoryRendezvous:
    """Thread-safe rendezvous implementation for tests and local loads."""

    def __init__(
        self,
        *,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_error_bytes: int = DEFAULT_MAX_ERROR_BYTES,
    ) -> None:
        _validate_size_limit("max_blob_bytes", max_blob_bytes)
        _validate_size_limit("max_error_bytes", max_error_bytes)
        self._max_blob_bytes = max_blob_bytes
        self._max_error_bytes = max_error_bytes
        self._values: dict[str, bytes] = {}
        self._condition = threading.Condition()

    def put_blob(self, namespace: RendezvousNamespace, name: str, value: bytes) -> None:
        _validate_name(name)
        encoded = bytes(value)
        _validate_payload_size("blob", encoded, self._max_blob_bytes)
        key = namespace.key("blob", name)
        with self._condition:
            self._values[key] = encoded
            self._condition.notify_all()

    def get_blob(
        self,
        namespace: RendezvousNamespace,
        name: str,
        *,
        timeout: float | None = None,
    ) -> bytes | None:
        _validate_name(name)
        return self._wait_for_key(namespace.key("blob", name), timeout)

    def publish_error(self, namespace: RendezvousNamespace, message: str) -> bool:
        encoded = message.encode("utf-8")
        _validate_payload_size("error", encoded, self._max_error_bytes)
        key = namespace.key("first-error")
        with self._condition:
            if key in self._values:
                return False
            self._values[key] = encoded
            self._condition.notify_all()
            return True

    def get_error(
        self,
        namespace: RendezvousNamespace,
        *,
        timeout: float | None = None,
    ) -> str | None:
        value = self._wait_for_key(namespace.key("first-error"), timeout)
        return value.decode("utf-8") if value is not None else None

    def _wait_for_key(self, key: str, timeout: float | None) -> bytes | None:
        deadline = _deadline(timeout)
        with self._condition:
            while key not in self._values:
                remaining = _remaining(deadline)
                if remaining == 0:
                    return None
                self._condition.wait(remaining)
            return self._values[key]


class C10dStore(Protocol):
    """The subset of ``torch.distributed.Store`` used by this module."""

    def set(self, key: str, value: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def check(self, keys: list[str]) -> bool: ...

    def compare_set(
        self, key: str, expected_value: bytes, desired_value: bytes
    ) -> bytes: ...


class C10dStoreRendezvous:
    """Namespaced facade over a c10d Store for small bootstrap values."""

    def __init__(
        self,
        store: C10dStore,
        *,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_error_bytes: int = DEFAULT_MAX_ERROR_BYTES,
    ) -> None:
        _validate_size_limit("max_blob_bytes", max_blob_bytes)
        _validate_size_limit("max_error_bytes", max_error_bytes)
        self._store = store
        self._max_blob_bytes = max_blob_bytes
        self._max_error_bytes = max_error_bytes

    def put_blob(self, namespace: RendezvousNamespace, name: str, value: bytes) -> None:
        _validate_name(name)
        encoded = bytes(value)
        _validate_payload_size("blob", encoded, self._max_blob_bytes)
        self._store.set(namespace.key("blob", name), encoded)

    def get_blob(
        self,
        namespace: RendezvousNamespace,
        name: str,
        *,
        timeout: float | None = None,
    ) -> bytes | None:
        _validate_name(name)
        return self._wait_for_key(namespace.key("blob", name), timeout)

    def publish_error(self, namespace: RendezvousNamespace, message: str) -> bool:
        encoded = message.encode("utf-8")
        _validate_payload_size("error", encoded, self._max_error_bytes)
        candidate = uuid.uuid4().hex.encode("ascii") + b"\n" + encoded
        stored = _coerce_bytes(
            self._store.compare_set(namespace.key("first-error"), b"", candidate)
        )
        return stored == candidate

    def get_error(
        self,
        namespace: RendezvousNamespace,
        *,
        timeout: float | None = None,
    ) -> str | None:
        value = self._wait_for_key(namespace.key("first-error"), timeout)
        if value is None:
            return None
        _, separator, message = value.partition(b"\n")
        if not separator:
            raise ValueError("stored rendezvous error has an invalid encoding")
        return message.decode("utf-8")

    def _wait_for_key(self, key: str, timeout: float | None) -> bytes | None:
        if timeout is None:
            return _coerce_bytes(self._store.get(key))
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        while not self._store.check([key]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.01, remaining))
        return _coerce_bytes(self._store.get(key))


def _coerce_bytes(value: bytes | str) -> bytes:
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _validate_name(name: str) -> None:
    if not name:
        raise ValueError("rendezvous value name must not be empty")


def _validate_size_limit(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_payload_size(kind: str, value: bytes, limit: int) -> None:
    if len(value) > limit:
        raise ValueError(f"{kind} is {len(value)} bytes, limit is {limit} bytes")


def _deadline(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    return time.monotonic() + timeout


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
