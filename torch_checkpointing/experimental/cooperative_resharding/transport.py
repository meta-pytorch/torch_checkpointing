# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded HTTP/1.1 control and byte-range transport for cooperative loading."""

from __future__ import annotations

import hmac
import http.client
import http.server
import json
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .shared_memory import (
    AmbiguousRangeError,
    RangeNotFoundError,
    RangeNotReadyError,
    RangeSpec,
    readinto_exact,
    SegmentIndex,
    SegmentSlice,
)

PROTOCOL_HEADER: str = "X-Coop-Protocol"
LOAD_TOKEN_HEADER: str = "X-Coop-Load-Token"
DEFAULT_MAX_CONTROL_BODY_BYTES: int = 16 * 1024 * 1024
DEFAULT_MAX_CONTROL_STORAGE_BYTES: int = 512 * 1024 * 1024
DEFAULT_MAX_FETCH_RANGES: int = 4096
DEFAULT_MAX_FETCH_BYTES: int = 1 << 40
_SERVER_POLL_INTERVAL_SECONDS: float = 0.05
_Value = TypeVar("_Value")


class TransportError(RuntimeError):
    pass


class ProtocolMismatchError(TransportError):
    pass


class RemoteLoadError(TransportError):
    pass


@dataclass(frozen=True)
class RangeRequest:
    file_id: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        RangeSpec(self.file_id, self.offset, self.length)

    def to_spec(self) -> RangeSpec:
        return RangeSpec(self.file_id, self.offset, self.length)


class _ServerState:
    def __init__(
        self,
        *,
        segment_index: SegmentIndex,
        protocol_version: int,
        load_token: str,
        max_control_body_bytes: int,
        max_control_storage_bytes: int,
        max_fetch_ranges: int,
        max_fetch_bytes: int,
        max_poll_seconds: float,
        keepalive_idle_seconds: float,
        request_io_seconds: float,
        max_requests_per_connection: int,
    ) -> None:
        if protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not load_token:
            raise ValueError("load_token must not be empty")
        if max_control_body_bytes <= 0:
            raise ValueError("max_control_body_bytes must be positive")
        if max_control_storage_bytes <= 0:
            raise ValueError("max_control_storage_bytes must be positive")
        if max_fetch_ranges <= 0:
            raise ValueError("max_fetch_ranges must be positive")
        if max_fetch_bytes <= 0:
            raise ValueError("max_fetch_bytes must be positive")
        if max_poll_seconds < 0:
            raise ValueError("max_poll_seconds must be non-negative")
        if keepalive_idle_seconds <= 0:
            raise ValueError("keepalive_idle_seconds must be positive")
        if request_io_seconds <= 0:
            raise ValueError("request_io_seconds must be positive")
        if max_requests_per_connection <= 0:
            raise ValueError("max_requests_per_connection must be positive")
        self.segment_index = segment_index
        self.protocol_version = protocol_version
        self.load_token = load_token
        self.max_control_body_bytes = max_control_body_bytes
        self.max_control_storage_bytes = max_control_storage_bytes
        self.max_fetch_ranges = max_fetch_ranges
        self.max_fetch_bytes = max_fetch_bytes
        self.max_poll_seconds = max_poll_seconds
        self.keepalive_idle_seconds = keepalive_idle_seconds
        self.request_io_seconds = request_io_seconds
        self.max_requests_per_connection = max_requests_per_connection
        self.blobs: dict[str, bytes] = {}
        self.blob_bytes = 0
        self.first_error: str | None = None
        self.first_error_publisher: str | None = None
        self.stopping = False
        self.condition = threading.Condition()

    def put_blob(self, tag: str, value: bytes) -> None:
        _validate_tag(tag)
        if len(value) > self.max_control_body_bytes:
            raise ValueError(
                f"blob is {len(value)} bytes, limit is "
                f"{self.max_control_body_bytes} bytes"
            )
        with self.condition:
            if self.stopping:
                raise RuntimeError("node server is stopping")
            previous = self.blobs.get(tag)
            previous_bytes = (
                _blob_storage_bytes(previous) if previous is not None else 0
            )
            updated_bytes = (
                self.blob_bytes - previous_bytes + _blob_storage_bytes(value)
            )
            if updated_bytes > self.max_control_storage_bytes:
                raise ValueError(
                    f"stored control blobs need {updated_bytes} bytes, limit is "
                    f"{self.max_control_storage_bytes} bytes"
                )
            self.blobs[tag] = bytes(value)
            self.blob_bytes = updated_bytes
            self.condition.notify_all()

    def get_blob(self, tag: str, timeout: float = 0) -> bytes | None:
        _validate_tag(tag)
        return self._wait_for(lambda: self.blobs.get(tag), timeout)

    def delete_blob(self, tag: str) -> bool:
        _validate_tag(tag)
        with self.condition:
            if self.stopping:
                raise RuntimeError("node server is stopping")
            value = self.blobs.pop(tag, None)
            if value is None:
                return False
            self.blob_bytes -= _blob_storage_bytes(value)
            return True

    def publish_error(self, message: str, publisher_id: str | None = None) -> bool:
        encoded = message.encode("utf-8")
        if len(encoded) > self.max_control_body_bytes:
            raise ValueError(
                f"error is {len(encoded)} bytes, limit is "
                f"{self.max_control_body_bytes} bytes"
            )
        with self.condition:
            if self.stopping:
                raise RuntimeError("node server is stopping")
            if self.first_error is not None:
                return publisher_id is not None and (
                    publisher_id == self.first_error_publisher
                )
            self.first_error = message
            self.first_error_publisher = publisher_id
            self.condition.notify_all()
            return True

    def get_error(self, timeout: float = 0) -> str | None:
        return self._wait_for(lambda: self.first_error, timeout)

    def _wait_for(
        self, getter: Callable[[], _Value | None], timeout: float
    ) -> _Value | None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self.condition:
            value = getter()
            while value is None and not self.stopping:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
                value = getter()
            return value

    def stop(self) -> None:
        with self.condition:
            self.stopping = True
            self.blobs.clear()
            self.blob_bytes = 0
            self.first_error = None
            self.first_error_publisher = None
            self.condition.notify_all()


class _BoundedHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[http.server.BaseHTTPRequestHandler],
        *,
        state: _ServerState,
        worker_count: int,
        pending_requests: int,
        thread_name_prefix: str,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if pending_requests < 0:
            raise ValueError("pending_requests must be non-negative")
        self.state = state
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=thread_name_prefix,
        )
        self._request_slots = threading.BoundedSemaphore(
            worker_count + pending_requests
        )
        self._stopping = threading.Event()
        self._sockets: set[socket.socket] = set()
        self._sockets_lock = threading.Lock()
        self.request_queue_size = max(worker_count + pending_requests, 16)
        try:
            super().__init__(server_address, handler_class)
        except Exception:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    def process_request(self, request: Any, client_address: Any) -> None:
        with self._sockets_lock:
            self._sockets.add(request)
        while not self._stopping.is_set():
            if self._request_slots.acquire(timeout=0.1):
                break
        else:
            self._forget_socket(request)
            self.shutdown_request(request)
            return
        try:
            self._executor.submit(self._process_request, request, client_address)
        except Exception:
            self._request_slots.release()
            self._forget_socket(request)
            self.shutdown_request(request)
            raise

    def _process_request(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        try:
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        except Exception:
            self.handle_error(request, client_address)
            self.shutdown_request(request)
        finally:
            self._forget_socket(request)
            self._request_slots.release()

    def initiate_shutdown(self) -> None:
        self._stopping.set()
        with self._sockets_lock:
            sockets = tuple(self._sockets)
        for request in sockets:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _forget_socket(self, request: socket.socket) -> None:
        with self._sockets_lock:
            self._sockets.discard(request)

    def handle_error(self, request: Any, client_address: Any) -> None:
        if not self._stopping.is_set():
            super().handle_error(request, client_address)

    def server_close(self) -> None:
        super().server_close()
        self._executor.shutdown(wait=True, cancel_futures=True)


class _IPv6BoundedHTTPServer(_BoundedHTTPServer):
    address_family = socket.AF_INET6


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PythonCooperativeLoader/1"
    _serves_control = False
    _serves_data = False

    def setup(self) -> None:
        self._handled_requests = 0
        super().setup()

    def handle_one_request(self) -> None:
        self.connection.settimeout(self.state.keepalive_idle_seconds)
        super().handle_one_request()

    def parse_request(self) -> bool:
        self.connection.settimeout(self.state.request_io_seconds)
        parsed = super().parse_request()
        if parsed:
            self._handled_requests += 1
            if self._handled_requests >= self.state.max_requests_per_connection:
                self.close_connection = True
        return parsed

    @property
    def state(self) -> _ServerState:
        server = self.server
        if not isinstance(server, _BoundedHTTPServer):
            raise RuntimeError("unexpected HTTP server type")
        return server.state

    def do_GET(self) -> None:
        if not self._validate_protocol():
            return
        parsed = urlsplit(self.path)
        if self._serves_control and parsed.path == "/v1/health":
            self._send_json(
                http.client.OK,
                {
                    "ok": True,
                    "protocol_version": self.state.protocol_version,
                },
            )
            return
        if self._serves_control and parsed.path.startswith("/v1/blob/"):
            self._get_blob(unquote(parsed.path.removeprefix("/v1/blob/")), parsed.query)
            return
        if self._serves_control and parsed.path == "/v1/error":
            self._get_error(parsed.query)
            return
        self._send_json(
            http.client.NOT_FOUND,
            {"error": "unknown endpoint"},
            close=True,
        )

    def do_PUT(self) -> None:
        if not self._validate_protocol():
            return
        parsed = urlsplit(self.path)
        if self._serves_control and parsed.path.startswith("/v1/blob/"):
            self._put_blob(unquote(parsed.path.removeprefix("/v1/blob/")))
            return
        self._send_json(
            http.client.NOT_FOUND,
            {"error": "unknown endpoint"},
            close=True,
        )

    def do_POST(self) -> None:
        if not self._validate_protocol():
            return
        parsed = urlsplit(self.path)
        if self._serves_control and parsed.path == "/v1/error":
            self._post_error()
            return
        if self._serves_data and parsed.path == "/v1/fetch":
            self._post_fetch()
            return
        if self._serves_data and parsed.path == "/v1/resolve":
            self._post_resolve()
            return
        self._send_json(
            http.client.NOT_FOUND,
            {"error": "unknown endpoint"},
            close=True,
        )

    def do_DELETE(self) -> None:
        if not self._validate_protocol():
            return
        parsed = urlsplit(self.path)
        if self._serves_control and parsed.path.startswith("/v1/blob/"):
            tag = unquote(parsed.path.removeprefix("/v1/blob/"))
            try:
                deleted = self.state.delete_blob(tag)
            except ValueError as error:
                self._send_json(http.client.BAD_REQUEST, {"error": str(error)})
                return
            self._send_json(http.client.OK, {"deleted": deleted})
            return
        self._send_json(
            http.client.NOT_FOUND,
            {"error": "unknown endpoint"},
            close=True,
        )

    def _validate_protocol(self) -> bool:
        expected_version = str(self.state.protocol_version)
        received_version = self.headers.get(PROTOCOL_HEADER, "")
        received_token = self.headers.get(LOAD_TOKEN_HEADER, "")
        if received_version == expected_version and hmac.compare_digest(
            received_token, self.state.load_token
        ):
            return True
        self.close_connection = True
        self._send_json(
            http.client.CONFLICT,
            {"error": "protocol version or load token mismatch"},
            close=True,
        )
        return False

    def _get_blob(self, tag: str, query: str) -> None:
        try:
            timeout = min(_poll_seconds(query), self.state.max_poll_seconds)
            value = self.state.get_blob(tag, timeout)
        except ValueError as error:
            self._send_json(http.client.BAD_REQUEST, {"error": str(error)})
            return
        if value is None:
            self._send_json(http.client.ACCEPTED, {"ready": False})
            return
        self._send_bytes(http.client.OK, value, "application/octet-stream")

    def _put_blob(self, tag: str) -> None:
        try:
            value = self._read_body()
            self.state.put_blob(tag, value)
        except ValueError as error:
            self._send_json(http.client.BAD_REQUEST, {"error": str(error)})
            return
        self._send_json(http.client.CREATED, {"stored": True})

    def _get_error(self, query: str) -> None:
        try:
            timeout = min(_poll_seconds(query), self.state.max_poll_seconds)
            message = self.state.get_error(timeout)
        except ValueError as error:
            self._send_json(http.client.BAD_REQUEST, {"error": str(error)})
            return
        if message is None:
            self._send_json(http.client.ACCEPTED, {"ready": False})
            return
        self._send_json(http.client.OK, {"message": message})

    def _post_error(self) -> None:
        try:
            payload = self._read_json()
            message = payload.get("message")
            if not isinstance(message, str) or not message:
                raise ValueError("message must be a non-empty string")
            publisher_id = payload.get("publisher_id")
            if not isinstance(publisher_id, str) or not publisher_id:
                raise ValueError("publisher_id must be a non-empty string")
            published = self.state.publish_error(message, publisher_id)
        except ValueError as error:
            self._send_json(http.client.BAD_REQUEST, {"error": str(error)})
            return
        self._send_json(
            http.client.CREATED if published else http.client.OK,
            {"published": published},
        )

    def _post_fetch(self) -> None:
        try:
            requests = self._parse_range_requests()
            expected_bytes = sum(request.length for request in requests)
            if expected_bytes > self.state.max_fetch_bytes:
                raise ValueError(
                    f"fetch requests {expected_bytes} bytes, limit is "
                    f"{self.state.max_fetch_bytes} bytes"
                )
            first_error = self.state.get_error()
            if first_error is not None:
                self._send_json(
                    http.client.SERVICE_UNAVAILABLE,
                    {"error": first_error},
                )
                return
            lease = self.state.segment_index.acquire_many(
                [request.to_spec() for request in requests]
            )
        except RangeNotReadyError as error:
            self.close_connection = True
            self._send_json(
                http.client.ACCEPTED,
                {"ready": False, "detail": str(error)},
                close=True,
            )
            return
        except RangeNotFoundError as error:
            self._send_json(http.client.NOT_FOUND, {"error": str(error)})
            return
        except (
            AmbiguousRangeError,
            ValueError,
        ) as error:
            self._send_json(http.client.BAD_REQUEST, {"error": str(error)})
            return

        with lease:
            self.send_response(http.client.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(expected_bytes))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            for resolved_range in lease.ranges:
                for segment in resolved_range:
                    _send_file_range(self.connection, segment)

    def _post_resolve(self) -> None:
        try:
            requests = self._parse_range_requests()
            expected_bytes = sum(request.length for request in requests)
            if expected_bytes > self.state.max_fetch_bytes:
                raise ValueError(
                    f"resolve requests {expected_bytes} bytes, limit is "
                    f"{self.state.max_fetch_bytes} bytes"
                )
            first_error = self.state.get_error()
            if first_error is not None:
                self._send_json(
                    http.client.SERVICE_UNAVAILABLE,
                    {"error": first_error},
                )
                return
            lease = self.state.segment_index.acquire_many(
                [request.to_spec() for request in requests]
            )
        except RangeNotReadyError as error:
            self.close_connection = True
            self._send_json(
                http.client.ACCEPTED,
                {"ready": False, "detail": str(error)},
                close=True,
            )
            return
        except RangeNotFoundError as error:
            self._send_json(http.client.NOT_FOUND, {"error": str(error)})
            return
        except (AmbiguousRangeError, ValueError) as error:
            self._send_json(http.client.BAD_REQUEST, {"error": str(error)})
            return

        with lease:
            self._send_json(
                http.client.OK,
                {
                    "ranges": [
                        [
                            {
                                "path": str(segment.path),
                                "file_offset": segment.file_offset,
                                "length": segment.length,
                            }
                            for segment in resolved_range
                        ]
                        for resolved_range in lease.ranges
                    ]
                },
            )

    def _parse_range_requests(self) -> list[RangeRequest]:
        payload = self._read_json()
        raw_ranges = payload.get("ranges")
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise ValueError("ranges must be a non-empty list")
        if len(raw_ranges) > self.state.max_fetch_ranges:
            raise ValueError(
                f"fetch has {len(raw_ranges)} ranges, limit is "
                f"{self.state.max_fetch_ranges}"
            )
        requests: list[RangeRequest] = []
        for raw_range in raw_ranges:
            if not isinstance(raw_range, dict):
                raise ValueError("each range must be an object")
            file_id = raw_range.get("file_id")
            offset = raw_range.get("offset")
            length = raw_range.get("length")
            if (
                not isinstance(file_id, str)
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
            ):
                raise ValueError("range fields have invalid types")
            requests.append(RangeRequest(file_id, offset, length))
        return requests

    def _read_json(self) -> dict[str, object]:
        payload = json.loads(self._read_body().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 0 or length > self.state.max_control_body_bytes:
            self.close_connection = True
            raise ValueError(
                f"request body is {length} bytes, limit is "
                f"{self.state.max_control_body_bytes} bytes"
            )
        value = self.rfile.read(length)
        if len(value) != length:
            self.close_connection = True
            raise ValueError(f"short request body: expected {length}, got {len(value)}")
        return value

    def _send_json(
        self,
        status: int,
        value: Mapping[str, object],
        *,
        close: bool = False,
    ) -> None:
        body = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        if len(body) > self.state.max_control_body_bytes:
            status = http.client.REQUEST_ENTITY_TOO_LARGE
            body = b'{"error":"control response exceeds configured size limit"}'
        self._send_bytes(status, body, "application/json", close=close)

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        close: bool = False,
    ) -> None:
        if close:
            self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
        elif self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _ControlRequestHandler(_RequestHandler):
    disable_nagle_algorithm = True
    _serves_control = True


class _DataRequestHandler(_RequestHandler):
    _serves_data = True


class NodeServer:
    """Isolated control and range-data listeners backed by shared node state.

    ``port``, ``worker_count``, and ``pending_requests`` are compatibility
    aliases for the control port, data worker count, and data pending limit.
    New callers should use the explicit control/data parameters.
    """

    def __init__(
        self,
        segment_index: SegmentIndex,
        *,
        protocol_version: int,
        load_token: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        worker_count: int | None = None,
        pending_requests: int | None = None,
        control_port: int = 0,
        data_port: int = 0,
        control_worker_count: int = 64,
        control_pending_requests: int | None = None,
        data_worker_count: int | None = None,
        data_pending_requests: int | None = None,
        max_control_body_bytes: int = DEFAULT_MAX_CONTROL_BODY_BYTES,
        max_control_storage_bytes: int = DEFAULT_MAX_CONTROL_STORAGE_BYTES,
        max_fetch_ranges: int = DEFAULT_MAX_FETCH_RANGES,
        max_fetch_bytes: int = DEFAULT_MAX_FETCH_BYTES,
        max_poll_seconds: float = 5.0,
        keepalive_idle_seconds: float = 0.25,
        request_io_seconds: float = 30.0,
        max_requests_per_connection: int = 16,
    ) -> None:
        if port is not None:
            if control_port != 0:
                raise ValueError("port and control_port cannot both be set")
            control_port = port
        if worker_count is not None:
            if data_worker_count is not None:
                raise ValueError(
                    "worker_count and data_worker_count cannot both be set"
                )
            data_worker_count = worker_count
        if pending_requests is not None:
            if data_pending_requests is not None:
                raise ValueError(
                    "pending_requests and data_pending_requests cannot both be set"
                )
            data_pending_requests = pending_requests
        if data_worker_count is None:
            data_worker_count = 64
        self._host = host
        self._control_port = control_port
        self._data_port = data_port
        self._control_worker_count = control_worker_count
        self._control_pending_requests = (
            control_worker_count
            if control_pending_requests is None
            else control_pending_requests
        )
        self._data_worker_count = data_worker_count
        self._data_pending_requests = (
            data_worker_count
            if data_pending_requests is None
            else data_pending_requests
        )
        self._state = _ServerState(
            segment_index=segment_index,
            protocol_version=protocol_version,
            load_token=load_token,
            max_control_body_bytes=max_control_body_bytes,
            max_control_storage_bytes=max_control_storage_bytes,
            max_fetch_ranges=max_fetch_ranges,
            max_fetch_bytes=max_fetch_bytes,
            max_poll_seconds=max_poll_seconds,
            keepalive_idle_seconds=keepalive_idle_seconds,
            request_io_seconds=request_io_seconds,
            max_requests_per_connection=max_requests_per_connection,
        )
        self._control_server: _BoundedHTTPServer | None = None
        self._data_server: _BoundedHTTPServer | None = None
        self._control_thread: threading.Thread | None = None
        self._data_thread: threading.Thread | None = None
        self._closed = False
        self._lifecycle_lock = threading.Lock()

    @property
    def control_base_url(self) -> str:
        return self._base_url(self._control_server, "control")

    @property
    def data_base_url(self) -> str:
        return self._base_url(self._data_server, "data")

    def _base_url(
        self,
        server: _BoundedHTTPServer | None,
        listener_name: str,
    ) -> str:
        if server is None:
            raise RuntimeError(f"node {listener_name} server has not been started")
        address = server.server_address
        host = address[0]
        port = address[1]
        formatted_host = f"[{host}]" if ":" in host else host
        return f"http://{formatted_host}:{port}"

    def start(self) -> NodeServer:
        with self._lifecycle_lock:
            if self._control_server is not None and self._data_server is not None:
                return self
            if self._closed:
                raise RuntimeError("node server cannot be restarted after close")
            server_type = (
                _IPv6BoundedHTTPServer if ":" in self._host else _BoundedHTTPServer
            )
            control_server: _BoundedHTTPServer | None = None
            data_server: _BoundedHTTPServer | None = None
            control_thread: threading.Thread | None = None
            data_thread: threading.Thread | None = None
            try:
                control_server = server_type(
                    (self._host, self._control_port),
                    _ControlRequestHandler,
                    state=self._state,
                    worker_count=self._control_worker_count,
                    pending_requests=self._control_pending_requests,
                    thread_name_prefix="coop-control-http",
                )
                data_server = server_type(
                    (self._host, self._data_port),
                    _DataRequestHandler,
                    state=self._state,
                    worker_count=self._data_worker_count,
                    pending_requests=self._data_pending_requests,
                    thread_name_prefix="coop-data-http",
                )
                control_thread = threading.Thread(
                    target=control_server.serve_forever,
                    kwargs={"poll_interval": _SERVER_POLL_INTERVAL_SECONDS},
                    name="coop-control-http-server",
                    daemon=True,
                )
                data_thread = threading.Thread(
                    target=data_server.serve_forever,
                    kwargs={"poll_interval": _SERVER_POLL_INTERVAL_SECONDS},
                    name="coop-data-http-server",
                    daemon=True,
                )
                control_thread.start()
                data_thread.start()
            except BaseException:
                self._closed = True
                self._state.stop()
                try:
                    _close_http_listeners(
                        (
                            (control_server, control_thread),
                            (data_server, data_thread),
                        )
                    )
                except Exception:
                    pass
                raise
            self._control_server = control_server
            self._data_server = data_server
            self._control_thread = control_thread
            self._data_thread = data_thread
        return self

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            listeners = (
                (self._control_server, self._control_thread),
                (self._data_server, self._data_thread),
            )
            self._control_server = None
            self._data_server = None
            self._control_thread = None
            self._data_thread = None
            self._closed = True
            self._state.stop()
        _close_http_listeners(listeners)

    def put_blob(self, tag: str, value: bytes) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("node server is closed")
            self._state.put_blob(tag, value)

    def get_blob(self, tag: str, timeout: float = 0) -> bytes | None:
        return self._state.get_blob(tag, timeout)

    def publish_error(self, message: str) -> bool:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("node server is closed")
            return self._state.publish_error(message)

    def get_error(self, timeout: float = 0) -> str | None:
        return self._state.get_error(timeout)

    def __enter__(self) -> NodeServer:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _close_http_listeners(
    listeners: Sequence[tuple[_BoundedHTTPServer | None, threading.Thread | None]],
) -> None:
    errors: list[Exception] = []
    for server, _ in listeners:
        if server is not None:
            _run_shutdown_step(errors, server.initiate_shutdown)
    for server, thread in listeners:
        if server is not None and thread is not None and thread.ident is not None:
            _run_shutdown_step(errors, server.shutdown)
    for server, _ in listeners:
        if server is not None:
            _run_shutdown_step(errors, server.server_close)
    for _, thread in listeners:
        if thread is not None and thread.ident is not None:
            _run_shutdown_step(errors, thread.join)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise RuntimeError(
            "multiple cooperative HTTP listener shutdown failures: "
            + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        ) from errors[0]


def _run_shutdown_step(
    errors: list[Exception],
    step: Callable[[], None],
) -> None:
    try:
        step()
    except Exception as error:
        errors.append(error)


class NodeClient:
    """Thread-safe persistent client; use one instance per concurrent worker."""

    def __init__(
        self,
        base_url: str,
        *,
        protocol_version: int,
        load_token: str,
        request_timeout: float = 30.0,
        max_attempts: int = 4,
        retry_delay: float = 0.05,
        max_control_body_bytes: int = DEFAULT_MAX_CONTROL_BODY_BYTES,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname is None:
            raise ValueError("base_url must be an HTTP URL with a hostname")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a path, query, or fragment")
        if protocol_version <= 0:
            raise ValueError("protocol_version must be positive")
        if not load_token:
            raise ValueError("load_token must not be empty")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._protocol_version = protocol_version
        self._load_token = load_token
        self._request_timeout = request_timeout
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._max_control_body_bytes = max_control_body_bytes
        self._connection: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    def health(self) -> dict[str, object]:
        status, body = self._small_request("GET", "/v1/health")
        self._expect_status(status, {http.client.OK}, body)
        return _decode_json_object(body)

    def put_blob(self, tag: str, value: bytes) -> None:
        _validate_tag(tag)
        status, body = self._small_request(
            "PUT",
            f"/v1/blob/{quote(tag, safe='')}",
            body=bytes(value),
            content_type="application/octet-stream",
        )
        self._expect_status(status, {http.client.CREATED}, body)

    def get_blob(self, tag: str, *, timeout: float = 0) -> bytes | None:
        _validate_tag(tag)
        value = self._poll_small_value(
            f"/v1/blob/{quote(tag, safe='')}", timeout, decode_json=False
        )
        if value is None or isinstance(value, bytes):
            return value
        raise TransportError("blob response unexpectedly contained JSON")

    def delete_blob(self, tag: str) -> bool:
        _validate_tag(tag)
        status, body, retried_after_transport_error = (
            self._small_request_with_retry_state(
                "DELETE",
                f"/v1/blob/{quote(tag, safe='')}",
            )
        )
        self._expect_status(status, {http.client.OK}, body)
        deleted = _decode_json_object(body).get("deleted")
        if not isinstance(deleted, bool):
            raise TransportError(
                "delete response did not contain a boolean deleted field"
            )
        # A retry can observe an absent blob after the first DELETE took effect
        # but its response was lost.
        return deleted or retried_after_transport_error

    def publish_error(self, message: str) -> bool:
        body = json.dumps(
            {"message": message, "publisher_id": uuid.uuid4().hex},
            separators=(",", ":"),
        ).encode()
        status, response_body = self._small_request(
            "POST", "/v1/error", body=body, content_type="application/json"
        )
        self._expect_status(
            status, {http.client.OK, http.client.CREATED}, response_body
        )
        published = _decode_json_object(response_body).get("published")
        if not isinstance(published, bool):
            raise TransportError(
                "error response did not contain a boolean published field"
            )
        return published

    def get_error(self, *, timeout: float = 0) -> str | None:
        payload = self._poll_small_value("/v1/error", timeout, decode_json=True)
        if payload is None:
            return None
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str):
            raise TransportError("error response did not contain a string message")
        return message

    def fetch_into(
        self,
        requests: Sequence[RangeRequest],
        destination: memoryview | bytearray,
        *,
        ready_timeout: float | None = None,
    ) -> int:
        if not requests:
            raise ValueError("at least one range is required")
        view = (
            destination
            if isinstance(destination, memoryview)
            else memoryview(destination)
        )
        release_view = not isinstance(destination, memoryview)
        try:
            if view.readonly or not view.c_contiguous:
                raise TypeError("destination must be writable and C-contiguous")
            byte_view = view.cast("B")
            try:
                expected_bytes = sum(request.length for request in requests)
                if len(byte_view) != expected_bytes:
                    raise ValueError(
                        f"destination has {len(byte_view)} bytes, "
                        f"expected {expected_bytes}"
                    )
                body = self._encode_range_requests(requests)
                return self._fetch_with_retries(body, byte_view, ready_timeout)
            finally:
                if byte_view is not view:
                    byte_view.release()
        finally:
            if release_view:
                view.release()

    def resolve_ranges(
        self,
        requests: Sequence[RangeRequest],
        *,
        ready_timeout: float | None = None,
    ) -> tuple[tuple[SegmentSlice, ...], ...]:
        body = self._encode_range_requests(requests)
        deadline = _ready_deadline(ready_timeout)
        last_error: Exception | None = None
        attempt = 0
        while True:
            try:
                resolved = self._resolve_once(body, requests)
                if resolved is not None:
                    return resolved
                last_error = RangeNotReadyError("requested ranges are not ready")
            except (EOFError, http.client.HTTPException, OSError) as error:
                last_error = error
                with self._lock:
                    self._close_connection()
            if not self._can_retry(attempt, deadline):
                break
            self._sleep_before_retry(attempt, deadline)
            attempt += 1
        if isinstance(last_error, RangeNotReadyError):
            raise last_error
        raise TransportError("range resolution failed after retries") from last_error

    def close(self) -> None:
        with self._lock:
            self._close_connection()

    def _fetch_with_retries(
        self, body: bytes, destination: memoryview, ready_timeout: float | None
    ) -> int:
        deadline = _ready_deadline(ready_timeout)
        last_error: Exception | None = None
        attempt = 0
        while True:
            try:
                if self._fetch_once(body, destination):
                    return len(destination)
                last_error = RangeNotReadyError("requested ranges are not ready")
            except (EOFError, http.client.HTTPException, OSError) as error:
                last_error = error
                with self._lock:
                    self._close_connection()
            if not self._can_retry(attempt, deadline):
                break
            self._sleep_before_retry(attempt, deadline)
            attempt += 1
        if isinstance(last_error, RangeNotReadyError):
            raise last_error
        raise TransportError("fetch failed after retries") from last_error

    def _fetch_once(self, body: bytes, destination: memoryview) -> bool:
        status, response = self._open_response(
            "POST", "/v1/fetch", body, "application/json"
        )
        with response:
            if status == http.client.ACCEPTED:
                self._read_small_response(response)
                return False
            if status in (http.client.SERVICE_UNAVAILABLE, http.client.CONFLICT):
                self._raise_coordination_error(status, response)
            if status == http.client.NOT_FOUND:
                payload = self._read_small_response(response)
                raise RangeNotFoundError(_error_message(payload))
            if status != http.client.OK:
                payload = self._read_small_response(response)
                raise TransportError(
                    f"fetch failed with HTTP {status}: {_error_message(payload)}"
                )
            self._validate_fetch_length(response, len(destination))
            readinto_exact(response, destination)
            return True

    def _resolve_once(
        self, body: bytes, requests: Sequence[RangeRequest]
    ) -> tuple[tuple[SegmentSlice, ...], ...] | None:
        status, response = self._open_response(
            "POST", "/v1/resolve", body, "application/json"
        )
        with response:
            if status == http.client.ACCEPTED:
                self._read_small_response(response)
                return None
            if status in (http.client.SERVICE_UNAVAILABLE, http.client.CONFLICT):
                self._raise_coordination_error(status, response)
            payload = self._read_small_response(response)
            if status == http.client.NOT_FOUND:
                raise RangeNotFoundError(_error_message(payload))
            if status != http.client.OK:
                raise TransportError(
                    f"range resolution failed with HTTP {status}: "
                    f"{_error_message(payload)}"
                )
            return _decode_segment_ranges(payload, requests)

    def _raise_coordination_error(self, status: int, response: _LockedResponse) -> None:
        message = _error_message(self._read_small_response(response))
        if status == http.client.SERVICE_UNAVAILABLE:
            raise RemoteLoadError(message)
        raise ProtocolMismatchError(message)

    def _validate_fetch_length(
        self, response: _LockedResponse, expected_bytes: int
    ) -> None:
        raw_length = response.getheader("Content-Length")
        try:
            response_length = int(raw_length) if raw_length is not None else -1
        except ValueError as error:
            raise TransportError(
                "fetch response has an invalid Content-Length"
            ) from error
        if response_length != expected_bytes:
            self._close_connection()
            raise TransportError(
                "fetch response Content-Length does not match destination"
            )

    def _encode_range_requests(self, requests: Sequence[RangeRequest]) -> bytes:
        if not requests:
            raise ValueError("at least one range is required")
        body = json.dumps(
            {
                "ranges": [
                    {
                        "file_id": request.file_id,
                        "offset": request.offset,
                        "length": request.length,
                    }
                    for request in requests
                ]
            },
            separators=(",", ":"),
        ).encode()
        if len(body) > self._max_control_body_bytes:
            raise ValueError("encoded range request exceeds control body limit")
        return body

    def _small_request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> tuple[int, bytes]:
        status, response_body, _ = self._small_request_with_retry_state(
            method,
            path,
            body=body,
            content_type=content_type,
        )
        return status, response_body

    def _small_request_with_retry_state(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> tuple[int, bytes, bool]:
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            retry_immediately = False
            try:
                status, response = self._open_response(method, path, body, content_type)
                with response:
                    payload = self._read_small_response(response)
                return status, payload, attempt > 0
            except (EOFError, http.client.HTTPException, OSError) as error:
                last_error = error
                # An idle keepalive can be closed by the server between requests.
                # Reconnect once without paying the general transient-error backoff.
                retry_immediately = attempt == 0 and isinstance(
                    error,
                    (EOFError, BrokenPipeError, ConnectionResetError),
                )
                with self._lock:
                    self._close_connection()
            if attempt + 1 < self._max_attempts:
                if not retry_immediately:
                    time.sleep(self._retry_delay * (2**attempt))
        raise TransportError(
            f"request {method} {path} failed after retries"
        ) from last_error

    def _open_response(
        self,
        method: str,
        path: str,
        body: bytes | None,
        content_type: str | None,
    ) -> tuple[int, _LockedResponse]:
        self._lock.acquire()
        try:
            connection = self._get_connection()
            headers = {
                PROTOCOL_HEADER: str(self._protocol_version),
                LOAD_TOKEN_HEADER: self._load_token,
            }
            if content_type is not None:
                headers["Content-Type"] = content_type
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
        except Exception:
            self._lock.release()
            raise
        return response.status, _LockedResponse(response, self._lock)

    def _poll_small_value(
        self, path: str, timeout: float, *, decode_json: bool
    ) -> bytes | dict[str, object] | None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            wait_ms = min(int(remaining * 1000), 5000)
            separator = "&" if "?" in path else "?"
            status, body = self._small_request(
                "GET", f"{path}{separator}wait_ms={wait_ms}"
            )
            if status == http.client.OK:
                return _decode_json_object(body) if decode_json else body
            self._expect_status(status, {http.client.ACCEPTED}, body)
            if remaining <= 0:
                return None

    def _read_small_response(self, response: _LockedResponse) -> bytes:
        raw_length = response.getheader("Content-Length")
        if raw_length is not None:
            try:
                response_length = int(raw_length)
            except ValueError as error:
                self._close_connection()
                raise TransportError(
                    "control response has an invalid Content-Length"
                ) from error
            if response_length > self._max_control_body_bytes:
                self._close_connection()
                raise TransportError("control response exceeds configured size limit")
        body = response.read(self._max_control_body_bytes + 1)
        if len(body) > self._max_control_body_bytes:
            self._close_connection()
            raise TransportError("control response exceeds configured size limit")
        return body

    def _expect_status(self, status: int, expected: set[int], body: bytes) -> None:
        if status in expected:
            return
        message = _error_message(body)
        if status == http.client.CONFLICT:
            raise ProtocolMismatchError(message)
        if status == http.client.SERVICE_UNAVAILABLE:
            raise RemoteLoadError(message)
        raise TransportError(f"unexpected HTTP {status}: {message}")

    def _get_connection(self) -> http.client.HTTPConnection:
        if self._connection is None:
            self._connection = http.client.HTTPConnection(
                self._host, self._port, timeout=self._request_timeout
            )
        return self._connection

    def _close_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _can_retry(self, attempt: int, deadline: float | None) -> bool:
        if deadline is not None:
            return time.monotonic() < deadline
        return attempt + 1 < self._max_attempts

    def _sleep_before_retry(self, attempt: int, deadline: float | None) -> None:
        delay = min(self._retry_delay * (2 ** min(attempt, 6)), 1.0)
        if deadline is not None:
            delay = min(delay, max(0.0, deadline - time.monotonic()))
        if delay > 0:
            time.sleep(delay)

    def __enter__(self) -> NodeClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _LockedResponse:
    """Delegates to HTTPResponse and releases the client lock on close."""

    def __init__(
        self, response: http.client.HTTPResponse, lock: threading.Lock
    ) -> None:
        self._response = response
        self._lock = lock
        self._closed = False

    @property
    def status(self) -> int:
        return self._response.status

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._response.getheader(name, default)

    def read(self, amount: int | None = None) -> bytes:
        return self._response.read(amount)

    def readinto(self, buffer: memoryview) -> int | None:
        return self._response.readinto(buffer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        finally:
            self._lock.release()

    def __enter__(self) -> _LockedResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _send_file_range(connection: socket.socket, segment: SegmentSlice) -> None:
    with segment.path.open("rb", buffering=0) as source:
        sent = connection.sendfile(
            source, offset=segment.file_offset, count=segment.length
        )
    if sent != segment.length:
        raise EOFError(f"sendfile stopped after {sent} of {segment.length} bytes")


def _poll_seconds(query: str) -> float:
    values = parse_qs(query).get("wait_ms", ["0"])
    try:
        wait_ms = int(values[-1])
    except ValueError as error:
        raise ValueError("wait_ms must be an integer") from error
    if wait_ms < 0:
        raise ValueError("wait_ms must be non-negative")
    return wait_ms / 1000


def _ready_deadline(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if timeout < 0:
        raise ValueError("ready_timeout must be non-negative")
    return time.monotonic() + timeout


def _decode_segment_ranges(
    body: bytes, requests: Sequence[RangeRequest]
) -> tuple[tuple[SegmentSlice, ...], ...]:
    payload = _decode_json_object(body)
    raw_ranges = payload.get("ranges")
    if not isinstance(raw_ranges, list) or len(raw_ranges) != len(requests):
        raise TransportError("resolved ranges do not match requested ranges")
    result: list[tuple[SegmentSlice, ...]] = []
    for request, raw_segments in zip(requests, raw_ranges):
        if not isinstance(raw_segments, list) or not raw_segments:
            raise TransportError("resolved range has no segments")
        segments: list[SegmentSlice] = []
        for raw_segment in raw_segments:
            segments.append(_decode_segment_slice(raw_segment))
        if sum(segment.length for segment in segments) != request.length:
            raise TransportError("resolved segment lengths do not cover request")
        result.append(tuple(segments))
    return tuple(result)


def _decode_segment_slice(value: object) -> SegmentSlice:
    if not isinstance(value, dict):
        raise TransportError("resolved segment must be an object")
    raw_path = value.get("path")
    file_offset = value.get("file_offset")
    length = value.get("length")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or isinstance(file_offset, bool)
        or not isinstance(file_offset, int)
        or file_offset < 0
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
    ):
        raise TransportError("resolved segment fields are invalid")
    path = Path(raw_path)
    if not path.is_absolute():
        raise TransportError("resolved segment path must be absolute")
    return SegmentSlice(path=path, file_offset=file_offset, length=length)


def _validate_tag(tag: str) -> None:
    if not tag:
        raise ValueError("blob tag must not be empty")
    if len(tag.encode("utf-8")) > 4096:
        raise ValueError("blob tag is too long")


def _blob_storage_bytes(value: bytes) -> int:
    return max(1, len(value))


def _decode_json_object(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TransportError("response is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TransportError("JSON response must be an object")
    return value


def _error_message(body: bytes) -> str:
    try:
        value = _decode_json_object(body)
    except TransportError:
        return body[:1024].decode("utf-8", errors="replace")
    message = value.get("error")
    return message if isinstance(message, str) else repr(value)
