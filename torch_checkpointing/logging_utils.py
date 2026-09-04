# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextvars
import enum
import time
import typing as t
from typing import Protocol, runtime_checkable

from . import version


class StrEnum(enum.Enum):
    def __str__(self) -> str:
        return self.value


class EventType(StrEnum):
    CHECKPOINT_START = "checkpoint_start"
    CHECKPOINT_SUBPROCESS_BARRIER_START = "checkpoint_subprocess_barrier_start"
    CHECKPOINT_SUBPROCESS_BARRIER_END = "checkpoint_subprocess_barrier_end"
    CHECKPOINT_BARRIER_START = "checkpoint_barrier_start"
    CHECKPOINT_BARRIER_END = "checkpoint_barrier_end"
    CHECKPOINT_LOAD_BARRIER_START = "checkpoint_load_barrier_start"
    CHECKPOINT_LOAD_BARRIER_END = "checkpoint_load_barrier_end"
    CHECKPOINT_END = "checkpoint_end"
    CHECKPOINT_THREAD_START = "checkpoint_thread_start"
    CHECKPOINT_THREAD_END = "checkpoint_thread_end"

    CHECKPOINT_STAGING_START = "checkpoint_staging_start"
    CHECKPOINT_STAGING_END = "checkpoint_staging_end"
    CHECKPOINT_WRITE_START = "checkpoint_write_start"
    CHECKPOINT_WRITE_END = "checkpoint_write_end"

    CHECKPOINT_SUBPROCESS_INIT_START = "checkpoint_subprocess_init_start"
    CHECKPOINT_SUBPROCESS_INIT_END = "checkpoint_subprocess_init_end"
    CHECKPOINT_WAIT_FOR_REQUEST = "checkpoint_wait_for_request"

    CHECKPOINT_PERSISTENT_SAVE_MODEL_START = "checkpoint_persistent_save_model_start"
    CHECKPOINT_PERSISTENT_SAVE_MODEL_END = "checkpoint_persistent_save_model_end"
    CHECKPOINT_PERSISTENT_SAVE_OPTIM_START = "checkpoint_persistent_save_optim_start"
    CHECKPOINT_PERSISTENT_SAVE_OPTIM_END = "checkpoint_persistent_save_optim_end"
    CHECKPOINT_PERSISTENT_SAVE_DATA_LOADER_START = (
        "checkpoint_persistent_save_data_loader_start"
    )
    CHECKPOINT_PERSISTENT_SAVE_DATA_LOADER_END = (
        "checkpoint_persistent_save_data_loader_end"
    )
    CHECKPOINT_PERSISTENT_SAVE_ALL_START = "checkpoint_persistent_save_all_start"
    CHECKPOINT_PERSISTENT_SAVE_ALL_END = "checkpoint_persistent_save_all_end"
    CHECKPOINT_PERSISTENT_SAVE_OTHER_START = "checkpoint_persistent_save_other_start"
    CHECKPOINT_PERSISTENT_SAVE_OTHER_END = "checkpoint_persistent_save_other_end"

    CHECKPOINT_SAVED_TMP = "checkpoint_saved_tmp"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_ERROR = "checkpoint_error"

    # Logging events for mainly logging metric
    LOG_METRIC = "log_metric"
    GENERIC = "generic"


class LogType(StrEnum):
    EVENT = "event"
    TEXT = "text"


class ExtraFields(StrEnum):
    LOG_TYPE = "log_type"
    LOG_TYPE_NAME = "log_type_name"
    EVENT_NAME = "event_name"
    STEP = "step"
    CONTEXT = "context"
    VALUE = "value"


CHECKPOINTING_VERSION_KEY = "checkpointing_version"


def dict_to_list_safe(d: dict[str, str] | None) -> list[str] | None:
    if d is None:
        return None

    try:
        return [f"{k}:{v}" for k, v in d.items()]
    except Exception:
        return None


def event_extra(
    event_type: StrEnum,
    event_name: str | None = None,
    step: int | None = None,
    value: float | int | None = None,
    context: dict[str, str] | None = None,
    **extra: t.Any,
) -> dict[str, t.Any]:
    d: dict[str, t.Any] = {
        str(ExtraFields.LOG_TYPE): str(LogType.EVENT),
        str(ExtraFields.LOG_TYPE_NAME): str(event_type),
        str(ExtraFields.EVENT_NAME): event_name,
        str(ExtraFields.STEP): step,
        str(ExtraFields.VALUE): value,
        str(ExtraFields.CONTEXT): dict_to_list_safe(context),
    }
    for key, val in extra.items():
        if val is not None:
            d[key] = val
    return d


CHECKPOINT_SAVE_EVENTS_MAP: dict[tuple[bool, str], EventType] = {
    # (start, name):
    (True, "*"): EventType.CHECKPOINT_PERSISTENT_SAVE_ALL_START,
    (False, "*"): EventType.CHECKPOINT_PERSISTENT_SAVE_ALL_END,
    (True, "model"): EventType.CHECKPOINT_PERSISTENT_SAVE_MODEL_START,
    (True, "model_state"): EventType.CHECKPOINT_PERSISTENT_SAVE_MODEL_START,
    (False, "model"): EventType.CHECKPOINT_PERSISTENT_SAVE_MODEL_END,
    (False, "model_state"): EventType.CHECKPOINT_PERSISTENT_SAVE_MODEL_END,
    (True, "optimizer"): EventType.CHECKPOINT_PERSISTENT_SAVE_OPTIM_START,
    (True, "optimizer_state"): EventType.CHECKPOINT_PERSISTENT_SAVE_OPTIM_START,
    (False, "optimizer"): EventType.CHECKPOINT_PERSISTENT_SAVE_OPTIM_END,
    (False, "optimizer_state"): EventType.CHECKPOINT_PERSISTENT_SAVE_OPTIM_END,
    (
        True,
        "dataloader",
    ): EventType.CHECKPOINT_PERSISTENT_SAVE_DATA_LOADER_START,
    (
        True,
        "dataloader_state",
    ): EventType.CHECKPOINT_PERSISTENT_SAVE_DATA_LOADER_START,
    (
        False,
        "dataloader",
    ): EventType.CHECKPOINT_PERSISTENT_SAVE_DATA_LOADER_END,
    (
        False,
        "dataloader_state",
    ): EventType.CHECKPOINT_PERSISTENT_SAVE_DATA_LOADER_END,
}


def get_log_event_type_for_file_save(name: str, is_start: bool) -> EventType:
    key = (is_start, name)
    if key in CHECKPOINT_SAVE_EVENTS_MAP:
        return CHECKPOINT_SAVE_EVENTS_MAP[key]
    else:
        return (
            EventType.CHECKPOINT_PERSISTENT_SAVE_OTHER_START
            if is_start
            else EventType.CHECKPOINT_PERSISTENT_SAVE_OTHER_END
        )


_checkpoint_context_var: contextvars.ContextVar[dict[str, t.Any]] = (
    contextvars.ContextVar("checkpoint_logging_context")
)


class _CheckpointLoggingContext:
    """Per-context logging context for checkpoint operations.

    Stores arbitrary key-value pairs (e.g. step, job_id, or any user-defined
    field) that are automatically attached to checkpoint log events.

    Backed by ``contextvars.ContextVar`` so that each thread or async task
    that copies the context (via ``contextvars.copy_context().run()``) gets
    an isolated snapshot.  The trainer sets values before spawning the comm
    thread; the comm thread inherits those values and is unaffected by later
    updates from the trainer.

    For cross-process communication (subprocess), use ``export_context`` /
    ``import_context`` to serialize the current values through the pipe.
    """

    def update(self, **kwargs: t.Any) -> None:
        """Merge key-value pairs into the current context."""
        ctx = dict(_checkpoint_context_var.get({}))
        ctx.update(kwargs)
        _checkpoint_context_var.set(ctx)

    def get(self, key: str, default: t.Any = None) -> t.Any:
        """Retrieve a single value from the current context."""
        return _checkpoint_context_var.get({}).get(key, default)

    def export_context(self) -> dict[str, t.Any]:
        """Snapshot the current context for sending across process boundaries."""
        return dict(_checkpoint_context_var.get({}))

    def import_context(self, ctx: dict[str, t.Any]) -> None:
        """Restore context received from another process."""
        _checkpoint_context_var.set(dict(ctx))


checkpoint_logging_context = _CheckpointLoggingContext()


# Interface for unified event logging
class EventLogger:
    def __init__(self) -> None:
        self.creation_time_ms: int = int(time.time() * 1000)
        self.last_event_time_ms = self.creation_time_ms

    def __call__(
        self,
        event_type: StrEnum,
        event_name: str | None = None,
        end_to_end: bool | None = False,
        metric_name: str | None = None,
        value: int | float | None = None,
        **kwargs,
    ) -> dict[str, t.Any]:
        current_time_ms = int(time.time() * 1000)
        time_since_creation_ms = current_time_ms - self.creation_time_ms
        time_since_last_event_ms = current_time_ms - self.last_event_time_ms
        self.last_event_time_ms = current_time_ms
        if value is None:
            value = time_since_creation_ms if end_to_end else time_since_last_event_ms
        if not event_name and metric_name:
            # If no event name is provided, use the metric name instead
            event_name = metric_name

        context = None
        if str(ExtraFields.CONTEXT) in kwargs:
            context = kwargs.pop(str(ExtraFields.CONTEXT))

        if version.__version__ != version.UNKNOWN:
            context = context or []
            context.append(f"{CHECKPOINTING_VERSION_KEY}:{version.__version__}")

        # Step from kwargs > checkpoint_logging_context > None
        step = kwargs.pop("step", checkpoint_logging_context.get("step"))

        metadata = {
            "metric_name": metric_name,
            str(ExtraFields.LOG_TYPE_NAME): str(event_type),
            str(ExtraFields.LOG_TYPE): str(LogType.EVENT),
            str(ExtraFields.EVENT_NAME): event_name,
            str(ExtraFields.STEP): step,
            "end_to_end": end_to_end,
            "total_time_ms": time_since_creation_ms,
            "last_interval_ms": time_since_last_event_ms,
            "value": value,
            **kwargs,
        }

        if context:
            metadata[str(ExtraFields.CONTEXT)] = context

        return metadata


@runtime_checkable
class LatencyTracker(Protocol):
    def track(self, event_name: str) -> None: ...
