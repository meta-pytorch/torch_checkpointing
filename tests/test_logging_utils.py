# Owner(s): ["oncall: distributed checkpointing"]

import contextvars
import threading
from unittest.mock import patch

from torch_checkpointing import version
from torch_checkpointing.logging_utils import (
    checkpoint_logging_context,
    CHECKPOINTING_VERSION_KEY,
    EventLogger,
    EventType,
    ExtraFields,
)


@patch("torch_checkpointing.logging_utils.version.__version__", "1.2.3")
def test_version_present():
    """Test EventLogger when __version__ is available (not UNKNOWN)."""
    logger = EventLogger()

    result = logger(EventType.CHECKPOINT_START)

    context = result.get(str(ExtraFields.CONTEXT))
    assert context == [f"{CHECKPOINTING_VERSION_KEY}:1.2.3"]


@patch("torch_checkpointing.logging_utils.version.__version__", version.UNKNOWN)
def test_version_unknown():
    """Test EventLogger when __version__ is UNKNOWN."""
    logger = EventLogger()

    result = logger(EventType.CHECKPOINT_START)

    context = result.get(str(ExtraFields.CONTEXT))
    assert context is None


@patch("torch_checkpointing.logging_utils.version.__version__", "2.0.1+abc123")
def test_version_with_existing_context():
    """Test EventLogger with version when context already exists."""
    logger = EventLogger()
    existing_context = ["existing:value"]

    result = logger(EventType.CHECKPOINT_START, context=existing_context)

    context = result.get(str(ExtraFields.CONTEXT))
    assert context is not None

    # Should contain both existing context and version info
    assert len(context) == 2
    assert "existing:value" in context
    version_entry = f"{CHECKPOINTING_VERSION_KEY}:2.0.1+abc123"
    assert version_entry in context


@patch("torch_checkpointing.logging_utils.version.__version__", version.UNKNOWN)
def test_version_unknown_with_existing_context():
    """Test EventLogger with UNKNOWN version when context already exists."""
    logger = EventLogger()
    existing_context = ["existing:value"]

    result = logger(EventType.CHECKPOINT_START, context=existing_context)

    context = result.get(str(ExtraFields.CONTEXT))
    assert context is not None

    # Should contain only existing context, no version info
    assert context == ["existing:value"]


@patch("torch_checkpointing.logging_utils.version.__version__", version.UNKNOWN)
def test_checkpoint_logging_context_isolated_across_threads():
    """The comm thread should see the step it was spawned with, not the step
    the trainer has since advanced to.

    This works because checkpoint_logging_context is backed by a ContextVar,
    and the comm thread is spawned via ``contextvars.copy_context().run()``
    which gives it an isolated snapshot of the step.
    """
    checkpoint_logging_context.import_context({})  # reset
    barrier = threading.Barrier(2)
    thread_steps: list[int | None] = []

    def fake_comm_thread():
        """Simulates the checkpoint comm thread (_write method).

        Creates an EventLogger and logs events, just like the real comm thread.
        The barrier forces this thread to still be running when the trainer
        updates the context for the next checkpoint.
        """
        event_logger = EventLogger()

        # Log an event before the trainer advances — should see step 100.
        result_before = event_logger(
            EventType.CHECKPOINT_THREAD_START,
            metric_name="train.checkpoint_write.execute.subprocess_comm.e2e.latency_ms",
        )
        thread_steps.append(result_before.get(str(ExtraFields.STEP)))

        # Wait for the trainer to update context to step 200.
        barrier.wait(timeout=5)

        # Log another event after the trainer advanced — should still see
        # step 100 because the thread has its own copy of the context.
        result_after = event_logger(
            EventType.CHECKPOINT_THREAD_END,
            metric_name="train.checkpoint_write.execute.subprocess_comm.e2e.latency_ms",
            end_to_end=True,
        )
        thread_steps.append(result_after.get(str(ExtraFields.STEP)))

    # --- Trainer: start checkpoint for step 100 ---
    checkpoint_logging_context.update(step=100)

    # Spawn comm thread with copy_context().run() — mirrors how
    # CheckpointProcess.write() submits to the ThreadPoolExecutor.
    ctx = contextvars.copy_context()
    thread = threading.Thread(target=ctx.run, args=(fake_comm_thread,))
    thread.start()

    # --- Trainer: start checkpoint for step 200 (before thread finishes) ---
    barrier.wait(timeout=5)
    checkpoint_logging_context.update(step=200)
    barrier.reset()

    thread.join(timeout=5)

    # Both events from the comm thread should see step 100.
    assert thread_steps[0] == 100, f"Expected step 100, got {thread_steps[0]}"
    assert thread_steps[1] == 100, (
        f"Race condition: comm thread for step 100 logged step {thread_steps[1]} "
        f"after trainer advanced to step 200"
    )

    # The trainer's context should independently reflect step 200.
    assert checkpoint_logging_context.get("step") == 200


@patch("torch_checkpointing.logging_utils.version.__version__", version.UNKNOWN)
def test_checkpoint_logging_context_export_import():
    """export_context snapshots the current context and import_context restores it.

    This is used for the subprocess path where contextvars don't cross process
    boundaries — the context is serialized into the pipe payload instead.
    """
    checkpoint_logging_context.import_context({})  # reset
    checkpoint_logging_context.update(step=42, job_id="abc123")
    exported = checkpoint_logging_context.export_context()
    assert exported == {"step": 42, "job_id": "abc123"}

    # Simulate a subprocess importing the context.
    checkpoint_logging_context.import_context({})
    checkpoint_logging_context.import_context(exported)
    assert checkpoint_logging_context.get("step") == 42
    assert checkpoint_logging_context.get("job_id") == "abc123"

    # EventLogger should pick up the imported step.
    event_logger = EventLogger()
    result = event_logger(EventType.CHECKPOINT_WRITE_START)
    assert result[str(ExtraFields.STEP)] == 42


@patch("torch_checkpointing.logging_utils.version.__version__", version.UNKNOWN)
def test_checkpoint_logging_context_explicit_step_overrides():
    """An explicit step= kwarg to EventLogger should override the context."""
    checkpoint_logging_context.import_context({})  # reset
    checkpoint_logging_context.update(step=100)
    event_logger = EventLogger()

    result = event_logger(EventType.LOG_METRIC, step=999)
    assert result[str(ExtraFields.STEP)] == 999


@patch("torch_checkpointing.logging_utils.version.__version__", version.UNKNOWN)
def test_checkpoint_logging_context_arbitrary_keys():
    """update() accepts arbitrary key-value pairs, get() retrieves them."""
    checkpoint_logging_context.import_context({})  # reset
    checkpoint_logging_context.update(step=10, job_id="j-42", region="us-west-1")

    assert checkpoint_logging_context.get("step") == 10
    assert checkpoint_logging_context.get("job_id") == "j-42"
    assert checkpoint_logging_context.get("region") == "us-west-1"
    assert checkpoint_logging_context.get("missing") is None
    assert checkpoint_logging_context.get("missing", "default") == "default"

    # update() merges — existing keys are preserved unless overwritten.
    checkpoint_logging_context.update(step=20)
    assert checkpoint_logging_context.get("step") == 20
    assert checkpoint_logging_context.get("job_id") == "j-42"
