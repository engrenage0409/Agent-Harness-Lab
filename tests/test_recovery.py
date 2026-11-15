"""Regression tests for the small, bounded recovery features."""
import pytest

from coding.recovery import FileCheckpoint, ProgressTracker
from coding.repair import RepairTools
from harness.provider import ScriptedProvider
from harness.runtime import AgentRuntime


def test_progress_fingerprint_requires_same_source_patch_and_error():
    tracker = ProgressTracker(repeat_limit=3)
    patch = {"path": "calculator.py", "old_text": "missing", "new_text": "found"}
    assert tracker.record_failure("v0", patch, "zero matches") == 1
    assert tracker.record_failure("v0", patch, "zero matches") == 2
    assert tracker.stalled is False
    assert tracker.record_failure("v1", patch, "zero matches") == 1
    assert tracker.record_failure("v0", patch, "zero matches") == 3
    assert tracker.stalled is True
    tracker.reset()
    assert tracker.stalled is False


def test_single_file_diff_and_opt_in_rollback(tmp_path):
    file = tmp_path / "calculator.py"
    file.write_bytes(b"def add(a, b): return a - b\r\n")
    checkpoint = FileCheckpoint(file)
    assert checkpoint.rollback() is False
    file.write_bytes(b"def add(a, b): return a + b\r\n")
    checkpoint.record_edit()
    assert "-def add(a, b): return a - b" in checkpoint.diff()
    assert "+def add(a, b): return a + b" in checkpoint.diff()
    assert checkpoint.rollback() is True
    assert file.read_bytes() == b"def add(a, b): return a - b\r\n"


def test_rollback_refuses_external_change(tmp_path):
    file = tmp_path / "calculator.py"
    file.write_text("value = 1", encoding="utf-8")
    checkpoint = FileCheckpoint(file)
    file.write_text("value = 2", encoding="utf-8")
    checkpoint.record_edit()
    file.write_text("user edit", encoding="utf-8")
    with pytest.raises(ValueError, match="outside this agent"):
        checkpoint.rollback()
    assert file.read_text(encoding="utf-8") == "user edit"


def test_repeated_failed_edits_stop_before_max_steps(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text(
        "def add(a, b): return a - b", encoding="utf-8"
    )
    registry = RepairTools(workspace, lambda: {
        "exit_code": 1, "stdout": "test_add failed", "stderr": "",
        "timed_out": False,
    })

    import json

    def call(n, name, args):
        return {"id": n, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args),
        }}

    bad = {"path": "calculator.py", "old_text": "no such text", "new_text": "replacement"}
    responses = [{"role": "assistant", "tool_calls": [call("t", "run_tests", {})]}]
    for i in range(3):
        responses.append({"role": "assistant", "tool_calls": [
            call(f"r{i}", "read_file", {"path": "calculator.py"})]})
        responses.append({"role": "assistant", "tool_calls": [
            call(f"e{i}", "edit_file", bad)]})

    result = AgentRuntime(
        ScriptedProvider(responses), registry, max_steps=20,
        stop_check=registry.stop_reason,
        completion_check=registry.completion_feedback,
    ).run("Repair the file")
    assert result.status == "stalled"
    assert result.steps == 7
    assert registry.successful_edits == 0
    assert registry.phase == "stalled"
    assert registry.checkpoint.diff() == ""
