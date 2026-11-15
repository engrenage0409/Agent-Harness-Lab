"""Offline tests: no Qwen weights, pytest subprocess, or GPU required."""
import json

from coding.repair import RepairTools
from harness.provider import ScriptedProvider
from harness.runtime import AgentRuntime


def call(id, name, args):
    return {"id": id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_repair_loop_fails_then_edits_then_passes(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()
    (folder / "calculator.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    observed = []

    def fake_tests():
        content = (folder / "calculator.py").read_text(encoding="utf-8")
        passed = "return a + b" in content
        observed.append(passed)
        return {"exit_code": 0 if passed else 1, "timed_out": False,
                "stdout": "1 passed" if passed else "assert -1 == 5", "stderr": ""}

    registry = RepairTools(folder, fake_tests)
    provider = ScriptedProvider([
        {"role": "assistant", "tool_calls": [call("t1", "run_tests", {})]},
        {"role": "assistant", "content": "Done"},  # must be rejected
        {"role": "assistant", "tool_calls": [
            call("r1", "read_file", {"path": "calculator.py"})]},
        {"role": "assistant", "tool_calls": [call("e1", "edit_file", {
            "path": "calculator.py",
            "old_text": "return a - b", "new_text": "return a + b"})]},
        {"role": "assistant", "content": "Fixed"},  # no fresh test => rejected
        {"role": "assistant", "tool_calls": [call("t2", "run_tests", {})]},
        {"role": "assistant", "content": "Tests passed."},
    ])
    result = AgentRuntime(
        provider, registry, max_steps=9,
        completion_check=registry.completion_feedback,
        trace_path=tmp_path / "trace.jsonl",
    ).run("Fix calculator.py")
    assert result.status == "completed"
    assert observed == [False, True]
    assert registry.successful_edits == 1
    assert registry.completion_feedback(result.messages) is None
    assert (tmp_path / "trace.jsonl").read_text().count("completion_rejected") == 2


def test_wrong_paths_and_unapproved_tools_cannot_execute(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()
    (folder / "calculator.py").write_text("value = 1", encoding="utf-8")
    runs = []
    registry = RepairTools(folder, lambda: runs.append(1) or {
        "exit_code": 0, "timed_out": False, "stdout": "", "stderr": ""
    })
    assert {definition["function"]["name"] for definition in registry.definitions()} == {
        "read_file", "edit_file", "run_tests"
    }
    assert "error" in registry.execute("read_file", {"path": "../secret.py"})
    assert "error" in registry.execute("edit_file", {
        "path": "../secret.py", "old_text": "a", "new_text": "b"
    })
    assert "error" in registry.execute("write_file", {
        "path": "calculator.py", "content": "wrong"
    })
    assert runs == []
    assert (folder / "calculator.py").read_text() == "value = 1"


def test_stale_pass_is_invalidated_by_another_edit(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()
    (folder / "calculator.py").write_text("value = 1", encoding="utf-8")
    results = iter([{"exit_code": 1, "timed_out": False},
                    {"exit_code": 0, "timed_out": False}])
    registry = RepairTools(folder, lambda: next(results))
    registry.execute("run_tests", {})
    registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "1", "new_text": "2"})
    registry.execute("run_tests", {})
    assert registry.completion_feedback([]) is None
    registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "2", "new_text": "3"})
    assert registry.completion_feedback([]) is not None
