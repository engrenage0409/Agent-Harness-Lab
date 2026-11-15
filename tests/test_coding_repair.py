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
    assert {definition["function"]["name"] for definition in registry.definitions()} == {"run_tests"}
    assert registry.execute("read_file", {"path": "../secret.py"})["error_category"] == "workflow_violation"
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
    registry.definitions()
    registry.execute("run_tests", {})
    registry.definitions()
    registry.execute("read_file", {"path": "calculator.py"})
    registry.definitions()
    registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "1", "new_text": "2"})
    registry.definitions()
    registry.execute("run_tests", {})
    assert registry.completion_feedback([]) is None
    # A new editing phase requires a new failing test; edits after success are blocked.
    assert registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "2", "new_text": "3"
    })["error_category"] == "workflow_violation"
    assert registry.completion_feedback([]) is None


def test_cannot_edit_before_read_or_repeat_failed_patch(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()
    (folder / "calculator.py").write_text(
        "def add(a, b): return a - b", encoding="utf-8"
    )
    registry = RepairTools(folder, lambda: {
        "exit_code": 1, "timed_out": False, "stdout": "test_add failed", "stderr": ""
    })
    assert [d["function"]["name"] for d in registry.definitions()] == ["run_tests"]
    assert registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "-1 == 5", "new_text": "5"
    })["error_category"] == "workflow_violation"

    registry.execute("run_tests", {})
    assert [d["function"]["name"] for d in registry.definitions()] == ["read_file"]
    assert registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "-1 == 5", "new_text": "5"
    })["error_category"] == "workflow_violation"

    assert "return a - b" in registry.execute(
        "read_file", {"path": "calculator.py"}
    )
    assert [d["function"]["name"] for d in registry.definitions()] == ["edit_file"]
    result = registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "-1 == 5", "new_text": "5"
    })
    assert "Expected exactly one match" in result["error"]
    assert registry.successful_edits == 0
    assert [d["function"]["name"] for d in registry.definitions()] == ["read_file"]
    assert registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "-1 == 5", "new_text": "5"
    })["error_category"] == "workflow_violation"
    assert "return a - b" in registry.execute(
        "read_file", {"path": "calculator.py"}
    )
    registry.definitions()
    assert registry.execute("edit_file", {
        "path": "calculator.py", "old_text": "return a - b",
        "new_text": "return a + b"
    })["replacements"] == 1
    assert [d["function"]["name"] for d in registry.definitions()] == ["run_tests"]


def test_same_turn_tool_chaining_cannot_skip_read(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()
    (folder / "calculator.py").write_text(
        "def add(a, b): return a - b", encoding="utf-8"
    )
    registry = RepairTools(folder, lambda: {
        "exit_code": 1, "timed_out": False, "stdout": "", "stderr": ""
    })
    registry.definitions()
    registry.execute("run_tests", {})
    # The prior turn exposed only run_tests; read_file cannot execute
    # until the subsequent provider request sees new definitions.
    assert registry.execute("read_file", {
        "path": "calculator.py"
    })["error_category"] == "workflow_violation"
    assert registry.definitions()[0]["function"]["name"] == "read_file"


def test_http_illegal_tool_feedback_then_repair(tmp_path):
    import pytest
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from serving.app import create_app

    target = tmp_path / 'calculator.py'
    target.write_text('def add(a, b): return a - b')
    executions = []
    def tests():
        executions.append(1)
        return {'exit_code': 0 if 'a + b' in target.read_text() else 1}
    registry = RepairTools(tmp_path, tests)
    outputs = iter([
        ('edit_file', {'path': 'calculator.py', 'old_text': 'a - b', 'new_text': 'BAD'}),
        ('unknown_tool', {}),
        ('run_tests', {}),
        ('read_file', {'path': 'calculator.py'}),
        ('edit_file', {'path': 'calculator.py', 'old_text': 'a - b', 'new_text': 'a + b'}),
        ('run_tests', {}),
        None,
    ])
    class Backend:
        def load(self): pass
        def generate(self, messages, *args):
            item = next(outputs)
            if item is None:
                return 'Fixed', 1, 1
            name, arguments = item
            return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>', 1, 1
    with TestClient(create_app(Backend(), model_id='test')) as client:
        class Provider:
            def complete(self, messages, tools):
                response = client.post('/v1/chat/completions', json={
                    'model': 'test', 'messages': messages, 'tools': tools,
                })
                assert response.status_code == 200, response.text
                return response.json()['choices'][0]['message']
        result = AgentRuntime(Provider(), registry, max_steps=8,
                              completion_check=registry.completion_feedback).run('Repair')
    assert result.status == 'completed'
    observations = [m for m in result.messages if m['role'] == 'tool']
    for observation in observations[:2]:
        error = json.loads(observation['content'])
        assert error['error_category'] == 'workflow_violation'
        assert error['allowed_tools'] == ['run_tests']
        assert error['phase'] == 'test'
        assert observation['tool_call_id']
    assert executions == [1, 1]
    assert registry.successful_edits == 1
    assert 'a + b' in target.read_text()


def test_phase_change_revokes_same_turn_permission(tmp_path):
    runs = []
    registry = RepairTools(tmp_path, lambda: runs.append(1) or {'exit_code': 1})
    registry.definitions()
    registry.execute('run_tests', {})
    assert registry.execute('run_tests', {})['error_category'] == 'workflow_violation'
    assert runs == [1]


def test_stage_guidance_does_not_supply_the_patch(tmp_path):
    from scripts.coding_repair_demo import PhaseGuidedProvider
    registry = RepairTools(tmp_path, lambda: {'exit_code': 1})
    seen = []
    class Provider:
        def complete(self, messages, tools):
            seen.append(messages)
            return {'content': 'ok'}
    provider = PhaseGuidedProvider(Provider(), registry)
    messages = [{'role': 'user', 'content': 'fix'}]
    registry.phase = 'read'
    provider.complete(messages, registry.definitions())
    assert len(messages) == 1
    assert '{"path":"calculator.py"}' in seen[0][-1]['content']
    registry.phase = 'edit'
    assert 'old_text' in registry.completion_feedback([])
    assert 'a + b' not in registry.next_action()


def test_failed_multiline_edit_diagnostics_and_recovery(tmp_path):
    source = 'def first():\n    return 1\ndef second():\n    return 1\n'
    target = tmp_path / 'calculator.py'
    target.write_bytes(source.encode())
    registry = RepairTools(tmp_path, lambda: {'exit_code': 1})
    registry.definitions()
    registry.execute('run_tests', {})
    for bad, diagnostic in [
        (source.replace('\n', r'\n'), 'literal backslash-n'),
        (source.replace('    ', '  '), 'indentation'),
        (source.replace('\n', ' '), 'line breaks'),
        ('return 1', 'ambiguous'),
    ]:
        registry.definitions()
        assert registry.execute('read_file', {'path': 'calculator.py'}) == source
        registry.definitions()
        result = registry.execute('edit_file', {
            'path': 'calculator.py', 'old_text': bad, 'new_text': 'replacement'})
        assert diagnostic in result['error']
        assert 'No file was changed' in result['error']
        assert 'read_file' in result['next_action']
        assert target.read_bytes() == source.encode()
        assert registry.successful_edits == 0
        registry.definitions()
        assert registry.execute('edit_file', {
            'path': 'calculator.py', 'old_text': bad, 'new_text': 'replacement'
        })['error_category'] == 'workflow_violation'
    assert registry.execute('read_file', {'path': 'calculator.py'}) == source
    registry.definitions()
    assert registry.execute('edit_file', {
        'path': 'calculator.py', 'old_text': 'def first():\n    return 1',
        'new_text': 'def first():\n    return 2',
    })['replacements'] == 1
    assert target.read_text() == source.replace('return 1', 'return 2', 1)
    assert registry.definitions()[0]['function']['name'] == 'run_tests'


def test_shared_edit_guidance(tmp_path):
    from coding.sandbox import EDIT_DESCRIPTION, register_coding_tools
    from harness.contracts import ToolRegistry
    general = ToolRegistry()
    register_coding_tools(general, tmp_path)
    repair = RepairTools(tmp_path, lambda: {'exit_code': 1})
    repair.phase = 'edit'
    for registry in (general, repair):
        edit = next(d['function'] for d in registry.definitions()
                    if d['function']['name'] == 'edit_file')
        assert edit['description'] == EDIT_DESCRIPTION
    assert 'Do not replace the whole file' in repair.next_action()
    assert 'smallest unique exact source snippet' in repair.next_action()
