import json
from harness.contracts import Tool, ToolRegistry
from harness.provider import ScriptedProvider
from harness.runtime import AgentRuntime

def test_agent_loop(tmp_path):
    reg=ToolRegistry()
    from rag.index import RAGStore
    from rag.tools import register_rag_tools
    (tmp_path/"doc.md").write_text("retrieval citation",encoding="utf-8")
    register_rag_tools(reg,RAGStore(tmp_path))
    scripted=ScriptedProvider([
      {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function",
       "function":{"name":"search_knowledge","arguments":json.dumps({"query":"retrieval"})}}]},
      {"role":"assistant","content":"Found [doc.md#chunk-0]"}
    ])
    result=AgentRuntime(scripted,reg,trace_path=tmp_path/"run.jsonl").run("Find retrieval")
    assert result.status=="completed" and result.steps==2
    assert result.messages[-2]["role"]=="tool"
    assert (tmp_path/"run.jsonl").exists()

def test_bounded():
    scripted=ScriptedProvider([{"role":"assistant","content":"done"}])
    out=AgentRuntime(scripted,ToolRegistry(),max_steps=1).run("hi")
    assert out.status=="completed"

def test_tool_registry_validates_schema_before_execution():
    called = []
    registry = ToolRegistry()
    registry.add(Tool(
        "multiply",
        "Multiply two numbers.",
        {"type": "object",
         "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
         "required": ["a", "b"],
         "additionalProperties": False},
        lambda a, b: called.append((a, b)) or a * b,
    ))

    wrong_type = registry.execute("multiply", {"a": 6, "b": "hello"})
    assert "error" in wrong_type
    assert "number" in wrong_type["error"]
    assert called == []

    assert registry.execute("multiply", {"a": 6, "b": 7}) == 42
    assert called == [(6, 7)]


def test_tool_registry_enforces_numeric_bounds():
    registry = ToolRegistry()
    registry.add(Tool(
        "limited",
        "Accept a value from 1 to 10.",
        {"type": "object",
         "properties": {"value": {"type": "integer", "minimum": 1, "maximum": 10}},
         "required": ["value"],
         "additionalProperties": False},
        lambda value: value,
    ))
    assert "error" in registry.execute("limited", {"value": 11})
    assert "error" in registry.execute("limited", {"value": True})
    assert registry.execute("limited", {"value": 5}) == 5


def test_completion_check_reprompts_instead_of_false_success(tmp_path):
    registry = ToolRegistry()
    performed = []
    registry.add(Tool(
        "write_file",
        "Save a file",
        {"type": "object", "properties": {"path": {"type": "string"}},
         "required": ["path"], "additionalProperties": False},
        lambda path: performed.append(path) or {"saved": path},
    ))
    scripted = ScriptedProvider([
        {"role": "assistant", "content": "Done!"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "write1", "type": "function",
                         "function": {"name": "write_file",
                                      "arguments": json.dumps({"path": "report.md"})}}]},
        {"role": "assistant", "content": "Saved report.md"},
    ])
    result = AgentRuntime(
        scripted, registry, max_steps=5,
        trace_path=tmp_path / "run.jsonl",
        completion_check=lambda messages: (
            "You have not saved report.md. Call write_file."
            if not performed else None
        ),
    ).run("Save report.md")
    assert result.status == "completed"
    assert result.steps == 3
    assert performed == ["report.md"]
    assert any(m.get("role") == "user" and "not saved" in m.get("content", "")
               for m in result.messages)
    assert '"event": "completion_rejected"' in (
        tmp_path / "run.jsonl"
    ).read_text(encoding="utf-8")


def test_completion_check_respects_max_steps():
    scripted = ScriptedProvider([
        {"role": "assistant", "content": "Done."},
        {"role": "assistant", "content": "Still done."},
    ])
    result = AgentRuntime(
        scripted, ToolRegistry(), max_steps=2,
        completion_check=lambda messages: "Missing required output",
    ).run("Create output")
    assert result.status == "max_steps_exceeded"
    assert result.steps == 2


def test_retry_safe_read_only_tool_recovers_transient_failure(tmp_path):
    calls = []
    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("temporary")
        return {"ok": True}
    registry = ToolRegistry().add(Tool(
        "lookup", "Read-only lookup",
        {"type": "object", "properties": {}}, flaky,
        retry_safe=True, max_transient_retries=1,
    ))
    scripted = ScriptedProvider([
        {"role": "assistant", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "assistant", "content": "retrieved"},
    ])
    path = tmp_path / "recovery.jsonl"
    result = AgentRuntime(scripted, registry, trace_path=path).run("lookup")
    assert result.status == "completed"
    assert len(calls) == 2
    assert result.messages[-2]["content"] == '{"ok": true}'
    assert '"event": "tool_retry"' in path.read_text(encoding="utf-8")


def test_non_idempotent_tool_is_never_retried():
    attempts = []
    def unstable_write():
        attempts.append(1)
        raise TimeoutError("unknown whether commit succeeded")
    registry = ToolRegistry().add(Tool(
        "write_file", "Side-effecting operation",
        {"type": "object", "properties": {}}, unstable_write,
    ))
    failure = registry.execute("write_file", {})
    assert failure["error_category"] == "transient_error"
    assert failure["retry_safe"] is False
    assert failure["attempts"] == 1
    assert len(attempts) == 1


def test_invalid_arguments_are_returned_without_execution(tmp_path):
    calls = []
    registry = ToolRegistry().add(Tool(
        "read_file", "Read a file",
        {"type": "object", "properties": {"path": {"type": "string"}},
         "required": ["path"], "additionalProperties": False},
        lambda path: calls.append(path) or path,
    ))
    scripted = ScriptedProvider([
        {"role": "assistant", "tool_calls": [{
            "id": "bad", "type": "function",
            "function": {"name": "read_file", "arguments": '{"file_name":"a.txt"}'}}]},
        {"role": "assistant", "tool_calls": [{
            "id": "fixed", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'}}]},
        {"role": "assistant", "content": "done"},
    ])
    trace = tmp_path / "errors.jsonl"
    result = AgentRuntime(scripted, registry, trace_path=trace).run("Read a.txt")
    assert result.status == "completed"
    assert calls == ["a.txt"]
    assert '"error_category": "invalid_arguments"' in result.messages[3]["content"]
    assert '"event": "tool_error"' in trace.read_text(encoding="utf-8")
