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
