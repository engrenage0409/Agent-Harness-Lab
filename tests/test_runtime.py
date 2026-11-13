import json
from harness.contracts import ToolRegistry
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
