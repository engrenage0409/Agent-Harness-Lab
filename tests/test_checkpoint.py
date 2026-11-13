import json
from harness.checkpoint import CheckpointStore
from harness.runtime import AgentRuntime
from harness.contracts import Tool,ToolRegistry
from harness.provider import ScriptedProvider

def test_resume_after_checkpoint(tmp_path):
    store=CheckpointStore(tmp_path/"state")
    calls={"count":0}
    def fn():
        calls["count"]+=1
        return {"ok":True}
    reg=ToolRegistry().add(Tool("ping","Ping",{"type":"object","properties":{}},fn))
    first=ScriptedProvider([
        {"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"ping","arguments":"{}"}}]},
        {"role":"assistant","content":"done"}])
    runtime=AgentRuntime(first,reg,checkpoints=store)
    finished=runtime.run("task",session_id="s1")
    assert finished.status=="completed" and calls["count"]==1
    resumed=AgentRuntime(ScriptedProvider([]),reg,checkpoints=store).run("task",session_id="s1",resume=True)
    assert resumed.status=="completed" and calls["count"]==1
    assert len(resumed.messages)==len(finished.messages)

def test_ambiguous_inflight_requires_review(tmp_path):
    store=CheckpointStore(tmp_path)
    store.save("s1",{"task":"t","messages":[{"role":"user","content":"t"}],
       "step":1,"pending":[{"id":"1","function":{"name":"ping","arguments":"{}"},"in_flight":True}],
       "status":"running","result":""})
    r=AgentRuntime(ScriptedProvider([]),ToolRegistry(),checkpoints=store).run("t",session_id="s1",resume=True)
    assert r.status=="needs_review"
