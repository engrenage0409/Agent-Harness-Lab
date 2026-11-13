"""Offline demonstration of persistent memory and checkpointed tool-call loop."""
import json
from harness.memory import MemoryStore,register_memory_tools
from harness.checkpoint import CheckpointStore
from harness.contracts import ToolRegistry
from harness.provider import ScriptedProvider
from harness.runtime import AgentRuntime

def main():
    registry=ToolRegistry()
    store=MemoryStore("runs/demo_memory.sqlite3",namespace="demo")
    register_memory_tools(registry,store)
    calls=[
        {"role":"assistant","content":None,"tool_calls":[{"id":"m1","type":"function",
          "function":{"name":"memory_put","arguments":json.dumps({"key":"project","value":"Agent Harness Lab"})}}]},
        {"role":"assistant","content":None,"tool_calls":[{"id":"m2","type":"function",
          "function":{"name":"memory_get","arguments":json.dumps({"key":"project"})}}]},
        {"role":"assistant","content":"Stored and retrieved explicit project memory."}]
    runtime=AgentRuntime(ScriptedProvider(calls),registry,max_steps=5,
        checkpoints=CheckpointStore("runs/checkpoints"),trace_path="runs/memory_demo.jsonl")
    result=runtime.run("Save and retrieve project name",session_id="memory-demo")
    print(json.dumps({"status":result.status,"steps":result.steps,"value":store.get("project")},
                     ensure_ascii=False,indent=2))

if __name__=="__main__":main()
