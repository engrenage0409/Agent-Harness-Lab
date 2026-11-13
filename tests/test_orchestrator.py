from harness.orchestrator import AgentSpec,SequentialOrchestrator
from harness.provider import ScriptedProvider
from harness.contracts import ToolRegistry

def test_sequential():
    reg=ToolRegistry()
    agents=[AgentSpec("research","Find evidence",ScriptedProvider([{"content":"Source evidence"}]),reg),
            AgentSpec("write","Summarize",ScriptedProvider([{"content":"Final result"}]),reg)]
    output=SequentialOrchestrator(agents).run("Explain question")
    assert output["status"]=="completed"
    assert [x["agent"] for x in output["history"]]==["research","write"]
