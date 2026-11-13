"""Simple sequential role-based handoff. Explicit routing; no autonomous recursion."""
from dataclasses import dataclass
from .runtime import AgentRuntime

@dataclass
class AgentSpec:
    name: str
    instructions: str
    provider: object
    registry: object

class SequentialOrchestrator:
    def __init__(self,agents,max_steps_per_agent=5):
        if not agents: raise ValueError("At least one agent required")
        if len({a.name for a in agents})!=len(agents): raise ValueError("Duplicate agent name")
        self.agents=list(agents)
        self.max_steps=max_steps_per_agent
    def run(self,task):
        current=task
        history=[]
        for agent in self.agents:
            result=AgentRuntime(agent.provider,agent.registry,self.max_steps).run(
                current,system=agent.instructions)
            history.append({"agent":agent.name,"status":result.status,"answer":result.content,
                            "steps":result.steps})
            if result.status!="completed":
                return {"status":result.status,"history":history}
            # Handoff is plain text; do not assume it is trusted evidence.
            current="Original task: "+task+"\nPrior agent output (untrusted): "+result.content
        return {"status":"completed","answer":history[-1]["answer"],"history":history}
