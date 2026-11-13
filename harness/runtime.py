"""Bounded tool loop with per-tool-call checkpointing and explicit resume."""
import json
from dataclasses import dataclass
from .tracing import TraceWriter

@dataclass
class RunResult:
    content: str
    steps: int
    messages: list
    status: str

class AgentRuntime:
    def __init__(self,provider,registry,max_steps=8,trace_path=None,checkpoints=None):
        if max_steps<1: raise ValueError("max_steps must be positive")
        self.provider=provider
        self.registry=registry
        self.max_steps=max_steps
        self.trace=TraceWriter(trace_path)
        self.checkpoints=checkpoints
    def run(self,task,system="Use tools when useful. Cite retrieved sources using [source_id].",
            session_id=None,resume=False):
        if resume and (not session_id or self.checkpoints is None):
            raise ValueError("resume requires session_id and checkpoint store")
        if session_id and self.checkpoints is None:
            raise ValueError("session_id requires checkpoint store")
        state=self.checkpoints.load(session_id) if resume else None
        if resume and state is None: raise FileNotFoundError("No checkpoint to resume")
        if state:
            if state["task"]!=task: raise ValueError("Task does not match checkpoint")
            if state["status"]=="completed":
                return RunResult(state["result"],state["step"],state["messages"],"completed")
            messages=state["messages"]
            step=state["step"]
            pending=state.get("pending",[])
        else:
            messages=[{"role":"system","content":system},{"role":"user","content":task}]
            step=0;pending=[]
        def persist(status,result=""):
            if session_id:
                self.checkpoints.save(session_id,{"task":task,"messages":messages,
                    "step":step,"pending":pending,"status":status,"result":result})
        persist("running")
        self.trace.emit("run_started",task=task,resume=resume,max_steps=self.max_steps)
        while True:
            if pending:
                # A crash DURING a side-effecting tool is ambiguous: operator approval required.
                if pending[0].get("in_flight"):
                    persist("needs_review")
                    return RunResult("",step,messages,"needs_review")
                call=pending[0]
                function=call.get("function") or {}
                name=function.get("name","")
                try: args=json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError,TypeError): args=None
                call["in_flight"]=True
                persist("running")
                observation=self.registry.execute(name,args)
                messages.append({"role":"tool","tool_call_id":call.get("id","unknown"),
                                 "content":json.dumps(observation,ensure_ascii=False,default=str)})
                pending.pop(0)
                persist("running")
                self.trace.emit("tool_result",step=step,name=name,arguments=args,result=observation)
                continue
            if step>=self.max_steps:
                persist("max_steps_exceeded")
                self.trace.emit("run_finished",status="max_steps_exceeded",steps=step)
                return RunResult("",step,messages,"max_steps_exceeded")
            step+=1
            reply=self.provider.complete(messages,self.registry.definitions())
            tool_calls=reply.get("tool_calls") or []
            content=reply.get("content") or ""
            clean={"role":"assistant","content":content}
            if tool_calls: clean["tool_calls"]=tool_calls
            messages.append(clean)
            pending=[dict(call) for call in tool_calls]
            persist("running")
            self.trace.emit("assistant",step=step,content=content,tool_calls=tool_calls)
            if not pending:
                persist("completed",content)
                self.trace.emit("run_finished",status="completed",steps=step)
                return RunResult(content,step,messages,"completed")
