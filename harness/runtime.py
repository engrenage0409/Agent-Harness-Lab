"""Bounded tool loop: model -> tool calls -> observations -> model."""
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
    def __init__(self,provider,registry,max_steps=8,trace_path=None):
        if max_steps<1: raise ValueError("max_steps must be positive")
        self.provider=provider
        self.registry=registry
        self.max_steps=max_steps
        self.trace=TraceWriter(trace_path)
    def run(self,task,system="Use tools when useful. Cite retrieved sources using [source_id]."):
        messages=[{"role":"system","content":system},{"role":"user","content":task}]
        self.trace.emit("run_started",task=task,max_steps=self.max_steps)
        for step in range(1,self.max_steps+1):
            reply=self.provider.complete(messages,self.registry.definitions())
            tool_calls=reply.get("tool_calls") or []
            content=reply.get("content") or ""
            clean={"role":"assistant","content":content}
            if tool_calls: clean["tool_calls"]=tool_calls
            messages.append(clean)
            self.trace.emit("assistant",step=step,content=content,tool_calls=tool_calls)
            if not tool_calls:
                self.trace.emit("run_finished",status="completed",steps=step)
                return RunResult(content,step,messages,"completed")
            for call in tool_calls:
                function=call.get("function") or {}
                name=function.get("name","")
                try: args=json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError: args=None
                observation=self.registry.execute(name,args)
                messages.append({"role":"tool","tool_call_id":call.get("id","unknown"),
                                 "content":json.dumps(observation,ensure_ascii=False,default=str)})
                self.trace.emit("tool_result",step=step,name=name,arguments=args,result=observation)
        self.trace.emit("run_finished",status="max_steps_exceeded",steps=self.max_steps)
        return RunResult("",self.max_steps,messages,"max_steps_exceeded")
