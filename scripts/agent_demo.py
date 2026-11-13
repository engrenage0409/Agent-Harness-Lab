"""Offline scripted tool-call demo; no paid API required."""
import argparse,json
from harness.contracts import ToolRegistry
from harness.provider import ScriptedProvider,OpenAICompatibleProvider
from harness.runtime import AgentRuntime
from coding.sandbox import register_coding_tools
from rag.index import RAGStore
from rag.tools import register_rag_tools

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--live",action="store_true",help="Use AGENT_BASE_URL, AGENT_MODEL, AGENT_API_KEY")
    p.add_argument("--task",default="What does the project say about citations?")
    p.add_argument("--docs",default="examples/knowledge")
    p.add_argument("--workspace",default="workspace")
    p.add_argument("--trace",default="runs/demo.jsonl")
    args=p.parse_args()
    registry=ToolRegistry()
    register_coding_tools(registry,args.workspace)
    register_rag_tools(registry,RAGStore(args.docs))
    if args.live:
        provider=OpenAICompatibleProvider()
    else:
        provider=ScriptedProvider([
            {"role":"assistant","content":None,"tool_calls":[{
                "id":"call1","type":"function","function":{"name":"search_knowledge",
                "arguments":json.dumps({"query":"citations"})}}]},
            {"role":"assistant","content":"The sample knowledge base asks for passage citations, e.g. [policy.md#chunk-0]. This is a scripted example; review actual retrieved passages for factual answers."}
        ])
    result=AgentRuntime(provider,registry,max_steps=5,trace_path=args.trace).run(args.task)
    print(json.dumps({"status":result.status,"steps":result.steps,
                      "answer":result.content},ensure_ascii=False,indent=2))
    print("Trace:",args.trace)

if __name__=="__main__":main()
