"""Two-tool Agent demo: retrieve evidence, then save a cited Markdown report.

Offline mode is a deterministic Harness integration test, not model reasoning.
Use --live for real Qwen planning over the same registered tools.
"""
import argparse
import json
from pathlib import Path

from coding.sandbox import register_coding_tools
from harness.contracts import ToolRegistry
from harness.provider import OpenAICompatibleProvider, ScriptedProvider
from harness.runtime import AgentRuntime
from rag.index import RAGStore
from rag.tools import register_rag_tools

DEFAULT_OUTPUT = "citation_report.md"
DEFAULT_TASK = (
    "First call search_knowledge to find the citation policy in the knowledge base. "
    "Then call write_file to save a short Markdown report to citation_report.md. "
    "Include the retrieved citation rule and the exact source_id. "
    "Do not invent sources. Finally tell me which file was saved."
)


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def demo_provider():
    """Prewritten tool decisions; useful for verifying runtime wiring offline."""
    return ScriptedProvider([
        {"role": "assistant", "content": None, "tool_calls": [
            tool_call("search1", "search_knowledge", {"query": "citation rules"})
        ]},
        {"role": "assistant", "content": None, "tool_calls": [
            tool_call("write1", "write_file", {
                "path": DEFAULT_OUTPUT,
                "content": (
                    "# Citation policy report\n\n"
                    "Cite retrieved passages with their source_id; do not invent citations.\n\n"
                    "Source: [policy.md#chunk-0]\n"
                ),
            })
        ]},
        {"role": "assistant", "content": "Saved citation_report.md with [policy.md#chunk-0]."},
    ])


def run_demo(*, live=False, docs="examples/knowledge", workspace="workspace",
             trace="runs/multi_tool_demo.jsonl", task=DEFAULT_TASK):
    registry = ToolRegistry()
    work = register_coding_tools(registry, workspace)
    register_rag_tools(registry, RAGStore(docs))
    provider = OpenAICompatibleProvider() if live else demo_provider()
    result = AgentRuntime(provider, registry, max_steps=6, trace_path=trace).run(task)

    observed = []
    for msg in result.messages:
        if msg.get("role") == "assistant":
            observed.extend(
                call.get("function", {}).get("name", "")
                for call in msg.get("tool_calls", [])
            )
    target = work.resolve(DEFAULT_OUTPUT)
    # A file from an earlier run does not count as a successful save.
    write_observed = False
    search_observed = False
    for idx, msg in enumerate(result.messages):
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls", []):
            name = call.get("function", {}).get("name")
            call_id = call.get("id")
            later = result.messages[idx + 1:]
            match = next((m for m in later if m.get("role") == "tool"
                          and m.get("tool_call_id") == call_id), None)
            if match is None:
                continue
            try:
                observation = json.loads(match.get("content", "null"))
            except (json.JSONDecodeError, TypeError):
                continue
            if name == "search_knowledge" and isinstance(observation, list) and any(
                hit.get("source_id") == "policy.md#chunk-0" for hit in observation
            ):
                search_observed = True
            if name == "write_file" and isinstance(observation, dict) and (
                observation.get("saved") == DEFAULT_OUTPUT
            ):
                write_observed = True

    exists = target.is_file() and write_observed
    report = target.read_text(encoding="utf-8") if exists else ""
    verified = (
        result.status == "completed"
        and search_observed
        and write_observed
        and observed.index("search_knowledge") < observed.index("write_file")
        and "[policy.md#chunk-0]" in report
    ) if "search_knowledge" in observed and "write_file" in observed else False
    return {
        "status": result.status,
        "steps": result.steps,
        "tool_calls": observed,
        "search_verified": search_observed,
        "write_verified": write_observed,
        "end_to_end_verified": verified,
        "output": str(target) if exists else None,
        "answer": result.content,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--docs", default="examples/knowledge")
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--trace", default="runs/multi_tool_demo.jsonl")
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()
    summary = run_demo(**vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["end_to_end_verified"]:
        raise SystemExit("End-to-end verification failed; inspect trace and output.")


if __name__ == "__main__":
    main()
