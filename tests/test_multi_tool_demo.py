"""End-to-end tool chaining, with real RAG and workspace tools, no GPU required."""
from scripts.multi_tool_demo import run_demo


def test_search_then_write(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text(
        "Citation rules: cite source identifiers and never invent a source.",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    trace = tmp_path / "trace.jsonl"

    summary = run_demo(
        docs=str(docs),
        workspace=str(workspace),
        trace=str(trace),
    )

    assert summary["status"] == "completed"
    assert summary["steps"] == 3
    assert summary["tool_calls"] == ["search_knowledge", "write_file"]
    assert summary["search_verified"]
    assert summary["write_verified"]
    assert summary["end_to_end_verified"]
    assert (workspace / "citation_report.md").is_file()
    assert "[policy.md#chunk-0]" in (
        workspace / "citation_report.md"
    ).read_text(encoding="utf-8")
    logs = trace.read_text(encoding="utf-8")
    assert '"name": "search_knowledge"' in logs
    assert '"name": "write_file"' in logs
