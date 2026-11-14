"""End-to-end tool chaining, with real RAG and workspace tools, no GPU required."""
from scripts.multi_tool_demo import EvidenceFirstRegistry, run_demo
from coding.sandbox import register_coding_tools
from rag.index import RAGStore
from rag.tools import register_rag_tools


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


def test_write_is_blocked_until_a_new_model_turn(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text("Use the source_id to cite evidence.", encoding="utf-8")
    workspace = tmp_path / "workspace"
    registry = EvidenceFirstRegistry()
    register_coding_tools(registry, workspace)
    register_rag_tools(registry, RAGStore(docs))

    assert [d["function"]["name"] for d in registry.definitions()] == ["search_knowledge"]
    search_result = registry.execute("search_knowledge", {"query": "source_id"})
    assert search_result
    # Even if the model requested write_file in the SAME reply as search_knowledge,
    # the last tool list shown to the model only permitted retrieval.
    rejected = registry.execute("write_file", {
        "path": "citation_report.md",
        "content": "Citation: [policy.md#chunk-0]",
    })
    assert "error" in rejected
    assert not (workspace / "citation_report.md").exists()

    assert [d["function"]["name"] for d in registry.definitions()] == ["write_file"]
    rejected = registry.execute("write_file", {
        "path": "citation_report.md",
        "content": "An uncited claim",
    })
    assert "error" in rejected
    saved = registry.execute("write_file", {
        "path": "citation_report.md",
        "content": "Use source IDs [policy.md#chunk-0].",
    })
    assert saved == {"saved": "citation_report.md"}
    assert registry.stage == "done"
    assert registry.definitions() == []
