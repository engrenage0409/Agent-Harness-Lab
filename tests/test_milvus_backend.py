"""Milvus backend wiring tests; no Milvus install or model downloads needed."""
from rag.index import RAGStore
from rag.retrieval import Passage, HybridRetriever


def test_milvus_backend_preserves_source_ids_and_hybrid_search(monkeypatch, tmp_path):
    from rag import milvus_backend

    class FakeMilvusDenseIndex:
        def __init__(self, passages, **kwargs):
            self.passages = list(passages)
            self.kwargs = kwargs
            self.closed = False

        def search(self, query, k=5):
            return [(p, 0.9) for p in reversed(self.passages[:k])]

        def close(self):
            self.closed = True

    monkeypatch.setattr(milvus_backend, "MilvusDenseIndex", FakeMilvusDenseIndex)
    (tmp_path / "policy.md").write_text(
        "The citation policy requires a real source identifier.", encoding="utf-8"
    )
    store = RAGStore(tmp_path, backend="milvus", milvus_uri="runs/unit-test.db")
    hits = store.query("citation policy", k=3)
    assert hits
    assert hits[0]["source_id"] == "policy.md#chunk-0"
    assert hits[0]["source"] == "policy.md"
    assert "citation policy" in hits[0]["text"]
    dense = store.retriever.dense
    assert dense.kwargs["uri"] == "runs/unit-test.db"
    store.close()
    assert dense.closed


def test_invalid_backend_rejected():
    try:
        HybridRetriever([Passage("one#chunk-0", "one", "hello")], backend="unknown")
    except ValueError as exc:
        assert "backend" in str(exc)
    else:
        raise AssertionError("invalid backend was accepted")
