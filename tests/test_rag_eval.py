from rag.evaluation import score_query,evaluate_queries
from rag.index import RAGStore

def test_retrieval_metrics():
    m=score_query(["a","b","c"],["b","c"],k=3)
    assert m["recall"]==1.
    assert m["mrr"]==0.5
    assert 0<m["ndcg"]<1

def test_sample_labels():
    store=RAGStore("examples/knowledge")
    result=evaluate_queries(store,[{"query":"citations","relevant_source_ids":["policy.md#chunk-0"]}],k=2)
    assert result["n"]==1
    assert 0<=result["metrics"]["hit_rate"]<=1
