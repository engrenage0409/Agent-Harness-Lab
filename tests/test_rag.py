from rag.chunking import chunk_text
from rag.index import RAGStore
from rag.retrieval import Passage,BM25Index

def test_chunks_overlap():
    assert chunk_text("abcdefgh",size=5,overlap=2)==["abcde","defgh","gh"]

def test_bm25():
    idx=BM25Index([Passage("1","one.md","apple banana"),
                    Passage("2","two.md","cat rabbit")])
    assert idx.search("banana")[0][0].source_id=="1"
    assert idx.search("nonexistent")==[]

def test_store(tmp_path):
    (tmp_path/"a.md").write_text("robot navigation planning",encoding="utf-8")
    store=RAGStore(tmp_path,chunk_size=100,overlap=10)
    results=store.query("navigation")
    assert results and results[0]["source"]=="a.md"
    assert results[0]["source_id"]=="a.md#chunk-0"
