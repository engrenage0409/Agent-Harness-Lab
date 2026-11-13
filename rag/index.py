"""Build an in-memory retriever and optional JSON chunk export."""
import json
from pathlib import Path
from .chunking import chunk_text,load_documents
from .retrieval import Passage,HybridRetriever

def build_passages(document_dir,chunk_size=500,overlap=80):
    passages=[]
    for doc in load_documents(document_dir):
        for i,text in enumerate(chunk_text(doc["text"],chunk_size,overlap)):
            passages.append(Passage(source_id=f"{doc['source']}#chunk-{i}",
                                    source=doc["source"],text=text))
    return passages

class RAGStore:
    def __init__(self,document_dir,dense=False,chunk_size=500,overlap=80):
        self.passages=build_passages(document_dir,chunk_size,overlap)
        self.retriever=HybridRetriever(self.passages,dense=dense)
    def query(self,question,k=4):
        hits=self.retriever.search(question,k)
        return [{"source_id":p.source_id,"source":p.source,
                 "score":score,"text":p.text} for p,score in hits]
    def save_chunks(self,out):
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps([p.__dict__ for p in self.passages],
                        ensure_ascii=False,indent=2),encoding="utf-8")
