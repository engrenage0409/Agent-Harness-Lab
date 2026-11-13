import argparse,json
from pathlib import Path
from rag.index import RAGStore
from rag.evaluation import evaluate_queries

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--docs",default="examples/knowledge")
    p.add_argument("--labels",default="examples/rag_eval.jsonl")
    p.add_argument("--dense",action="store_true")
    p.add_argument("--rerank",action="store_true")
    p.add_argument("-k",type=int,default=5)
    p.add_argument("--out",default="runs/retrieval_eval.json")
    a=p.parse_args()
    with open(a.labels,encoding="utf-8") as f:
        labels=[json.loads(line) for line in f if line.strip()]
    result=evaluate_queries(RAGStore(a.docs,dense=a.dense,rerank=a.rerank),labels,a.k)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result["metrics"],indent=2))

if __name__=="__main__":main()
