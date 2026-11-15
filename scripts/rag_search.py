import argparse,json
from rag.index import RAGStore

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--docs",default="examples/knowledge")
    parser.add_argument("--dense",action="store_true")
    parser.add_argument("--backend",choices=["local","milvus"],default="local",
                        help="Dense backend: local numpy (default) or Milvus Lite")
    parser.add_argument("--milvus-uri",default="runs/rag_milvus.db")
    parser.add_argument("--model-name",default=None,
                        help="SentenceTransformer model (multilingual default for Milvus)")
    parser.add_argument("--rerank",action="store_true",help="CrossEncoder reranking (requires dense extra)")
    parser.add_argument("-k",type=int,default=4)
    args=parser.parse_args()
    store=RAGStore(args.docs,dense=args.dense,rerank=args.rerank,\n                   backend=args.backend,milvus_uri=args.milvus_uri,model_name=args.model_name)
    try:\n        print(json.dumps(store.query(args.query,args.k),ensure_ascii=False,indent=2))\n    finally:\n        store.close()

if __name__=="__main__":main()
