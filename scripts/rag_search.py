import argparse,json
from rag.index import RAGStore

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--docs",default="examples/knowledge")
    parser.add_argument("--dense",action="store_true")
    parser.add_argument("--rerank",action="store_true",help="CrossEncoder reranking (requires dense extra)")
    parser.add_argument("-k",type=int,default=4)
    args=parser.parse_args()
    store=RAGStore(args.docs,dense=args.dense,rerank=args.rerank)
    print(json.dumps(store.query(args.query,args.k),ensure_ascii=False,indent=2))

if __name__=="__main__":main()
