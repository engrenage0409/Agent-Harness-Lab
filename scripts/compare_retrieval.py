"""Compare BM25, Milvus dense, and RRF using the same document chunks and query.

Usage:
    python -m scripts.compare_retrieval "successful repair verification" --docs docs -k 5
"""
import argparse
from math import isfinite
from rag.index import build_passages
from rag.retrieval import BM25Index
from rag.milvus_backend import MilvusDenseIndex


def compare(query, docs="docs", k=5, candidate_k=20, uri="runs/rag_milvus.db",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            bm25_weight=1.0, dense_weight=1.0):
    if any(not isfinite(w) or w < 0 for w in (bm25_weight,dense_weight)) or not (bm25_weight or dense_weight):
        raise ValueError("RRF weights must be finite, nonnegative and not both zero")
    passages = build_passages(docs)
    if not passages:
        raise ValueError(f"No Markdown/TXT chunks found under: {docs}")
    sparse = BM25Index(passages)
    dense = MilvusDenseIndex(passages, uri=uri, model_name=model_name)
    try:
        limit = max(k, candidate_k)
        bm25 = sparse.search(query, limit)
        semantic = dense.search(query, limit)

        # Exactly the same RRF formula and tie-break as HybridRetriever.
        rankings = {}
        for label, hits, weight in (("BM25", bm25, bm25_weight), ("Dense", semantic, dense_weight)):
            for rank, (passage, score) in enumerate(hits, 1):
                item = rankings.setdefault(passage.source_id, {
                    "passage": passage, "bm25_rank": None, "bm25_score": None,
                    "dense_rank": None, "cosine": None, "rrf": 0.0
                })
                item["rrf"] += weight / (60 + rank)
                if label == "BM25":
                    item["bm25_rank"] = rank
                    item["bm25_score"] = score
                else:
                    item["dense_rank"] = rank
                    item["cosine"] = score

        def print_hits(label, hits):
            print(f"\n{label}")
            for i, (passage, score) in enumerate(hits[:k], 1):
                print(f"  {i}. {passage.source_id}  score={score:.6f}")

        print(f"Query: {query}")
        print(f"Chunks indexed: {len(passages)}")
        print(f"RRF weights: BM25={bm25_weight:g}, Dense={dense_weight:g}")
        print_hits("BM25 ONLY", bm25)
        print_hits("MILVUS DENSE ONLY (cosine similarity)", semantic)

        fused = sorted(rankings.values(),
                       key=lambda x: (-x["rrf"], x["passage"].source_id))
        print("\nHYBRID RRF TOP-K (BM25 + Dense)")
        for i, item in enumerate(fused[:k], 1):
            print(f"  {i}. {item['passage'].source_id}")
            print(f"     BM25 rank={item['bm25_rank']}, score={item['bm25_score']}")
            print(f"     Dense rank={item['dense_rank']}, cosine={item['cosine']}")
            print(f"     RRF={item['rrf']:.6f}")
            print(f"     Text={item['passage'].text[:120]!r}")
    finally:
        dense.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--milvus-uri", default="runs/rag_milvus.db")
    parser.add_argument("--model-name",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()
    if args.k < 1 or args.candidate_k < 1:
        parser.error("-k and --candidate-k must be positive")
    compare(args.query, args.docs, args.k, args.candidate_k,
            args.milvus_uri, args.model_name, args.bm25_weight, args.dense_weight)


if __name__ == "__main__":
    main()
