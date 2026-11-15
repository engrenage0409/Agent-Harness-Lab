"""Optional Milvus-backed dense retrieval; BM25 and citation IDs stay in the existing RAG pipeline.

The demo intentionally rebuilds one dedicated collection on initialization.
Do not point it at a collection containing data you need to preserve.
"""
from pathlib import Path


class MilvusDenseIndex:
    """Drop-in DenseIndex interface backed by a Milvus collection."""

    def __init__(
        self,
        passages,
        *,
        uri="runs/rag_milvus.db",
        collection="agent_harness_rag",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        from pymilvus import MilvusClient
        from sentence_transformers import SentenceTransformer

        self.passages = list(passages)
        self.model = SentenceTransformer(model_name)
        self.collection = collection
        self.client = None
        if not self.passages:
            return

        # For Milvus Lite, avoid accidentally treating a URI as a local path.
        if "://" not in uri:
            path = Path(uri)
            path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = self.model.encode(
            [p.text for p in self.passages], normalize_embeddings=True
        )
        dimensions = len(embeddings[0])
        self.client = MilvusClient(uri=uri)
        if self.client.has_collection(collection):
            self.client.drop_collection(collection)
        self.client.create_collection(
            collection_name=collection, dimension=dimensions, metric_type="COSINE"
        )
        self.client.insert(
            collection_name=collection,
            data=[
                {"id": i, "vector": vector.tolist()}
                for i, vector in enumerate(embeddings)
            ],
        )

    def search(self, query, k=5):
        if self.client is None or k < 1:
            return []
        vector = self.model.encode([query], normalize_embeddings=True)[0]
        hits = self.client.search(
            collection_name=self.collection,
            data=[vector.tolist()],
            limit=min(k, len(self.passages)),
        )[0]
        return [
            (self.passages[int(hit["id"])], float(hit["distance"]))
            for hit in hits
        ]

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
