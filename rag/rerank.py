"""Optional CrossEncoder reranking after sparse/dense/hybrid retrieval."""
class CrossEncoderReranker:
    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model=CrossEncoder(model_name)
    def rerank(self,query,candidates,k=5):
        if not candidates: return []
        scores=self.model.predict([(query,p.text) for p,_ in candidates])
        order=sorted(range(len(candidates)),key=lambda i:(-float(scores[i]),i))
        return [(candidates[i][0],float(scores[i])) for i in order[:max(0,k)]]
