"""BM25 sparse retrieval and optional dense/hybrid retrieval with RRF."""
import math
import re
from collections import Counter
from dataclasses import dataclass

def tokenize(text):
    # English word and individual CJK character tokens; baseline only, not full Chinese segmentation.
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]",text.lower())

@dataclass
class Passage:
    source_id: str
    source: str
    text: str

class BM25Index:
    def __init__(self,passages,k1=1.5,b=0.75):
        self.passages=list(passages)
        self.tf=[Counter(tokenize(p.text)) for p in self.passages]
        self.lengths=[sum(x.values()) for x in self.tf]
        self.avgdl=sum(self.lengths)/len(self.lengths) if self.lengths else 1
        self.k1,self.b=k1,b
        df=Counter()
        for counts in self.tf: df.update(counts.keys())
        n=len(self.tf)
        self.idf={term:math.log(1+(n-count+0.5)/(count+0.5)) for term,count in df.items()}
    def search(self,query,k=5):
        terms=set(tokenize(query))
        scores=[]
        for i,counts in enumerate(self.tf):
            score=0
            for term in terms:
                tf=counts.get(term,0)
                if not tf: continue
                norm=self.k1*(1-self.b+self.b*self.lengths[i]/self.avgdl)
                score+=self.idf[term]*tf*(self.k1+1)/(tf+norm)
            if score>0: scores.append((i,score))
        return [(self.passages[i],round(score,6)) for i,score in
                sorted(scores,key=lambda s:(-s[1],s[0]))[:max(0,k)]]

class DenseIndex:
    """Optional sentence-transformers embeddings; heavy dependency downloaded on first use."""
    def __init__(self,passages,model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.passages=list(passages)
        self.model=SentenceTransformer(model_name)
        self.embeddings=self.model.encode([p.text for p in self.passages],
                      normalize_embeddings=True) if self.passages else []
    def search(self,query,k=5):
        if not self.passages: return []
        q=self.model.encode([query],normalize_embeddings=True)[0]
        scores=self.embeddings@q
        order=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))[:k]
        return [(self.passages[i],float(scores[i])) for i in order]

def weighted_rrf(rankings, weights=(1.0, 1.0), constant=60):
    """Combine ranked Passage lists with weighted reciprocal-rank fusion."""
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError("RRF weights must be finite, nonnegative and not both zero")
    scores, passages = {}, {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, (passage, _) in enumerate(ranking, 1):
            scores[passage.source_id] = scores.get(passage.source_id, 0) + weight / (constant + rank)
            passages[passage.source_id] = passage
    return [(passages[sid], round(score, 6))
            for sid, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


class HybridRetriever:
    def __init__(self,passages,dense=False,backend="local",milvus_uri="runs/rag_milvus.db",model_name=None,bm25_weight=1.0,dense_weight=1.0):
        if backend not in ("local", "milvus"):
            raise ValueError("backend must be 'local' or 'milvus'")
        self.weights=(float(bm25_weight),float(dense_weight))
        if any(not math.isfinite(w) or w < 0 for w in self.weights) or not any(self.weights):
            raise ValueError("RRF weights must be finite, nonnegative and not both zero")
        self.sparse=BM25Index(passages)
        self.dense=None
        if backend == "milvus":
            from .milvus_backend import MilvusDenseIndex
            kwargs={"uri":milvus_uri}
            if model_name: kwargs["model_name"]=model_name
            self.dense=MilvusDenseIndex(passages,**kwargs)
        elif dense:
            kwargs={"model_name":model_name} if model_name else {}
            self.dense=DenseIndex(passages,**kwargs)
    def search(self,query,k=5):
        if not self.dense: return self.sparse.search(query,k)
        lists=[self.sparse.search(query,max(20,k)),self.dense.search(query,max(20,k))]
        return weighted_rrf(lists, self.weights)[:k]
