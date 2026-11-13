"""Labeled retrieval metrics: Recall@k, MRR@k, nDCG@k, hit-rate@k."""
import math

def score_query(retrieved,relevant,k=5):
    ranked=list(retrieved)[:k]
    truth=set(relevant)
    if not truth: raise ValueError("Each query needs at least one relevant source_id")
    hits=[1 if sid in truth else 0 for sid in ranked]
    found=sum(hits)
    first=next((i+1 for i,v in enumerate(hits) if v),None)
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(hits))
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(truth))))
    return {"recall":found/len(truth),"hit_rate":float(bool(found)),
            "mrr":1/first if first else 0.0,"ndcg":dcg/ideal if ideal else 0.0}

def evaluate_queries(store,examples,k=5):
    if not examples: raise ValueError("Empty evaluation set")
    totals={"recall":0.0,"hit_rate":0.0,"mrr":0.0,"ndcg":0.0}
    rows=[]
    for example in examples:
        retrieved=store.query(example["query"],k)
        ids=[x["source_id"] for x in retrieved]
        metrics=score_query(ids,example["relevant_source_ids"],k)
        rows.append({"query":example["query"],"retrieved":ids,**metrics})
        for key in totals: totals[key]+=metrics[key]
    return {"k":k,"n":len(rows),"metrics":{key:val/len(rows) for key,val in totals.items()},
            "per_query":rows}
