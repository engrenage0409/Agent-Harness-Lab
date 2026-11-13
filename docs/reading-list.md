# Agent Harness and RAG reading list

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — tool-aware thought/action/observation loops.
- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) — retrieve external knowledge to condition generation.
- [BM25 / probabilistic relevance](https://en.wikipedia.org/wiki/Okapi_BM25) — sparse lexical retrieval baseline.
- [Sentence Transformers](https://www.sbert.net/) — sentence embeddings for dense retrieval.
- [Reciprocal rank fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — combining retrieval rankings.
- [LangGraph](https://langchain-ai.github.io/langgraph/) — graph-structured agent orchestration, useful comparison.
- [SWE-bench](https://www.swebench.com/) — benchmark for coding-agent issue resolution.
- [RAGAS](https://docs.ragas.io/) — RAG answer and retrieval evaluation concepts.

## Roadmap toward an industrial knowledge agent

1. Add PDF/OCR ingestion with provenance and page/coordinate references.
2. Compare sparse BM25, dense embeddings, hybrid + reranking on labeled questions.
3. Add chunk overlap/size experiments, recall@k, MRR, nDCG, citation correctness.
4. Model contextual rules as evidence with versioned sources; enforce rule IDs and audit logs.
5. Connect specialized industrial agents only through explicitly authorized APIs.
6. Benchmark tasks and tool usage; add permission levels, execution sandboxing and human approval.

**Current implementation note:** this repository is a starter baseline, not a complete coding sandbox or production-ready secured agent runtime. The local file tools deliberately do not expose shell execution.
