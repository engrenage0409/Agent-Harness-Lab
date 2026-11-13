# Phase 2 architecture and boundaries

## Memory
SQLite key/value storage scoped by namespace. The model may search, read, write or delete through explicit registered tools. There is no semantic vector memory, automatic summarization, privacy filtering, or approval gate built into this experimental API. Do not store personal secrets.

## Checkpoint / Resume
`CheckpointStore` saves complete model/tool conversation and pending tool calls after each transition. Resuming a completed task returns the completed result. Completed tool observations are not replayed. If the process crashes after starting a side-effecting tool but before recording its result, the session enters `needs_review` instead of re-running it automatically. Checkpoint data is local, not encrypted and not robust to concurrent writers or tampering.

## Multi-agent
`SequentialOrchestrator` executes named specialist agents in a fixed order, passing prior text output to the next agent as **untrusted** context. No parallel scheduling, task graph, MCP network agent routing, or conflict resolution is claimed.

## RAG Evaluation
`rag/evaluation.py` calculates document/passage-level Recall@K, MRR@K, nDCG@K and hit-rate. Labels must identify **source_id**, not merely filenames. Real comparisons need enough labeled queries and disjoint test sets. Answer-level groundedness and citation faithfulness are not measured yet.

## Reranking
Optional `CrossEncoderReranker` reranks the candidate pool. This loads a separate model and may consume more GPU/CPU resources. Default English cross-encoder is unsuitable for specialized multilingual industrial documents without replacement.

## Coding tests
Explicit CLI-only pytest runner, not an exposed model tool. Test execution can run malicious code and must only be used on **trusted** workspaces in isolated environments. `--approve` is only a user-supplied acknowledgement, not an OS sandbox. No shell agent tool is enabled.

## Future work
Durable task queues, optimistic locking, approval workflows, retrieval ACL and provenance, reranker benchmarking, model-based routing, retry policy, context compaction, structured handoff artifacts, and full production-level telemetry.
