# Harness architecture

The agent executes a bounded loop: model response, approved tool invocation,
observations, and another model response. Each turn is logged in JSONL form.
The RAG layer chunks UTF-8 Markdown/text, retrieves passages with BM25, and optionally
fuses dense embedding rankings using reciprocal rank fusion.
Coding tools are limited to an explicit workspace path.
