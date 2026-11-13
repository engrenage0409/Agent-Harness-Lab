# Agent Harness Lab — Tool-Calling Agent + Coding Tools + Hybrid RAG

A compact, inspectable Python implementation of an **agent runtime and local retrieval-augmented generation pipeline**. Designed to grow into a Coding Agent and industrial-document multi-agent platform.

**Status:** implementation scaffold, unit tests and offline demonstration provided. No live-provider evaluation or industrial integration has been executed yet. No claim of production sandbox security.

## What's inside

| Module | Implemented capability |
|---|---|
| `harness/` | bounded model/tool execution loop, tool registry, JSONL trace writer, scripted and OpenAI-compatible providers |
| `coding/` | allowlisted workspace file listing/reading/writing; path traversal rejection and file-size limits |
| `rag/` | Markdown/TXT ingestion, overlapping chunking, BM25 retrieval, optional sentence-transformer dense embeddings and hybrid reciprocal rank fusion |
| `scripts/` | offline agent demonstration, standalone knowledge-base search |
| `tests/` | retrieval, sandbox, tool loop tests |
| `docs/` | papers and production-extension roadmap |

## Architecture

```text
User request
  -> AgentRuntime (bounded tool loop, JSONL trace)
      -> OpenAI-compatible chat provider / scripted provider
      -> Tool Registry
          -> search_knowledge -> RAGStore -> chunking -> BM25 (+ optional dense RRF)
          -> list_files / read_file / write_file -> allowlisted workspace
      -> Tool observations -> model -> final response
```

The agent receives retrieved chunks with stable `source_id` identifiers like `policy.md#chunk-0` and should cite `[policy.md#chunk-0]` when synthesizing answers. **Retrieval is implemented; citation fidelity is not yet automatically verified.**

## Setup (Python 3.10+, Windows PowerShell)

```powershell
git clone https://github.com/engrenage0409/Agent-Harness-Lab.git
cd Agent-Harness-Lab
python -m pip install -e ".[test]"
python -m pytest -q
```

No model download or API credentials are needed for the offline demo.

### 1. Standalone RAG (BM25)

```powershell
python -m scripts.rag_search "How are sources cited?"
```

By default, retrieves passages from `examples/knowledge/`. To search your own documents, add UTF-8 `.md` or `.txt` files to a folder and run:

```powershell
python -m scripts.rag_search "rule validation" --docs my_documents
```

Optional dense embeddings + hybrid retrieval (downloads the English-oriented MiniLM model; adjust to multilingual embeddings for Chinese documents):

```powershell
python -m pip install -e ".[dense]"
python -m scripts.rag_search "tool invocation" --dense
```

This is genuinely **sparse/dense retrieval**, not a pretend semantic search: scores, text, source files and citation IDs are returned. No answer-generation LLM is called by `rag_search`.

### 2. Offline Harness + RAG + Coding Tools

```powershell
python -m scripts.agent_demo
```

The demo exercises a scripted tool-call cycle, BM25 retrieval, and JSONL tracing in `runs/demo.jsonl`. **Its final natural-language response is scripted rather than grounded generation**, so it is only a framework smoke test.

### 3. Connect an OpenAI-compatible model

Configure a model-serving endpoint you control (local `http://localhost:...` or HTTPS remote) and optionally an API key:

```powershell
$env:AGENT_BASE_URL="http://localhost:8000/v1"
$env:AGENT_MODEL="your-tool-calling-model"
$env:AGENT_API_KEY="your-key-if-needed"
python -m scripts.agent_demo --live --task "Find the citation policy and cite the source"
```

The provider sends `/chat/completions` requests. Your model must support OpenAI-style `tools` and `tool_calls`. Keep credentials in environment variables; do not commit keys.

**Safety:** this is educational code. The coding tools can overwrite files in the dedicated `workspace/` folder. The workspace checks reject absolute paths and `../` escapes, but they are **not a secure isolation boundary** against adversarial code or all filesystem races. There is deliberately no general shell-execution tool. Do not grant a model access to sensitive directories or use on untrusted code.

## RAG experiments to build next

- Documents: PDF/OCR ingestion, multilingual embeddings, metadata/ACL, source versions.
- Retrieval: BM25 vs dense vs hybrid, optional cross-encoder reranking, retrieval Recall@k/MRR/nDCG.
- Answering: grounded generation with enforced source citations, unsupported-claim detection and RAGAS-style evaluation.
- Coding Agent: authorized diffs, git workflow, tests and explicit approval before unsafe operations.
- Harness: state checkpointing, resumable workflows, costs/token budget, retries and multi-agent coordination.
- Industrial adaptation: reuse rules/DSL and evidence schema *without uploading confidential datasets or proprietary code*.

See [docs/reading-list.md](docs/reading-list.md).
