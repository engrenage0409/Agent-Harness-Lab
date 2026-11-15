# Agent Harness Lab — Tool-Calling Agent + Coding Tools + Hybrid RAG

A compact, inspectable Python implementation of an **agent runtime, persistent memory, resumable tool loop, sequential multi-agent orchestration and local retrieval-augmented generation pipeline**. Designed as a learning scaffold for coding agents and industrial-document workflows.

**Status:** educational MVP with offline tests and an operator-reported live Qwen3-VL-2B calculator repair run (`completed`, 5 model steps, 1 edit, verified). This is a narrow, local demonstration—not a production-grade coding sandbox, general-purpose repository agent, or industrial deployment.

## What's inside

| Module | Implemented capability |
|---|---|
| `harness/` | bounded model/tool execution loop, tool registry, JSONL traces, SQLite memory, JSON checkpoints/resume, sequential multi-agent orchestrator, scripted and OpenAI-compatible providers |
| `coding/` | scoped workspace list/read/write/edit; separately approved fixed calculator pytest tool; repeat-failure detector, file diff and opt-in rollback |
| `rag/` | Markdown/TXT ingestion, overlapping chunking, BM25, optional dense/hybrid RRF, optional CrossEncoder reranking; labeled Recall@K/MRR/nDCG/HitRate evaluation |
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

## Local Qwen Model Server — text and tool calling

This optional experiment serves an already cached Hugging Face Qwen model via a
small FastAPI service. **No Ollama or model re-download is required.** It loads
weights locally at server startup and serves `/health`, `/v1/models`, and
`/v1/chat/completions`. The server supports Qwen2.5 Hermes-style `<tool_call>` parsing and tool-result
history for the existing live AgentRuntime. Streaming is not implemented.
Malformed tool-call JSON fails explicitly; unknown tool names are forwarded for runtime rejection and structured feedback (not executed). Actual tool execution,
argument checks and trace logging are handled by the Harness, not the server.

In Windows PowerShell:

```powershell
git pull origin main
python -m pip install -e ".[serve,test]"
python -m pytest -q
$env:AGENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

The server uses `local_files_only=True` and will fail startup if that model
is not fully cached. Run the following request **in a second PowerShell window**:

```powershell
$body = @{
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    messages = @(@{role="user"; content="Explain what an AI agent is in one sentence."})
    max_tokens = 100
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
```

In a second PowerShell window, run the live Agent Loop with the server already
running in the first window:

```powershell
$env:AGENT_BASE_URL="http://127.0.0.1:8000/v1"
$env:AGENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
python -m scripts.agent_demo --live --task "Search the knowledge base for citation rules and cite the source_id you find."
```

Inspect `runs/demo.jsonl` for `assistant` and `tool_result` events. A 0.5B
model can fail to select the correct tool; a completed status alone does not
prove that retrieval happened. This is a learning implementation, not a
production-safe service. It is localhost-only, without authentication,
concurrency controls or streaming.

## Sequential tool calling: search and save

This demonstration reuses the existing AgentRuntime, RAG `search_knowledge`, and
allowlisted workspace `write_file`. It asks the agent to retrieve the citation
policy and save `workspace/citation_report.md` with a source ID.

Offline deterministic integration check (no model server needed):

```powershell
python -m scripts.multi_tool_demo
```

To run with Qwen instead, first start the local model server described above,
then in another PowerShell terminal:

```powershell
$env:AGENT_BASE_URL="http://127.0.0.1:8000/v1"
$env:AGENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
python -m scripts.multi_tool_demo --live
```

Review `runs/multi_tool_demo.jsonl` and `workspace/citation_report.md`.
The demo now uses a task-specific evidence-first tool gate: first only
`search_knowledge` is visible and executable; after its tool observation, only
`write_file` is available. The local Qwen server stops generation after the first completed tool-call
block whenever tools are provided, so a model turn does not generate multiple
dependent calls before observations arrive. Writing also requires a source ID returned by the
search. This blocks same-turn dependent writes and checks basic citation presence;
it does **not** establish full factual faithfulness. The demo checks for a
successful search result, a successful file write, separate model turns, and an
expected citation. It prints `end_to_end_verified`.
In offline mode, the tool decisions and report text are scripted, not
generated from retrieval. In live mode, the demo uses `AgentRuntime(completion_check=...)`: a final
answer before the evidence-backed file write is rejected, with a corrective
message added to the conversation. The loop remains bounded by `max_steps`,
and each rejection is logged as `completion_rejected`. This does not guarantee
that a small model will comply; inspect the trace and output file.
Note that write_file overwrites files in the configured dedicated workspace;
do not use sensitive workspaces.

## Tool Error Recovery — bounded and conservative

The ToolRegistry validates argument types and required fields before invoking a
handler. Invalid arguments and unknown tools return structured observations
(`error` plus `error_category`), so the model can correct its next call.
`AgentRuntime` records `tool_error` and `tool_retry` events in JSONL traces.

Only tools **explicitly declared** `retry_safe=True` may automatically retry
`TimeoutError` and `ConnectionError` (one additional attempt by default).
`search_knowledge` opts in. A model-facing `write_file` remains non-retryable
because an interrupted write might have already changed state. Other exceptions
do not trigger automatic retries. The run is still bounded by `max_steps`.

The multi-tool demo also rejects any bracketed `#chunk-N` citations that
were not returned by its search result, even if another real citation is present.
This prevents the earlier `[citation.md#chunk-0]` error but does not validate
all natural-language factual claims.

```powershell
git pull origin main
python -m pytest -q
python -m scripts.multi_tool_demo
python -m scripts.multi_tool_demo --live
Get-Content runs/multi_tool_demo.jsonl -Tail 12
```

## Phase 2: persistent Agent Memory

Memory is **explicit key/value storage** in local SQLite, scoped by namespace. It is not automatic fact extraction or vector memory. The offline demo calls `memory_put` and `memory_get` tools and saves checkpoint state:

```powershell
python -m scripts.agent_memory_demo
```

See `harness/memory.py` for `MemoryStore`, `memory_search`, `memory_delete`, and `register_memory_tools`. The stored database is at `runs/demo_memory.sqlite3`, excluded from Git. Do not put credentials or sensitive user information in it.

## Phase 2: resumable Agent execution

`AgentRuntime(..., checkpoints=CheckpointStore("runs/checkpoints"))` supports `run(task,session_id="task-001")` and later `run(task,session_id="task-001",resume=True)`.

State is saved after assistant/tool events. Successfully completed tool calls are not replayed. **If a crash occurs after invoking a side-effecting tool and before saving its observation, resume stops with `needs_review`**; do not automatically replay the operation. There is no automatic approval/resolution protocol or secure checkpoint encryption. See `tests/test_checkpoint.py`.

## Phase 2: sequential Multi-Agent

`harness/orchestrator.py` provides `AgentSpec` and `SequentialOrchestrator`; use it to connect retrieval, planning, and response agents in a fixed sequence. Previous agent output is passed as untrusted context, not authoritative evidence. Parallel scheduling, multi-agent graph routing and consensus are **not** implemented.

## Phase 2: RAG Reranker and Evaluation

Use `--rerank` for the optional pretrained CrossEncoder reranker (requires sentence-transformers):

```powershell
python -m pip install -e ".[dense]"
python -m scripts.rag_search "citation policy" --rerank
python -m scripts.rag_search "agent loop" --dense --rerank
```

The included `examples/rag_eval.jsonl` labels refer to source IDs in the toy knowledge base. Run labeled retrieval evaluation:

```powershell
python -m scripts.rag_evaluate
python -m scripts.rag_evaluate --dense --rerank -k 5
```

Outputs `runs/retrieval_eval.json`: **Recall@K, MRR@K, nDCG@K and HitRate@K** plus per-question source IDs. These evaluate retrieval, **not generated-answer factuality or citation faithfulness**. A larger, hand-labeled held-out corpus is required before meaningful benchmarking.

## Phase 2: manually approved Coding Agent test runner

`coding/test_runner.py` runs `python -m pytest -q tests` only through an **explicitly approved CLI**:

```powershell
python -m scripts.run_tests --workspace workspace --approve --timeout 30
```

This **executes arbitrary test code**. A timeout and workspace path are *not* a security boundary. Run only code you own or trust in an isolated/disposable environment. The command is deliberately **not** exposed to the model as a tool.

More detailed tradeoffs: [Phase 2 design and limitations](docs/phase2.md).

## RAG experiments to build next

- Documents: PDF/OCR ingestion, multilingual embeddings, metadata/ACL, source versions.
- Retrieval: perform controlled BM25 vs dense vs hybrid+reranking ablations on a labeled held-out dataset; add multilingual domain embeddings.
- Answering: grounded generation with enforced source citations, unsupported-claim detection and RAGAS-style evaluation.
- Coding Agent: authorized diffs, git workflow, test execution in an actual OS/container sandbox, plus approvals.
- Harness: safe resolution of ambiguous tool execution, state concurrency, retries, budget control and multi-agent task graphs.
- Industrial adaptation: reuse rules/DSL and evidence schema *without uploading confidential datasets or proprietary code*.

See [docs/reading-list.md](docs/reading-list.md).


## Coding Agent MVP — verified repair and bounded recovery

This prototype is **not Codex**. It fixes a trusted local `workspace/calculator.py`
using real model-generated edits, fixed acceptance tests, tool-stage constraints,
and an evidence-based completion gate. The operator reported a successful
Qwen3-VL-2B run (5 model steps, 1 successful edit, latest pytest passed).

The final learning increment adds:
- `coding/recovery.py`: fingerprints repeated *identical* failed edits against
  the same file version and error; after three repeats, returns `stalled`
  rather than consuming the remaining model-step budget.
- An in-memory pre-run `calculator.py` snapshot, unified diff output, and
  **opt-in** rollback on unsuccessful runs. Rollback refuses to overwrite
  external changes detected after the agent's last recorded edit.
- The model cannot choose arbitrary pytest commands or file paths; only the
  pre-approved calculator suite is executed.

Offline regression tests (no model service needed):

```powershell
git switch main
git pull --ff-only origin main
python -m pytest -q
```

Live demo (start a locally cached Qwen model server first; use the **same**
`AGENT_MODEL` value for both processes):

```powershell
$env:AGENT_BASE_URL="http://127.0.0.1:8000/v1"
$env:AGENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
python -m scripts.coding_repair_demo --approve --trace runs/coding_repair_final.jsonl
```

For an experimental failed run, opt in to restoring only the trusted calculator
snapshot (the result still records the attempted unified diff):

```powershell
python -m scripts.coding_repair_demo --approve --rollback-on-failure --trace runs/coding_repair_rollback.jsonl
```

The snapshot lives only in the current Python process; it is **not** a
crash-persistent recovery mechanism. It does not detect every concurrent-write
race, and the file path restriction is not an OS sandbox. Inspect and trust
the generated code before executing pytest; use isolation for untrusted code.

Full walkthrough, observed v1–v4 experiments, failure analysis, exact verification and trace inspection: [Coding Agent experiment log](docs/coding-repair.md).

## Live Coding Agent experiments (v1–v4)

Four successive local trials were used to diagnose **model reasoning vs. tool-protocol vs. exact-edit failures**. The table records the operator's reported observations; these runs were **not** a repeated-sample benchmark. The `runs/*.jsonl` traces and `workspace/calculator.py` were local artifacts and are not bundled as reproducible outputs in this repository.

| Trial | Model | Observed result | Finding → engineering response |
|---|---|---|---|
| v1 | Qwen2.5-0.5B-Instruct | Incomplete; no successful edit; step budget exhausted | Model used pytest assertion text as if it were source and attempted a nonmatching patch without reading actual code → introduced a more explicit test/read/edit/verify tool sequence. |
| v2 | Local Qwen repair experiment (later switched to Qwen3-VL-2B) | Server HTTP 502 for an unknown/unadvertised tool name | The serving layer rejected the model's tool name before Harness could return structured feedback → route well-formed calls to ToolRegistry for allowed-tool checks; malformed tool JSON still fails. |
| v3 | Qwen3-VL-2B-Instruct | `max_steps_exceeded` at 10 steps; zero successful edits | Model identified the intended `add` fix, but submitted a whole-file `old_text` whose newline/indentation representation failed literal matching → clarified exact-edit instructions, minimal unique snippets and error diagnostics. |
| v4 | Qwen3-VL-2B-Instruct | **`completed`, 5 model steps, 1 successful edit, `verified=true`, `last_test_passed=true`** | A constrained test → read → edit → test loop succeeded; evidence gate required an initial failing test and a fresh passing test after the edit. |

**Interpretation:** v1–v4 are iterative debugging observations, **not** a controlled model ablation or measured success rate. The v4 final summary establishes a passing outcome for the fixed calculator tests, not general coding reliability. The precise v2 model/configuration and exact v4 patch arguments should be recovered from the corresponding local traces rather than inferred.

### Reproduce and inspect

Use only your own, inspected `workspace/calculator.py`; pytest imports and runs this file. With a deliberate bug in `add`, start the cached `Qwen/Qwen3-VL-2B-Instruct` server and run the approved client:

```powershell
git switch main
git pull --ff-only origin main
python -m pip install -e ".[serve,test]"
python -m pytest -q

# Terminal 1: model weights must already be cached locally
$env:AGENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000

# Terminal 2:
$env:AGENT_BASE_URL="http://127.0.0.1:8000/v1"
$env:AGENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
python -m scripts.coding_repair_demo --approve --trace runs/coding_repair_final.jsonl
```

Inspect actual model/tool decisions (PowerShell):

```powershell
Get-Content runs/coding_repair_final.jsonl |
  ConvertFrom-Json |
  Where-Object { $_.event -in @("assistant","tool_result","tool_error","completion_rejected","run_finished") } |
  Select-Object step,event,name,arguments,result,content,status |
  Format-List
```

For the new recovery safeguards without running the model or trusted workspace code:

```powershell
python -m pytest -v tests/test_recovery.py
```

See [the detailed experiment log](docs/coding-repair.md) for failure signatures, causal hypotheses, repair-stage constraints, results, and limitations.
