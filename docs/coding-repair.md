# Phase 3 — Calculator repair feedback loop

This is an **opt-in learning experiment**, not a production-ready coding
sandbox. It reuses the existing Qwen provider, AgentRuntime, ToolRegistry,
read_file and edit_file; it adds a fixed `run_tests` tool and a success gate.

## What is new?

- `coding/repair.py`: `RepairTools` registers `read_file`,
  `edit_file` and `run_tests`, but exposes **only the tool permitted in the current phase** (`run_tests → read_file → edit_file → run_tests`). A failed edit sends it back to `read_file`; attempts to skip phases are rejected even if the model sends an unadvertised tool call. Read/edit requests can address **only**
  `workspace/calculator.py`. The model cannot choose a shell command or
  a test filename.
- `run_tests` executes only the committed
  `examples/coding_agent/test_calculator.py` suite.
- `completion_feedback` rejects an early final message until a failing test
  has been observed, at least one edit succeeded, and a fresh subsequent test
  passed. It is deliberately a demo-specific gate.
- `tests/test_coding_repair.py` uses a scripted provider and fake runner to
  test the feedback loop **without executing generated code**.

## Try it locally, step by step

1. Inspect and trust your generated `workspace/calculator.py` and the test
   suite first. `pytest` **executes the generated Python module**; it can
   execute arbitrary code with the current user's privileges. Path limits
   and timeouts do not provide OS isolation.
2. Introduce a simple, deliberate bug in the trusted workspace file, e.g.
   change `return a + b` inside `add` to `return a - b`. Do **not**
   modify the committed acceptance tests.
3. Start the local Qwen server in one terminal:

   ```powershell
   $env:AGENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
   python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
   ```

4. In another terminal, run the experiment with explicit permission:

   ```powershell
   $env:AGENT_BASE_URL="http://127.0.0.1:8000/v1"
   $env:AGENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
   python -m scripts.coding_repair_demo --approve
   ```

   Review `runs/coding_repair.jsonl` and the changed calculator file.
   A small 0.5B model may fail to choose tools correctly, exhaust the step
   budget, or generate a bad patch. This is part of the experiment. The gate
   cannot force the LLM to reason correctly. Its purpose is to prevent a blind
   patch before inspecting source, and make unsuccessful edits require a fresh
   read. The test runner and workspace are not a hardened sandbox.

To run just offline tests (no Qwen, no generated-code execution):

```powershell
python -m pytest -q tests/test_coding_repair.py
```

**Safety:** Do not use this with untrusted code or sensitive files. For a
general-purpose Coding Agent, isolate the code in a disposable container/VM,
restrict file and network permissions, and require explicit user consent.

## Exact-edit recovery

Both tool registries request one line or the smallest unique exact source
snippet, never a whole-file replacement. Matching remains literal; there is
no fuzzy substitution or automatic unescaping. Missing matches get whitespace
and newline diagnostics; ambiguous matches request minimal adjacent context.
Edit errors return a next_action to reread the allowed source before retrying.

With the existing Qwen3-VL-2B server running, rerun from the repository root
(use the model ID configured by your server):

```powershell
$env:AGENT_BASE_URL="http://127.0.0.1:8000/v1"
$env:AGENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
python -m scripts.coding_repair_demo --approve --max-steps 16 --trace runs/coding_repair_exact_edit.jsonl
```

Inspect/trust the workspace first and ensure it contains the intended bug.
Offline regressions do not establish that a real model will complete repair.
