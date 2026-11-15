# Coding Agent — local repair experiments, v1–v4

This is a **learning-scale, explicitly approved calculator repair experiment**, not a general-purpose Codex replacement or an OS-security sandbox. It combines `AgentRuntime`, a local Qwen OpenAI-compatible endpoint, `RepairTools`, a fixed pytest acceptance suite and a task-specific success gate.

## Architecture and evidence

```text
User task
  -> AgentRuntime: send messages + phase-specific tool definitions
  -> Qwen: choose next tool call and patch content
  -> RepairTools / ToolRegistry: validate arguments and phase
  -> run_tests (fixed pytest) | read_file | edit_file
  -> structured observation appended to messages
  -> Qwen: continue with new evidence
  -> completion_feedback: reject premature final answer
  -> completed only if failing test observed, edit applied,
     and latest code version subsequently passed tests
```

Relevant source: `harness/runtime.py`, `harness/contracts.py`, `coding/repair.py`, `coding/sandbox.py`, `coding/recovery.py`, `scripts/coding_repair_demo.py`, `serving/app.py`, `examples/coding_agent/test_calculator.py`.

The accepted workflow is `test → read → edit → verify → done`. A failed edit returns to `read`; a failing verification test also returns to `read`. Only `calculator.py` can be read or edited in this demo and the model does not choose the pytest command. The five committed acceptance tests cover add, subtract, multiply, divide and divide-by-zero.

**Evidence provenance:** results below are **operator-reported local live runs** from the development sessions, not checked-in trace files. The repository's offline pytest tests and CI validate harness logic with scripted/fake providers, not that every real model invocation succeeds. Do not treat the four iterative runs as a controlled benchmark.

## Experimental timeline

| Run | Model / configuration | Recorded observation | Diagnosis | Engineering response |
|---|---|---|---|---|
| **v1** | `Qwen/Qwen2.5-0.5B-Instruct` | Repair run exhausted available steps, `successful_edits=0`; failure assertion `assert -1 == 5` was visible | The model treated the pytest assertion as if it were an exact source fragment (`old_text="-1 == 5"`), did not inspect the file with `read_file` before attempting to edit | Constrain the action order; require a source-reading stage after failed tests |
| **v2** | Local Qwen experiment; exact model ID for this attempt not independently recorded here | Server returned HTTP 502 on an unknown/unadvertised tool name | Server rejected well-formed but unauthorized tool requests before the Harness could give actionable feedback | Forward syntactically valid model tool calls to the Harness, where `RepairTools.execute()` returns `workflow_violation`; malformed tool JSON remains an error |
| **v3** | `Qwen/Qwen3-VL-2B-Instruct` | `max_steps_exceeded` after 10 steps, no successful edits; source was read; repeated literal edit mismatch (`Expected exactly one match; found 0`) | The proposed logical fix to `add` was correct (`a-b → a+b`), but `old_text` contained whole-file content with newline escape/indentation discrepancies, so exact matching failed | Strengthen `EDIT_DESCRIPTION`: copy smallest unique exact snippet, avoid whole-file replacement, explain indentation/line-break errors and request reread |
| **v4** | `Qwen/Qwen3-VL-2B-Instruct` | **`status=completed`, `steps=5`, `verified=true`, `saw_failing_test=true`, `successful_edits=1`, `last_test_passed=true`** | The constrained loop completed a real local model-generated fix with a fresh passing test | Retain evidence-based completion check and exact-edit instructions; add conservative stalled-edit and recovery tests for the final MVP |

The successful v4 natural-language final answer was reported as:

> The tests have passed successfully after the fix. The bug in the `add` function has been corrected, and the test results confirm that the implementation is now correct.

That *model-written answer is not the verification evidence*. The `verified` result was calculated by Python from the Harness's tool and test state. Only the fixed calculator acceptance suite was checked. Detailed v4 tool-call arguments and all step-level outputs require the local `runs/coding_repair_v4.jsonl` to inspect; do not invent or assume their exact content.

## What changed, technically?

### 1. Phase-specific tool exposure, not hard-coded repair logic

`RepairTools.definitions()` exposes only one tool at a time according to `phase`:

- `test`: `run_tests`
- `read`: `read_file`
- `edit`: `edit_file`
- `verify`: `run_tests`
- `done` / `stalled`: no tools

`RepairTools.execute()` also checks phase authorization at execution time. The server forwards well-formed but unadvertised tool names for structured rejection; this does **not** grant permission to execute them.

The model still decides *what source edit to propose*. The harness does not contain a secret `replace "-" with "+"` fix.

### 2. Precise edits and observable failure

`Workspace.edit(path, old_text, new_text)` requires exactly one literal source match. An ambiguous or missing match changes no file and returns a structured error with a suggestion to reread and use a smaller exact snippet. The safe example is:

```json
{
  "path": "calculator.py",
  "old_text": "def add(a, b): return a - b",
  "new_text": "def add(a, b): return a + b"
}
```

This JSON is an **illustrative minimal patch**, **not** a verified copy of the v4 trace. Using only `"return a - b"` would be ambiguous if both `add` and `subtract` contain that line.

### 3. Fresh-test success gate

After a successful edit, `edit_version` increases and `last_test_passed` is reset to false. Running tests sets `test_version = edit_version`. The task-specific completion checker accepts a final answer only when all four predicates hold:

```python
saw_failing_test
and successful_edits > 0
and test_version == edit_version
and last_test_passed
```

A fluent answer from the model does **not** bypass this gate. This is evidence for the *tracked edits and fixed tests*, not proof of program correctness for untested inputs or an immutable on-disk artifact under arbitrary concurrency.

### 4. Bounded recovery and opt-in rollback (post-v4 MVP additions)

`ProgressTracker` fingerprints the source version, patch fields and exact error text. **After three occurrences of the same failed-edit signature**, the repair run returns `stalled`. Counts are **cumulative for identical fingerprints**, not strictly consecutive. This mechanism is covered by offline tests; it is **not** claimed to have been exercised in the earlier live v1–v4 trials.

`FileCheckpoint` stores the pre-run bytes of the approved single calculator file and generates a unified diff. If the CLI was started with `--rollback-on-failure`, an unsuccessful run attempts to restore the saved bytes. It refuses to overwrite detected intervening changes. This is an **in-memory, opt-in, limited rollback**, not Git history restoration, a persistent crash checkpoint, an OS sandbox or a guaranteed race-free atomic transaction.

There is **no implemented test-by-test regression-diff policy or general multi-file rollback**.

## Reproduce locally (Windows PowerShell)

```powershell
git switch main
git pull --ff-only origin main
python -m pip install -e ".[serve,test]"
python -m pytest -q
python -m pytest -v tests/test_recovery.py
```

A live run executes `workspace/calculator.py` as Python code. **Only use trusted, inspected local code**, and approve it explicitly. If your `add` function is already correct, the demo does not manufacture a failing test for you; the repair-specific success gate may not verify an already-passing codebase.

First, inspect `workspace/calculator.py` and, for a deliberate repair demonstration only, change `add` from addition to subtraction; do not edit the committed tests. Make sure the model is cached locally, since serving uses `local_files_only=True`.

Terminal 1:

```powershell
$env:AGENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
$env:AGENT_BASE_URL="http://127.0.0.1:8000/v1"
$env:AGENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
python -m scripts.coding_repair_demo --approve --trace runs/coding_repair_final.jsonl
```

Opt-in rollback for an unsuccessful run:

```powershell
python -m scripts.coding_repair_demo --approve --rollback-on-failure --trace runs/coding_repair_rollback.jsonl
```

Inspect outcomes and step-level evidence:

```powershell
Get-Content runs/coding_repair_final.jsonl |
    ConvertFrom-Json |
    Where-Object { $_.event -in @("assistant","tool_result","tool_error","completion_rejected","run_finished") } |
    Select-Object step,event,name,arguments,result,content,status |
    Format-List

Get-Content workspace/calculator.py
python -m pytest -q examples/coding_agent/test_calculator.py
```

The trace file path is **local**, not a URL checked into GitHub. If you still have the historical v1–v4 `runs/*.jsonl` files, inspect those directly to recover exact model arguments and individual failure timings.

For offline tests without model weights and without importing generated calculator code, use `python -m pytest -q`. GitHub Actions runs the offline suite on pull requests; it is not a live Qwen benchmark.

## Boundaries

- No general command-execution tool is exposed to the repair model; the fixed pytest command still **executes arbitrary code in the trusted workspace** with user privileges.
- Path allowlists, short timeouts, atomic file replacement and in-memory snapshots do **not** constitute an adversarial-code security boundary.
- The in-memory snapshot is lost on crash and protects only one file; an external writer can race with checks.
- Unknown-tool recovery, exact-edit diagnostics, repeated-failure stopping and evidence gating improve reliability, but they do not guarantee every model will repair every bug.
- No controlled success-rate comparison, real-world benchmark, multi-file test impact analysis, automatic regression rollback or production deployment has been conducted.

This is a bounded and inspectable **Coding Agent MVP**, intended to teach real model/tool feedback and support a reproducible small-scale project demonstration.
