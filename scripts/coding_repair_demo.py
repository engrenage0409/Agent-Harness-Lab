"""Run a local-Qwen calculator Bug→pytest→edit→pytest experiment.

ONLY run on code you own/trust. pytest executes workspace/calculator.py; a
timeout, path allowlist, and fixed argv are NOT a secure sandbox.
"""
import argparse
import json
from pathlib import Path

from coding.repair import RepairTools, run_calculator_tests
from harness.provider import OpenAICompatibleProvider
from harness.runtime import AgentRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve", action="store_true",
                        help="I inspected/trust workspace/calculator.py and authorize pytest execution")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--trace", default="runs/coding_repair.jsonl")
    args = parser.parse_args()
    if not args.approve:
        parser.error("Explicit --approve required: pytest executes workspace code")
    repo_root = Path(__file__).resolve().parents[1]
    tools = RepairTools(
        repo_root / "workspace",
        lambda: run_calculator_tests(repo_root),
    )
    runtime = AgentRuntime(
        OpenAICompatibleProvider(), tools,
        max_steps=args.max_steps,
        trace_path=repo_root / args.trace,
        completion_check=tools.completion_feedback,
    )
    result = runtime.run(
        "Repair calculator.py. First call run_tests to inspect a failing test. "
        "Then read_file calculator.py, use edit_file to fix the bug, and call "
        "run_tests again. Stop only after tests pass. You may not execute other "
        "commands or modify other files.",
        system=(
            "You are a coding agent repairing a trusted local example. "
            "Call tools to inspect actual test feedback. Use only the "
            "provided read_file, edit_file and run_tests functions. "
            "Never claim success without a fresh passing test result."
        ),
    )
    verified = tools.completion_feedback(result.messages) is None
    summary = {
        "status": result.status, "steps": result.steps,
        "verified": result.status == "completed" and verified,
        "saw_failing_test": tools.saw_failing_test,
        "successful_edits": tools.successful_edits,
        "last_test_passed": tools.last_test_passed,
        "answer": result.content,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["verified"]:
        raise SystemExit("Repair was not independently verified; inspect trace")


if __name__ == "__main__":
    main()
