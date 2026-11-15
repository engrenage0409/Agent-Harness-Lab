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


class PhaseGuidedProvider:
    """Append a current-stage reminder; the model still chooses every edit."""
    def __init__(self, provider, registry):
        self.provider = provider
        self.registry = registry

    def complete(self, messages, tools):
        return self.provider.complete(messages + [{"role": "user", "content":
            self.registry.next_action() + " Use a tool call, not a description of actions."
        }], tools)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approve", action="store_true",
                        help="I inspected/trust workspace/calculator.py and authorize pytest execution")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--rollback-on-failure", action="store_true",
                        help="Opt-in: restore the approved pre-run calculator after an unsuccessful run")
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
        PhaseGuidedProvider(OpenAICompatibleProvider(), tools), tools,
        max_steps=args.max_steps,
        trace_path=repo_root / args.trace,
        completion_check=tools.completion_feedback,
        stop_check=tools.stop_reason,
    )
    result = runtime.run(
        "Fix the bug in calculator.py. Run tests, read the source, edit the faulty "
        "function, then run tests again. Never modify tests. Use path calculator.py.",
        system=(
            "You are a coding agent repairing a trusted local example. "
            "Call tools to inspect actual test feedback. Use only the "
            "provided read_file, edit_file and run_tests functions. "
            "Never claim success without a fresh passing test result. "
            "Use available tools rather than describing future actions; "
            "do not ask the user to provide code already accessible with read_file."
        ),
    )
    verified = tools.completion_feedback(result.messages) is None
    diff = tools.checkpoint.diff() if tools.checkpoint else ""
    rolled_back = False
    rollback_error = None
    if args.rollback_on_failure and not (result.status == "completed" and verified):
        try:
            rolled_back = tools.checkpoint.rollback() if tools.checkpoint else False
        except (ValueError, OSError) as exc:
            rollback_error = str(exc)
    summary = {
        "status": result.status, "steps": result.steps,
        "verified": result.status == "completed" and verified,
        "saw_failing_test": tools.saw_failing_test,
        "successful_edits": tools.successful_edits,
        "last_test_passed": tools.last_test_passed,
        "answer": result.content,
        "patch_diff": diff,
        "rolled_back": rolled_back,
        "rollback_error": rollback_error,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["verified"]:
        raise SystemExit("Repair was not independently verified; inspect trace")


if __name__ == "__main__":
    main()
