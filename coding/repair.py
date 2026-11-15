"""Constrained calculator repair loop, not an OS security sandbox.

The user must explicitly approve execution of trusted workspace code BEFORE
run_tests is registered. The model cannot select commands or test paths.
"""
import subprocess
import sys
from pathlib import Path

from coding.sandbox import Workspace
from harness.contracts import Tool, ToolRegistry

TARGET = "calculator.py"


def run_calculator_tests(repo_root, timeout=30):
    """Run the fixed, repository-owned acceptance test (executes workspace code)."""
    repo_root = Path(repo_root).resolve()
    test_file = repo_root / "examples" / "coding_agent" / "test_calculator.py"
    target = repo_root / "workspace" / TARGET
    if not test_file.is_file() or not target.is_file():
        return {"exit_code": None, "stdout": "",
                "stderr": "Missing acceptance test or calculator.py", "timed_out": False}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test_file)],
            cwd=repo_root, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout[-12000:],
                "stderr": proc.stderr[-12000:], "timed_out": False}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "",
                "stderr": "Approved test run timed out", "timed_out": True}


class RepairTools(ToolRegistry):
    """Only expose a single file and a pre-approved fixed test command."""

    def __init__(self, workspace, test_runner):
        super().__init__()
        self.workspace = Workspace(workspace)
        self.test_runner = test_runner
        self.edit_version = 0
        self.test_version = -1
        self.last_test_passed = False
        self.saw_failing_test = False
        self.successful_edits = 0
        self.phase = "test"  # test -> read -> edit -> verify; failed edit -> read
        self.exposed = set()

        self.add(Tool(
            "read_file", "Read calculator.py in the approved workspace.",
            {"type": "object", "properties": {"path": {"type": "string"}},
             "required": ["path"], "additionalProperties": False},
            self.read_file,
        ))
        self.add(Tool(
            "edit_file", "Replace exactly one occurrence in calculator.py.",
            {"type": "object", "properties": {
                "path": {"type": "string"}, "old_text": {"type": "string"},
                "new_text": {"type": "string"}},
             "required": ["path", "old_text", "new_text"],
             "additionalProperties": False},
            self.edit_file,
        ))
        self.add(Tool(
            "run_tests", "Run the fixed, user-approved calculator pytest suite.",
            {"type": "object", "properties": {},
             "required": [], "additionalProperties": False},
            self.run_tests,
        ))

    def definitions(self):
        # Stage-specific tool visibility helps small models choose the next action.
        allowed = {
            "test": {"run_tests"},
            "read": {"read_file"},
            "edit": {"edit_file"},
            "verify": {"run_tests"},
            "done": set(),
        }[self.phase]
        self.exposed = allowed.copy()
        return [tool.definition() for name, tool in self.tools.items()
                if name in allowed]

    def execute(self, name, arguments):
        self.last_events = []
        if name not in self.exposed:
            return self._error(
                f"Tool {name} is not allowed in phase {self.phase}. "
                f"Allowed: {sorted(self.exposed)}. Follow the next tool phase.",
                "workflow_violation",
                tool_name=name, phase=self.phase, allowed_tools=sorted(self.exposed),
            )
        # ToolRegistry handles schema validation and returns structured errors.
        previous_phase = self.phase
        result = super().execute(name, arguments)
        if self.phase != previous_phase:
            # No further call in this reply may reuse the previous phase.
            self.exposed = set()
        if name == "edit_file" and isinstance(result, dict) and "error" in result:
            # A bad patch must not be repeated without rereading the source.
            self.phase = "read"
            self.exposed = set()
        return result

    @staticmethod
    def _check_path(path):
        if path != TARGET:
            raise ValueError("Only calculator.py is permitted")

    def read_file(self, path):
        self._check_path(path)
        result = self.workspace.read(path)
        self.phase = "edit"
        return result

    def edit_file(self, path, old_text, new_text):
        self._check_path(path)
        result = self.workspace.edit(path, old_text, new_text)
        self.edit_version += 1
        self.successful_edits += 1
        self.last_test_passed = False
        self.phase = "verify"
        return result

    def run_tests(self):
        result = self.test_runner()
        passed = result.get("exit_code") == 0 and not result.get("timed_out", False)
        self.test_version = self.edit_version
        self.last_test_passed = passed
        if not passed:
            self.saw_failing_test = True
            self.phase = "read"
        elif self.saw_failing_test and self.successful_edits > 0:
            self.phase = "done"
        else:
            # Already passing without edits: not a repair success.
            self.phase = "done"
        return result

    def completion_feedback(self, _messages):
        if not self.saw_failing_test:
            return "Not finished: first run run_tests to observe the failing test."
        if self.successful_edits < 1:
            return "Not finished: read calculator.py and fix the source, not the test assertion."
        if self.test_version != self.edit_version or not self.last_test_passed:
            return "Not finished: run_tests must pass AFTER the latest edit."
        return None
