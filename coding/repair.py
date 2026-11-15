"""Constrained calculator repair loop, not an OS security sandbox.

The user must explicitly approve execution of trusted workspace code BEFORE
run_tests is registered. The model cannot select commands or test paths.
"""
import subprocess
import sys
from coding.recovery import FileCheckpoint, ProgressTracker
from pathlib import Path

from coding.sandbox import EDIT_DESCRIPTION, Workspace
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
            [sys.executable, "-m", "pytest", "-q", "--tb=short", str(test_file)],
            cwd=repo_root, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        # Preserve full diagnostics locally; give the model only test evidence.
        log = repo_root / "runs" / "calculator_pytest.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        concise = "\n".join(line for line in proc.stdout.splitlines()
                            if (line.startswith("E ") and "where " not in line) or line.startswith("FAILED ")
                            or " passed" in line or " failed" in line)
        return {"exit_code": proc.returncode, "stdout": concise[-4000:],
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
        self.checkpoint = FileCheckpoint(self.workspace.resolve(TARGET))
        self.progress = ProgressTracker(repeat_limit=3)

        self.add(Tool(
            "read_file", "Read calculator.py in the approved workspace.",
            {"type": "object", "properties": {"path": {"type": "string"}},
             "required": ["path"], "additionalProperties": False},
            self.read_file,
        ))
        self.add(Tool(
            "edit_file", EDIT_DESCRIPTION,
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
            "stalled": set(),
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
        source_hash = self.checkpoint.expected_hash
        result = super().execute(name, arguments)
        if self.phase != previous_phase:
            # No further call in this reply may reuse the previous phase.
            self.exposed = set()
        if name == "edit_file" and isinstance(result, dict) and "error" in result:
            # A bad patch must not be repeated without rereading the source.
            count = self.progress.record_failure(source_hash, arguments or {}, result["error"])
            self.phase = "stalled" if self.progress.stalled else "read"
            self.exposed = set()
            result["repeated_failure_count"] = count
            result["next_action"] = self.next_action()
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
        self.checkpoint.ensure_unchanged()
        result = self.workspace.edit(path, old_text, new_text)
        self.checkpoint.record_edit()
        self.progress.reset()
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

    def next_action(self):
        return {
            "test": 'Call run_tests with arguments {} now.',
            "read": 'Call read_file with arguments {"path":"calculator.py"} now.',
            "edit": ('Call edit_file now with path="calculator.py". ' + EDIT_DESCRIPTION +
                     ' Change implementation only. Do not copy pytest output into the file.'),
            "verify": 'Call run_tests with arguments {} now to verify your edit.',
            "done": 'Report the actual test outcome. Do not call more tools.',
            "stalled": "Stop: the identical edit failed repeatedly. Report the blockage.",
        }[self.phase]

    def stop_reason(self):
        return "stalled" if self.progress.stalled else None

    def completion_feedback(self, _messages):
        if (self.saw_failing_test and self.successful_edits > 0
                and self.test_version == self.edit_version and self.last_test_passed):
            return None
        return "Repair is not verified. " + self.next_action()
