"""Manual-only allowlisted pytest launcher. Not registered as an autonomous agent tool.

WARNING: pytest executes arbitrary Python in tests/conftest; a timeout is NOT a security sandbox.
Only run trusted tests in a disposable environment with explicit --approve.
"""
import subprocess,sys
from pathlib import Path

def run_tests(workspace,approved=False,timeout=30):
    if not approved: raise PermissionError("Explicit operator approval required")
    workspace=Path(workspace).resolve()
    if not workspace.is_dir(): raise ValueError("Workspace directory does not exist")
    # Fixed argv. No shell; pytest is still fully trusted-code execution.
    try:
        result=subprocess.run([sys.executable,"-m","pytest","-q","tests"],
            cwd=workspace,capture_output=True,text=True,timeout=timeout,check=False)
        return {"exit_code":result.returncode,"stdout":result.stdout[-12000:],
                "stderr":result.stderr[-12000:],"timed_out":False}
    except subprocess.TimeoutExpired:
        return {"exit_code":None,"stdout":"","stderr":"Test run timed out","timed_out":True}
