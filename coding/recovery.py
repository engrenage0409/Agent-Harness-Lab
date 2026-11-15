"""Small single-file recovery helpers. No OS sandbox or automatic Git reset."""
import difflib
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FileCheckpoint:
    """In-memory pre-run snapshot with conservative, opt-in rollback.

    Restore only when the current bytes still match the last edit made by
    this agent. Never overwrite unrelated edits performed after that edit.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.before = self.path.read_bytes()
        self.expected_hash = digest(self.before)
        self.edited = False

    def ensure_unchanged(self):
        if digest(self.path.read_bytes()) != self.expected_hash:
            raise ValueError("File changed outside this agent; refusing to overwrite")

    def record_edit(self):
        self.expected_hash = digest(self.path.read_bytes())
        self.edited = True

    def diff(self):
        current = self.path.read_bytes()
        before = self.before.decode("utf-8").splitlines(keepends=True)
        after = current.decode("utf-8").splitlines(keepends=True)
        return "".join(difflib.unified_diff(
            before, after, fromfile="calculator.py (before)",
            tofile="calculator.py (current)",
        ))

    def rollback(self):
        if not self.edited:
            return False
        self.ensure_unchanged()
        # Only the explicitly approved calculator snapshot can be restored;
        # no arbitrary paths, git reset, or shell commands.
        import os
        import tempfile
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent, prefix=".agent-rollback-",
                suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(self.before)
            os.chmod(temporary, self.path.stat().st_mode)
            self.ensure_unchanged()
            os.replace(temporary, self.path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        self.expected_hash = digest(self.before)
        self.edited = False
        return True


class ProgressTracker:
    """Detect repeated identical failed edits against the same file version."""

    def __init__(self, repeat_limit=3):
        if repeat_limit < 2:
            raise ValueError("repeat_limit must be at least 2")
        self.repeat_limit = repeat_limit
        self.counts = {}
        self.stalled = False

    def record_failure(self, source_hash, arguments, error):
        payload = json.dumps(
            [source_hash, arguments.get("path"), arguments.get("old_text"),
             arguments.get("new_text"), error],
            ensure_ascii=False, sort_keys=True,
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.counts[key] = self.counts.get(key, 0) + 1
        count = self.counts[key]
        if count >= self.repeat_limit:
            self.stalled = True
        return count

    def reset(self):
        self.counts.clear()
        self.stalled = False
