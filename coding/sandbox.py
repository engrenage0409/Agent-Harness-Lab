"""Allowlisted local file operations. Never let tool arguments escape workspace."""
from pathlib import Path
import os
import tempfile
from harness.contracts import Tool,ToolRegistry

class Workspace:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
    def resolve(self,path):
        requested=Path(path)
        if requested.is_absolute():
            raise ValueError("Absolute paths are forbidden")
        candidate=(self.root/requested).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("Path escapes workspace")
        return candidate
    def list_files(self):
        return sorted(x.relative_to(self.root).as_posix() for x in self.root.rglob("*")
                      if x.is_file() and x.resolve().is_relative_to(self.root))[:200]
    def read(self,path):
        p=self.resolve(path)
        if p.stat().st_size>100_000: raise ValueError("File too large")
        return p.read_text(encoding="utf-8")
    def edit(self, path, old_text, new_text):
        """Replace exactly one literal match in an existing UTF-8 file.

        Refuse missing/ambiguous matches, oversize files, and unchanged edits.
        Use a same-directory temporary file and atomic replace for ordinary
        filesystem failures. This is not a hostile-code security sandbox.
        """
        if not isinstance(old_text, str) or not old_text:
            raise ValueError("old_text must be a nonempty string")
        if not isinstance(new_text, str) or old_text == new_text:
            raise ValueError("new_text must differ from old_text")
        target = self.resolve(path)
        if not target.is_file():
            raise ValueError("Target must be an existing file")
        if target.stat().st_size > 100_000:
            raise ValueError("File too large")
        with target.open("r", encoding="utf-8", newline="") as stream:
            content = stream.read()
        count = content.count(old_text)
        if count != 1:
            raise ValueError(f"Expected exactly one match; found {count}")
        updated = content.replace(old_text, new_text, 1)
        if len(updated.encode("utf-8")) > 100_000:
            raise ValueError("Updated file too large")

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", delete=False,
                dir=target.parent, prefix=".agent-edit-", suffix=".tmp",
            ) as stream:
                temporary = Path(stream.name)
                stream.write(updated)
            os.chmod(temporary, target.stat().st_mode)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {"edited": target.relative_to(self.root).as_posix(), "replacements": 1}
    def write(self,path,content):
        if len(content)>100_000: raise ValueError("Content too large")
        p=self.resolve(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        return {"saved":p.relative_to(self.root).as_posix()}

def register_coding_tools(registry:ToolRegistry,root):
    workspace=Workspace(root)
    def tool(name,description,properties,required,handler):
        registry.add(Tool(name,description,
            {"type":"object","properties":properties,"required":required,
             "additionalProperties":False},handler))
    tool("list_files","List workspace files (max 200).",{},[],workspace.list_files)
    tool("read_file","Read UTF-8 workspace file (up to 100KB).",
         {"path":{"type":"string"}},["path"],workspace.read)
    tool("write_file","Write UTF-8 workspace file; overwrites existing data. Use only with authorized tasks.",
         {"path":{"type":"string"},"content":{"type":"string"}},["path","content"],workspace.write)
    tool("edit_file","Replace exactly one literal occurrence in an existing UTF-8 workspace file.",
         {"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},
         ["path","old_text","new_text"],workspace.edit)
    return workspace
