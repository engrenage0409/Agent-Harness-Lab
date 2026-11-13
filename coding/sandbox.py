"""Allowlisted local file operations. Never let tool arguments escape workspace."""
from pathlib import Path
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
        return sorted(str(x.relative_to(self.root)) for x in self.root.rglob("*")
                      if x.is_file() and x.resolve().is_relative_to(self.root))[:200]
    def read(self,path):
        p=self.resolve(path)
        if p.stat().st_size>100_000: raise ValueError("File too large")
        return p.read_text(encoding="utf-8")
    def write(self,path,content):
        if len(content)>100_000: raise ValueError("Content too large")
        p=self.resolve(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(content,encoding="utf-8")
        return {"saved":str(p.relative_to(self.root))}

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
    return workspace
