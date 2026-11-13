"""Atomic JSON step checkpoints; stores complete messages incl. tool outputs. Local trusted runs only."""
import json,os,re,tempfile
from pathlib import Path

class CheckpointStore:
    def __init__(self,root="runs/checkpoints"):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def path(self,session_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}",session_id):
            raise ValueError("Invalid session id")
        return self.root/(session_id+".json")
    def save(self,session_id,state):
        target=self.path(session_id)
        fd,tmp=tempfile.mkstemp(prefix=".checkpoint-",dir=self.root)
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as f:
                json.dump(state,f,ensure_ascii=False)
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def load(self,session_id):
        path=self.path(session_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
