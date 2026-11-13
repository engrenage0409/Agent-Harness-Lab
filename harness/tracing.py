import json
from pathlib import Path
from datetime import datetime,timezone

class TraceWriter:
    def __init__(self,path=None):
        self.path=Path(path) if path else None
        if self.path: self.path.parent.mkdir(parents=True,exist_ok=True)
    def emit(self,event,**payload):
        record={"time":datetime.now(timezone.utc).isoformat(),"event":event,**payload}
        if self.path:
            with self.path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")
        return record
