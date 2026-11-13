"""Character-based overlap chunking for local UTF-8 documents."""
from pathlib import Path

def chunk_text(text,size=500,overlap=80):
    if size<1 or overlap<0 or overlap>=size: raise ValueError("Invalid chunk settings")
    if not text.strip(): return []
    return [text[start:start+size] for start in range(0,len(text),size-overlap)
            if text[start:start+size].strip()]

def load_documents(root,extensions=(".md",".txt")):
    root=Path(root).resolve()
    result=[]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions: continue
        if not path.resolve().is_relative_to(root): continue
        if path.stat().st_size>1_000_000: continue
        result.append({"source":str(path.relative_to(root)),"text":path.read_text(encoding="utf-8")})
    return result
