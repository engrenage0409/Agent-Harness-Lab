"""Scoped SQLite key/value long-term memory. Explicit writes only; no automatic memory extraction."""
import json
import sqlite3
from pathlib import Path
from harness.contracts import Tool

class MemoryStore:
    def __init__(self,path="runs/memory.sqlite3",namespace="default"):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.namespace=namespace
        with self._connect() as con:
            con.execute("CREATE TABLE IF NOT EXISTS memories (namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(namespace,key))")
    def _connect(self):
        return sqlite3.connect(self.path,timeout=5)
    def put(self,key,value):
        if not key or len(key)>120 or len(value)>10000: raise ValueError("Memory size limit")
        with self._connect() as con:
            con.execute("INSERT OR REPLACE INTO memories VALUES (?,?,?)",(self.namespace,key,value))
        return {"saved":key}
    def get(self,key):
        with self._connect() as con:
            row=con.execute("SELECT value FROM memories WHERE namespace=? AND key=?",(self.namespace,key)).fetchone()
        return {"key":key,"value":row[0] if row else None}
    def search(self,query,k=5):
        from rag.retrieval import tokenize
        terms=set(tokenize(query))
        with self._connect() as con:
            rows=con.execute("SELECT key,value FROM memories WHERE namespace=? ORDER BY key",(self.namespace,)).fetchall()
        ranked=[]
        for key,value in rows:
            overlap=len(terms.intersection(set(tokenize(key+" "+value))))
            if overlap: ranked.append((overlap,key,value))
        return [{"key":key,"value":value,"score":score} for score,key,value in sorted(ranked,key=lambda x:(-x[0],x[1]))[:max(0,k)]]
    def delete(self,key):
        with self._connect() as con:
            count=con.execute("DELETE FROM memories WHERE namespace=? AND key=?",(self.namespace,key)).rowcount
        return {"deleted":bool(count)}

def register_memory_tools(registry,memory):
    def add(name,description,properties,required,fn):
        registry.add(Tool(name,description,{"type":"object","properties":properties,"required":required,
                                           "additionalProperties":False},fn))
    add("memory_get","Get previously stored memory by key",{"key":{"type":"string"}},["key"],memory.get)
    add("memory_put","Store explicit key/value memory, not secrets",
        {"key":{"type":"string"},"value":{"type":"string"}},["key","value"],memory.put)
    add("memory_search","Search stored memory lexically",{"query":{"type":"string"}},["query"],memory.search)
    add("memory_delete","Delete memory by key",{"key":{"type":"string"}},["key"],memory.delete)
    return registry
