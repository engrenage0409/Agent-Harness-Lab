from harness.memory import MemoryStore

def test_scoped_memory(tmp_path):
    a=MemoryStore(tmp_path/"memory.db","a")
    b=MemoryStore(tmp_path/"memory.db","b")
    a.put("project","visual grounding")
    assert a.get("project")["value"]=="visual grounding"
    assert b.get("project")["value"] is None
    assert a.search("grounding")[0]["key"]=="project"
    assert a.delete("project")["deleted"]
    assert a.get("project")["value"] is None
