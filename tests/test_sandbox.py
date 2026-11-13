import pytest
from coding.sandbox import Workspace

def test_workspace(tmp_path):
    ws=Workspace(tmp_path/"sandbox")
    ws.write("notes/a.txt","hello")
    assert ws.read("notes/a.txt")=="hello"
    assert ws.list_files()==["notes/a.txt"]

def test_path_traversal(tmp_path):
    ws=Workspace(tmp_path/"sandbox")
    with pytest.raises(ValueError): ws.resolve("../secret.txt")
    with pytest.raises(ValueError): ws.resolve(str(tmp_path/"outside.txt"))
