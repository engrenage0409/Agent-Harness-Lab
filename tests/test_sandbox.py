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

def test_edit_exact_match_preserves_other_content(tmp_path):
    ws = Workspace(tmp_path / "sandbox")
    ws.write("src/main.py", "alpha\r\nbeta\r\ngamma\r\n")
    result = ws.edit("src/main.py", "beta", "BETA")
    assert result == {"edited": "src/main.py", "replacements": 1}
    with (tmp_path / "sandbox" / "src" / "main.py").open(
        encoding="utf-8", newline=""
    ) as file:
        assert file.read() == "alpha\r\nBETA\r\ngamma\r\n"


@pytest.mark.parametrize("old,new", [
    ("missing", "replacement"),
    ("repeat", "changed"),
    ("", "changed"),
    ("same", "same"),
])
def test_edit_rejects_bad_matches_without_changing_file(tmp_path, old, new):
    ws = Workspace(tmp_path / "sandbox")
    ws.write("example.txt", "repeat repeat same")
    with pytest.raises(ValueError):
        ws.edit("example.txt", old, new)
    assert ws.read("example.txt") == "repeat repeat same"


def test_edit_cannot_create_file_or_escape_workspace(tmp_path):
    ws = Workspace(tmp_path / "sandbox")
    with pytest.raises(ValueError):
        ws.edit("missing.txt", "a", "b")
    with pytest.raises(ValueError):
        ws.edit("../secret.txt", "a", "b")
    assert not (tmp_path / "secret.txt").exists()


def test_edit_rejects_oversize_result(tmp_path):
    ws = Workspace(tmp_path / "sandbox")
    ws.write("example.txt", "hello")
    with pytest.raises(ValueError, match="too large"):
        ws.edit("example.txt", "hello", "x" * 100_001)
    assert ws.read("example.txt") == "hello"


def test_edit_tool_registry_and_schema(tmp_path):
    from coding.sandbox import register_coding_tools
    from harness.contracts import ToolRegistry

    registry = ToolRegistry()
    ws = register_coding_tools(registry, tmp_path / "sandbox")
    ws.write("test.py", "value = 1")
    assert "edit_file" in [d["function"]["name"] for d in registry.definitions()]
    result = registry.execute("edit_file", {
        "path": "test.py", "old_text": "value = 1", "new_text": "value = 2"
    })
    assert result["replacements"] == 1
    assert ws.read("test.py") == "value = 2"
    assert registry.execute("edit_file", {"path": "test.py", "old_text": "2"})[
        "error_category"
    ] == "invalid_arguments"


def test_edit_read_roundtrip_preserves_crlf(tmp_path):
    ws = Workspace(tmp_path)
    target = tmp_path / 'code.py'
    target.write_bytes(b'def add(a, b):\r\n    return a - b\r\n')
    source = ws.read('code.py')
    ws.edit('code.py', source, source.replace('a - b', 'a + b'))
    assert target.read_bytes() == b'def add(a, b):\r\n    return a + b\r\n'


def test_edit_mixed_newlines_keeps_unmodified_bytes(tmp_path):
    ws = Workspace(tmp_path)
    target = tmp_path / 'code.py'
    target.write_bytes(b'first\r\nwrong\nlast\r\n')
    ws.edit('code.py', '\nwrong\n', '\nright\n')
    assert target.read_bytes() == b'first\nright\nlast\r\n'
