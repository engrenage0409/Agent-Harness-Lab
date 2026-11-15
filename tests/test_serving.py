"""Model-free serving contract tests: no Qwen weights or GPU required."""
import json
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from serving.app import (create_app, parse_qwen_tool_calls, qwen_messages,
                         completed_tool_call, StopAfterFirstToolCall)


class FakeBackend:
    def __init__(self):
        self.loaded = False
        self.calls = []
        self.next_output = 'Hello from fake Qwen'

    def load(self):
        self.loaded = True

    def generate(self, messages, max_tokens, temperature, tools=None):
        self.calls.append((messages, max_tokens, temperature, tools))
        return (self.next_output, 5, 4)


def test_text_chat_api():
    backend = FakeBackend()
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        assert backend.loaded
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/v1/models").json()["data"][0]["id"] == "test-qwen"
        response = client.post("/v1/chat/completions", json={
            "model": "test-qwen",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "Hello from fake Qwen"
        assert body["usage"]["total_tokens"] == 9
        assert backend.calls[0][0][0]["role"] == "user"


def test_tool_call_response_and_result_roundtrip():
    backend = FakeBackend()
    backend.next_output = '<tool_call>\n{"name":"search_knowledge","arguments":{"query":"citations"}}\n</tool_call>'
    defs = [{"type": "function", "function": {
        "name": "search_knowledge", "description": "Search citations",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
    }}]
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        first = client.post("/v1/chat/completions", json={
            "model": "test-qwen",
            "messages": [{"role": "user", "content": "Find citation rules"}],
            "tools": defs,
        })
        assert first.status_code == 200, first.text
        choice = first.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "search_knowledge"
        assert json.loads(call["function"]["arguments"]) == {"query": "citations"}

        backend.next_output = "Here is the citation rule."
        second = client.post("/v1/chat/completions", json={
            "model": "test-qwen",
            "messages": [
                {"role": "user", "content": "Find citation rules"},
                choice["message"],
                {"role": "tool", "tool_call_id": call["id"],
                 "content": '{"source_id":"policy.md#chunk-0"}'},
            ],
            "tools": defs,
        })
        assert second.status_code == 200, second.text
        assert second.json()["choices"][0]["message"]["content"] == "Here is the citation rule."
        converted = qwen_messages(backend.calls[1][0])
        assert converted[1]["tool_calls"][0]["arguments"] == {"query": "citations"}
        assert converted[2]["role"] == "tool"


def test_malformed_tool_call_fails_closed():
    with pytest.raises(ValueError):
        parse_qwen_tool_calls('<tool_call>{"name":"bad","arguments":{oops}}</tool_call>')


def test_unknown_tool_call_is_forwarded_for_runtime_feedback():
    backend = FakeBackend()
    backend.next_output = '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-qwen",
            "messages": [{"role": "user", "content": "Calculate"}],
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
        })
        assert response.status_code == 200
        call = response.json()["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "unknown"
        assert call["id"]



def test_invalid_model_is_rejected():
    backend = FakeBackend()
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "other-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 404


def test_malformed_tool_call_reports_token_limit(monkeypatch):
    backend = FakeBackend()
    backend.next_output = '<tool_call>{"name":"search_knowledge","arguments":{bad}}</tool_call>'
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-qwen",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{"type": "function", "function": {"name": "search_knowledge"}}],
        })
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "Invalid JSON in Qwen tool call" in detail
        assert "completion_tokens=4" in detail
        assert "generation_limit_reached=False" in detail


def test_stop_after_first_complete_tool_call():
    """Simulate token-by-token decoding without loading Transformers or a model."""
    class FakeTokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            return "".join(chr(token) for token in tokens)

    class FakeIds:
        def __init__(self, content):
            self.content = content

        def __getitem__(self, key):
            row, selection = key
            assert row == 0
            return [ord(character) for character in self.content][selection]

    criterion = StopAfterFirstToolCall(FakeTokenizer(), 0)
    prefix = '<tool_call>{"name":"search_knowledge","arguments":{"query":"citation policy"}}'
    assert not completed_tool_call(prefix)
    assert not criterion(FakeIds(prefix), None)
    closed = prefix + '</tool_call>'
    assert completed_tool_call(closed)
    assert criterion(FakeIds(closed), None)
    # The model must stop before starting a second dependent write_file call.
    assert not ('<tool_call>' in closed[len(prefix):])


def test_tags_inside_json_strings_are_not_boundaries():
    arguments = {"path": "citation_report.md", "content": 'Literal </tool_call> and "quoted" text\\end'}
    body = '<tool_call>' + json.dumps({"name": "write_file", "arguments": arguments})
    assert not completed_tool_call(body)
    content, calls = parse_qwen_tool_calls(body + '</tool_call>')
    assert content is None
    assert json.loads(calls[0]["function"]["arguments"]) == arguments


def test_malformed_output_retries_once_then_succeeds():
    class RecoveringBackend(FakeBackend):
        def generate(self, *args):
            result = super().generate(*args)
            self.next_output = '<tool_call>{"name":"write_file","arguments":{"path":"citation_report.md","content":"Rule [policy.md#chunk-0]"}}</tool_call>'
            return result
    backend = RecoveringBackend()
    backend.next_output = '<tool_call>{bad}</tool_call>'
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post('/v1/chat/completions', json={
            'model': 'test-qwen', 'messages': [{'role': 'user', 'content': 'Save report'}],
            'tools': [{'type': 'function', 'function': {'name': 'write_file'}}],
        })
    assert response.status_code == 200
    assert len(backend.calls) == 2
    assert response.json()['usage']['completion_tokens'] == 8
    assert response.json()['choices'][0]['finish_reason'] == 'tool_calls'


def test_incomplete_second_call_is_not_silently_executed():
    with pytest.raises(ValueError):
        parse_qwen_tool_calls('<tool_call>{"name":"search","arguments":{}}</tool_call><tool_call>{')


@pytest.mark.parametrize('tools,choice', [([], 'auto'), (None, 'none')])
def test_unrequested_call_is_forwarded_without_execution(tools, choice):
    backend = FakeBackend()
    backend.next_output = '<tool_call>{"name":"run_tests","arguments":{}}</tool_call>'
    with TestClient(create_app(backend, model_id='test')) as client:
        response = client.post('/v1/chat/completions', json={
            'model': 'test', 'messages': [{'role': 'user', 'content': 'Done'}],
            'tools': tools, 'tool_choice': choice,
        })
    assert response.status_code == 200
    assert response.json()['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'run_tests'
    assert not backend.calls[0][3]
