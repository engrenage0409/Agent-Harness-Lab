"""Model-free serving contract tests: no Qwen weights or GPU required."""
import json
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from serving.app import create_app, parse_qwen_tool_calls, qwen_messages


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


def test_unknown_tool_call_is_rejected():
    backend = FakeBackend()
    backend.next_output = '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-qwen",
            "messages": [{"role": "user", "content": "Calculate"}],
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
        })
        assert response.status_code == 502



def test_invalid_model_is_rejected():
    backend = FakeBackend()
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "other-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 404
