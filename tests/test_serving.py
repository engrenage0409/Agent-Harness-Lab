"""Model-free serving contract tests: no Qwen weights or GPU required."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from serving.app import create_app


class FakeBackend:
    def __init__(self):
        self.loaded = False
        self.calls = []

    def load(self):
        self.loaded = True

    def generate(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        return ("Hello from fake Qwen", 5, 4)


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


def test_unsupported_tool_calls_are_rejected():
    backend = FakeBackend()
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "test-qwen",
            "messages": [{"role": "user", "content": "Calculate"}],
            "tools": [{"type": "function", "function": {"name": "calculator"}}],
        })
        assert response.status_code == 400
        assert backend.calls == []


def test_invalid_model_is_rejected():
    backend = FakeBackend()
    with TestClient(create_app(backend, model_id="test-qwen")) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "other-model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert response.status_code == 404
