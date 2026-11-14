"""Provider errors should surface the FastAPI reason instead of a bare HTTP 502."""
from io import BytesIO
from urllib.error import HTTPError

import pytest

from harness.provider import OpenAICompatibleProvider


def test_provider_surfaces_model_server_detail(monkeypatch):
    def failed_request(req, timeout):
        raise HTTPError(
            req.full_url, 502, "Bad Gateway", {},
            BytesIO(b'{"detail":"Model emitted unknown tool name"}'),
        )

    monkeypatch.setattr("harness.provider.urlopen", failed_request)
    provider = OpenAICompatibleProvider(
        model="test-qwen", base_url="http://127.0.0.1:8000/v1"
    )
    with pytest.raises(RuntimeError, match="Model emitted unknown tool name"):
        provider.complete([{"role": "user", "content": "hello"}], [])
