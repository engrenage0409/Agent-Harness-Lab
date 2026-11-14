"""Minimal local OpenAI-style text chat API for an existing Hugging Face Qwen model.

Phase 1 supports text-only, non-streaming chat completions. Tool calling comes next.
Run locally: python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
"""
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("AGENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class QwenBackend:
    """Loads cached weights once and generates a single text response."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, local_files_only=True, dtype="auto"
        )
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def generate(self, messages: list[dict], max_tokens: int, temperature: float):
        import torch

        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.inference_mode():
            output = self.model.generate(tokens, **kwargs)
        generated = output[0, tokens.shape[-1]:]
        return (
            self.tokenizer.decode(generated, skip_special_tokens=True),
            int(tokens.shape[-1]),
            int(generated.shape[-1]),
        )


def create_app(backend=None, model_id: str | None = None):
    selected_model = model_id or MODEL_ID
    runner = backend if backend is not None else QwenBackend(selected_model)

    @asynccontextmanager
    async def lifespan(app):
        runner.load()
        yield

    api = FastAPI(title="Agent Harness Local Qwen Server", lifespan=lifespan)

    @api.get("/health")
    def health():
        return {"status": "ok", "model": selected_model}

    @api.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": selected_model, "object": "model"}]}

    @api.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        if request.model != selected_model:
            raise HTTPException(status_code=404, detail="Model not served here")
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming not supported yet")
        if request.tools or request.tool_choice not in (None, "none"):
            raise HTTPException(status_code=400, detail="Tool calling not supported yet")
        if any(message.role not in ("system", "user", "assistant") for message in request.messages):
            raise HTTPException(status_code=400, detail="Only text chat roles are supported")
        text, prompt_tokens, completion_tokens = runner.generate(
            [message.model_dump() for message in request.messages],
            request.max_tokens,
            request.temperature,
        )
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return api


app = create_app()
