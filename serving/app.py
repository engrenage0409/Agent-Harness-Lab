"""Minimal local OpenAI-style text chat API for an existing Hugging Face Qwen model.

Supports non-streaming text chat and Qwen2.5 Hermes-style function calling.
Run locally: python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
"""
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("AGENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=384, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_qwen_tool_calls(text: str):
    """Convert Qwen's tool tags to OpenAI-style calls without executing tools."""
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        if "<tool_call>" in text or "</tool_call>" in text:
            raise ValueError("Incomplete Qwen tool-call tags")
        return text, []

    calls = []
    for match in matches:
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON in Qwen tool call") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            raise ValueError("Qwen tool call is missing a function name")
        if not isinstance(obj.get("arguments"), dict):
            raise ValueError("Qwen tool-call arguments must be a JSON object")
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:12],
            "type": "function",
            "function": {
                "name": obj["name"],
                "arguments": json.dumps(obj["arguments"], ensure_ascii=False),
            },
        })
    content = TOOL_CALL_PATTERN.sub("", text).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("Unmatched Qwen tool-call tags")
    return content or None, calls


def qwen_messages(messages: list[dict]):
    """Adapt OpenAI tool-call history to Qwen2.5 tokenizer chat-template input."""
    converted = []
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                fn = call.get("function") or {}
                try:
                    arguments = json.loads(fn.get("arguments", "{}"))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("Invalid arguments in prior assistant tool call") from exc
                calls.append({"name": fn["name"], "arguments": arguments})
            converted.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": calls,
            })
        elif role == "tool":
            converted.append({
                "role": "tool",
                "name": message.get("name") or "tool",
                "content": message.get("content") or "",
            })
        else:
            converted.append({"role": role, "content": message.get("content") or ""})
    return converted


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

    def generate(self, messages: list[dict], max_tokens: int, temperature: float, tools=None):
        import torch

        options = {"tokenize": True, "add_generation_prompt": True,
                   "return_tensors": "pt"}
        if tools:
            options["tools"] = tools
        tokens = self.tokenizer.apply_chat_template(
            qwen_messages(messages), **options
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
        if request.tool_choice not in (None, "auto", "none"):
            raise HTTPException(status_code=400, detail="Only auto/none tool_choice is supported")
        if any(message.role not in ("system", "user", "assistant", "tool")
               for message in request.messages):
            raise HTTPException(status_code=400, detail="Unsupported message role")
        allowed_tools = request.tools if request.tool_choice != "none" else None
        if allowed_tools and any(
            tool.get("type") != "function" or not isinstance(tool.get("function"), dict)
            for tool in allowed_tools
        ):
            raise HTTPException(status_code=400, detail="Invalid tool definitions")
        try:
            text, prompt_tokens, completion_tokens = runner.generate(
                [message.model_dump(exclude_none=True) for message in request.messages],
                request.max_tokens,
                request.temperature,
                allowed_tools,
            )
            content, tool_calls = parse_qwen_tool_calls(text)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if tool_calls and not allowed_tools:
            raise HTTPException(status_code=502, detail="Model emitted unrequested tool calls")
        if tool_calls:
            known = {tool["function"].get("name") for tool in allowed_tools}
            if any(call["function"]["name"] not in known for call in tool_calls):
                raise HTTPException(status_code=502, detail="Model emitted unknown tool name")
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, **(
                    {"tool_calls": tool_calls} if tool_calls else {}
                )},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return api


app = create_app()
