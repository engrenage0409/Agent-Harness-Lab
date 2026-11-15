"""Minimal local OpenAI-style text chat API for an existing Hugging Face Qwen model.

Supports non-streaming text chat and Qwen2.5 Hermes-style function calling.
Run locally: python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("AGENT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
logger = logging.getLogger(__name__)


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


def tool_call_blocks(text):
    """Locate closing tags only outside JSON strings (including escaped quotes)."""
    start = 0
    while True:
        opening = text.find("<tool_call>", start)
        if opening < 0:
            return
        body = opening + len("<tool_call>")
        quoted = escaped = False
        for pos in range(body, len(text)):
            char = text[pos]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif text.startswith("</tool_call>", pos):
                end = pos + len("</tool_call>")
                yield opening, end, text[body:pos]
                start = end
                break
        else:
            return


def parse_qwen_tool_calls(text: str):
    """Convert Qwen's tool tags to OpenAI-style calls without executing tools."""
    matches = list(tool_call_blocks(text))
    if not matches:
        if "<tool_call>" in text or "</tool_call>" in text:
            raise ValueError("Incomplete Qwen tool-call tags")
        return text, []

    calls = []
    for match in matches:
        try:
            obj = json.loads(match[2])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in Qwen tool call at character {exc.pos}: {exc.msg}") from exc
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
    pieces = []
    cursor = 0
    for start, end, _ in matches:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    content = "".join(pieces).strip()
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


def completed_tool_call(text: str) -> bool:
    """Return True once the model has finished its first tool-call block."""
    return next(tool_call_blocks(text), None) is not None


class StopAfterFirstToolCall:
    """Transformers stopping criterion, applied only when tools are offered.

    End the generation at the FIRST closed tool_call, before the model has a
    chance to speculate about subsequent dependent calls.
    """

    def __init__(self, tokenizer, prompt_tokens: int):
        self.tokenizer = tokenizer
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids, scores, **kwargs):
        new_tokens = input_ids[0, self.prompt_tokens:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return completed_tool_call(generated_text)


class QwenBackend:
    """Loads cached weights once and generates a single text response."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    def load(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, local_files_only=True
        )
        config = AutoConfig.from_pretrained(self.model_id, local_files_only=True)
        loader = AutoModelForCausalLM
        if config.model_type == "qwen3_vl":
            from transformers import AutoModelForImageTextToText
            loader = AutoModelForImageTextToText
        self.model = loader.from_pretrained(
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
        if tools:
            from transformers import StoppingCriteriaList
            kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopAfterFirstToolCall(self.tokenizer, tokens.shape[-1])
            ])
        with torch.inference_mode():
            output = self.model.generate(tokens, attention_mask=torch.ones_like(tokens), **kwargs)
        generated = output[0, tokens.shape[-1]:]
        return (
            self.tokenizer.decode(generated, skip_special_tokens=False).split("<|im_end|>")[0],
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
            messages = [message.model_dump(exclude_none=True) for message in request.messages]
            prompt_tokens = completion_tokens = 0
            for attempt in range(2):
                text, used_prompt, used_completion = runner.generate(
                    messages, request.max_tokens, request.temperature, allowed_tools,
                )
                prompt_tokens += used_prompt
                completion_tokens += used_completion
                try:
                    content, tool_calls = parse_qwen_tool_calls(text)
                    break
                except ValueError as exc:
                    if allowed_tools and attempt == 0:
                        # Regenerate, never repair or execute partial JSON. Keep raw
                        # malformed output out of history and diagnostic responses.
                        messages = messages + [{"role": "user", "content": (
                            "Your previous response had invalid or incomplete tool JSON. "
                            "Return one short <tool_call> JSON object </tool_call>. "
                            "Use exactly name and arguments, with schema field names. "
                            "Escape newlines and quotes inside strings. Keep content concise."
                        )}]
                        continue
                    if os.getenv("AGENT_DEBUG_OUTPUT") == "1":
                        logger.warning("Rejected Qwen output (first 600 chars): %r", text[:600])
                    raise HTTPException(status_code=502, detail=(
                        f"{exc}; completion_tokens={used_completion}; "
                        f"max_tokens={request.max_tokens}; "
                        f"generation_limit_reached={used_completion >= request.max_tokens}. "
                        "Enable AGENT_DEBUG_OUTPUT=1 for raw output diagnostics."
                    )) from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        # Parsing validates the wire format, not execution permissions. A model
        # may select a tool hidden in this phase; preserve its call ID so the
        # runtime registry can deny it and return a correlated tool observation.
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
