"""Tool-driven agent runtime with bounded execution and structured traces."""
from .runtime import AgentRuntime, RunResult
from .provider import OpenAICompatibleProvider, ScriptedProvider
