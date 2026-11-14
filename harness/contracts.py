"""Tool definitions, argument validation, and conservative transient retries."""
from dataclasses import dataclass
from typing import Callable, Any
from jsonschema import validate, ValidationError


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]
    retry_safe: bool = False
    max_transient_retries: int = 1

    def definition(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }}


class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.last_events = []

    def add(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        if tool.max_transient_retries < 0:
            raise ValueError("max_transient_retries must be nonnegative")
        self.tools[tool.name] = tool
        return self

    def definitions(self):
        return [tool.definition() for tool in self.tools.values()]

    def _error(self, message, category, **extra):
        return {"error": message, "error_category": category, **extra}

    def execute(self, name, arguments):
        self.last_events = []
        tool = self.tools.get(name)
        if tool is None:
            return self._error(f"Unknown tool: {name}", "unknown_tool")
        if not isinstance(arguments, dict):
            return self._error("Tool arguments must be an object", "invalid_arguments")
        allowed = set(tool.parameters.get("properties", {}))
        required = set(tool.parameters.get("required", []))
        missing = required - set(arguments)
        unknown = set(arguments) - allowed
        if missing or unknown:
            return self._error(
                f"Invalid arguments. Missing={sorted(missing)} unexpected={sorted(unknown)}",
                "invalid_arguments",
            )
        try:
            validate(instance=arguments, schema=tool.parameters)
        except ValidationError as exc:
            return self._error(f"Invalid tool arguments: {exc.message}", "invalid_arguments")

        total_attempts = 1 + (tool.max_transient_retries if tool.retry_safe else 0)
        for attempt in range(1, total_attempts + 1):
            try:
                return tool.handler(**arguments)
            except (TimeoutError, ConnectionError) as exc:
                # Only explicitly retry-safe tools can be attempted again.
                transient = tool.retry_safe and attempt < total_attempts
                self.last_events.append({
                    "event": "tool_retry" if transient else "tool_error",
                    "name": name, "attempt": attempt,
                    "category": "transient_error",
                    "retry_safe": tool.retry_safe,
                })
                if transient:
                    continue
                return self._error(
                    f"{type(exc).__name__}: {exc}", "transient_error",
                    attempts=attempt, retry_safe=tool.retry_safe,
                )
            except Exception as exc:
                self.last_events.append({
                    "event": "tool_error", "name": name,
                    "attempt": attempt, "category": "execution_error",
                    "retry_safe": tool.retry_safe,
                })
                return self._error(
                    f"{type(exc).__name__}: {exc}", "execution_error",
                    attempts=attempt, retry_safe=tool.retry_safe,
                )
