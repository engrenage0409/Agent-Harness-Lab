"""Minimal JSON-schema-like tool contracts without runtime dependencies."""
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]

    def definition(self):
        return {"type":"function","function":{"name":self.name,
               "description":self.description,"parameters":self.parameters}}

class ToolRegistry:
    def __init__(self):
        self.tools={}
    def add(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self.tools[tool.name]=tool
        return self
    def definitions(self):
        return [tool.definition() for tool in self.tools.values()]
    def execute(self,name,arguments):
        tool=self.tools.get(name)
        if tool is None:
            return {"error":f"Unknown tool: {name}"}
        if not isinstance(arguments,dict):
            return {"error":"Tool arguments must be an object"}
        allowed=set(tool.parameters.get("properties",{}))
        required=set(tool.parameters.get("required",[]))
        missing=required-set(arguments)
        unknown=set(arguments)-allowed
        if missing or unknown:
            return {"error":f"Invalid arguments. Missing={sorted(missing)} unexpected={sorted(unknown)}"}
        try:
            return tool.handler(**arguments)
        except Exception as exc:
            return {"error":f"{type(exc).__name__}: {exc}"}
