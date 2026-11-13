"""Provider adapter: OpenAI-compatible /chat/completions endpoint or offline scripted replies."""
import json
import os
from urllib.request import Request,urlopen

class ScriptedProvider:
    def __init__(self,responses):
        self.responses=iter(responses)
    def complete(self,messages,tools):
        try:
            return next(self.responses)
        except StopIteration:
            return {"role":"assistant","content":"Scripted responses exhausted."}

class OpenAICompatibleProvider:
    def __init__(self,model=None,base_url=None,api_key=None,timeout=60):
        self.model=model or os.getenv("AGENT_MODEL","")
        self.base_url=(base_url or os.getenv("AGENT_BASE_URL","")).rstrip("/")
        self.api_key=api_key if api_key is not None else os.getenv("AGENT_API_KEY","")
        self.timeout=timeout
        if not self.model or not self.base_url:
            raise ValueError("Set AGENT_MODEL and AGENT_BASE_URL for an OpenAI-compatible server")
        if not self.base_url.startswith(("https://","http://localhost:","http://127.0.0.1:")):
            raise ValueError("Remote provider endpoints must use HTTPS")
    def complete(self,messages,tools):
        payload={"model":self.model,"messages":messages,"temperature":0}
        if tools: payload.update({"tools":tools,"tool_choice":"auto"})
        headers={"Content-Type":"application/json"}
        if self.api_key: headers["Authorization"]="Bearer "+self.api_key
        req=Request(self.base_url+"/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),headers=headers,method="POST")
        with urlopen(req,timeout=self.timeout) as response:
            obj=json.load(response)
        return obj["choices"][0]["message"]
