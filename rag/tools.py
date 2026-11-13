from harness.contracts import Tool

def register_rag_tools(registry,store):
    registry.add(Tool("search_knowledge",
        "Search local knowledge-base passages. Cite the returned source_id values in your answer.",
        {"type":"object","properties":{"query":{"type":"string"},
            "k":{"type":"integer","minimum":1,"maximum":10}},
         "required":["query"],"additionalProperties":False},
        lambda query,k=4:store.query(query,max(1,min(int(k),10)))))
    return registry
