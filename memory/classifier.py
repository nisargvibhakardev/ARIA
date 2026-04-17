from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory.structured import StructuredStore
    from memory.vector import VectorStore
    from memory.knowledge_graph import KnowledgeGraph


class MemoryClassifier:
    def __init__(
        self,
        structured: "StructuredStore",
        vector: "VectorStore",
        kg: "KnowledgeGraph",
    ) -> None:
        self._structured = structured
        self._vector = vector
        self._kg = kg

    def route(self, llm_json: dict[str, Any]) -> None:
        t = llm_json.get("type", "context")
        text = llm_json.get("text", "")

        if t == "task":
            self._structured.add_task(text, deadline=llm_json.get("deadline"))
            self._vector.add(text, important=True)
        elif t == "commitment":
            self._structured.add_commitment(text, to_whom=llm_json.get("to_whom"))
        elif t == "fact":
            self._structured.add_key_fact(text)
            self._vector.add(text, important=True)
        elif t == "contradiction":
            from memory.knowledge_graph import NodeType, EdgeType
            new_id = self._kg.add_node(NodeType.FACT, label=text)
            old_id = self._kg.add_node(NodeType.FACT, label=llm_json.get("contradicts", ""))
            self._kg.add_edge(new_id, old_id, EdgeType.CONTRADICTS)
        else:
            self._vector.add(text, important=False)
