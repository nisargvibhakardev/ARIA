from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(Enum):
    PERSON = "person"
    TASK = "task"
    DEADLINE = "deadline"
    COMMITMENT = "commitment"
    FACT = "fact"
    OBSERVATION = "observation"
    ACTIVITY = "activity"


class EdgeType(Enum):
    DUE_BY = "due_by"
    ASSIGNED_TO = "assigned_to"
    FULFILLS = "fulfills"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"


@dataclass
class Node:
    id: str
    type: NodeType
    label: str
    attrs: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class Edge:
    source: str
    target: str
    edge_type: EdgeType
    created_at: float = field(default_factory=time.time)


class KnowledgeGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []

    def add_node(self, node_type: NodeType, label: str, attrs: dict | None = None) -> str:
        nid = str(uuid.uuid4())
        self._nodes[nid] = Node(id=nid, type=node_type, label=label, attrs=attrs or {})
        return nid

    def get_node(self, node_id: str) -> dict | None:
        node = self._nodes.get(node_id)
        if not node:
            return None
        return {"id": node.id, "type": node.type, "label": node.label,
                "attrs": node.attrs, "created_at": node.created_at}

    def add_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        self._edges.append(Edge(source=source, target=target, edge_type=edge_type))

    def get_edges(self, source: str) -> list[dict]:
        return [
            {"source": e.source, "target": e.target,
             "edge_type": e.edge_type, "created_at": e.created_at}
            for e in self._edges if e.source == source
        ]

    def subgraph(self, keywords: list[str]) -> list[dict]:
        results = []
        for node in self._nodes.values():
            if any(kw.lower() in node.label.lower() for kw in keywords):
                results.append({"id": node.id, "type": node.type,
                                 "label": node.label, "attrs": node.attrs})
        return results

    def get_at_risk_commitments(self, within_hours: float = 3.0) -> list[dict]:
        cutoff = time.time() + within_hours * 3600
        results = []
        for node in self._nodes.values():
            if node.type != NodeType.COMMITMENT:
                continue
            if node.attrs.get("fulfilled"):
                continue
            deadline = node.attrs.get("deadline_epoch")
            if deadline and deadline <= cutoff:
                results.append({"id": node.id, "label": node.label, "attrs": node.attrs})
        return results
