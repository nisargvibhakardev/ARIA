# MNEMON — Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the multi-tier memory system — ChromaDB vector store with Ebbinghaus forgetting, SQLite structured store, temporal knowledge graph, episodic summarizer, and memory classifier that routes LLM JSON output to the right store.

**Architecture:** Five independent Python classes behind a thin interface. `VectorStore` handles semantic search + Ebbinghaus decay. `StructuredStore` owns tasks/commitments/facts in SQLite. `KnowledgeGraph` is an in-memory temporal graph of entities + relations. `EpisodicMemory` compresses 30-min windows into summaries. `MemoryClassifier` routes LLM JSON to whichever store owns that data type.

**Tech Stack:** ChromaDB, `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`), SQLite3 (stdlib), Python `dataclasses`, `math.exp` (Ebbinghaus), `config.py`

**Worktree:** Work in `../aria-memory/`. Do not touch files outside your ownership list.

---

## File Map

| File | Responsibility |
|------|---------------|
| `memory/vector.py` | `VectorStore` — ChromaDB + Ebbinghaus decay + retrieval reinforcement |
| `memory/structured.py` | `StructuredStore` — SQLite tasks, commitments, key_facts tables |
| `memory/knowledge_graph.py` | `KnowledgeGraph` — temporal entity nodes + typed edges |
| `memory/episodic.py` | `EpisodicMemory` — 30-min window summaries |
| `memory/classifier.py` | `MemoryClassifier` — routes `{type, data}` JSON to correct store |
| `tests/test_memory.py` | All memory tests |

---

## Task 1: memory/structured.py — SQLite store

**Files:**
- Create: `memory/structured.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory.py
import pytest
import tempfile
import os
from memory.structured import StructuredStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = StructuredStore(db_path=db_path)
    yield s
    s.close()


def test_add_and_get_task(store):
    store.add_task("finish report", deadline="2026-04-20")
    tasks = store.get_tasks()
    assert len(tasks) == 1
    assert tasks[0]["text"] == "finish report"
    assert tasks[0]["status"] == "pending"


def test_task_status_update(store):
    store.add_task("write tests")
    tasks = store.get_tasks()
    task_id = tasks[0]["id"]
    store.update_task_status(task_id, "done")
    updated = store.get_tasks(status="done")
    assert len(updated) == 1


def test_add_commitment(store):
    store.add_commitment("send slides to Alice", to_whom="Alice")
    commitments = store.get_commitments()
    assert len(commitments) == 1
    assert commitments[0]["to_whom"] == "Alice"
    assert not commitments[0]["fulfilled"]


def test_fulfill_commitment(store):
    store.add_commitment("review PR", to_whom="Bob")
    cids = store.get_commitments()
    store.fulfill_commitment(cids[0]["id"])
    pending = store.get_commitments(fulfilled=False)
    assert len(pending) == 0


def test_add_key_fact(store):
    store.add_key_fact("user is left-handed")
    facts = store.get_key_facts()
    assert any(f["text"] == "user is left-handed" for f in facts)


def test_get_at_risk_commitments(store):
    import time
    store.add_commitment("urgent task", to_whom="team", deadline_epoch=time.time() + 3600)
    store.add_commitment("later task", to_whom="self", deadline_epoch=time.time() + 86400)
    at_risk = store.get_at_risk_commitments(within_hours=2)
    assert len(at_risk) == 1
    assert at_risk[0]["text"] == "urgent task"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_memory.py -v
```

Expected: `ImportError: cannot import name 'StructuredStore'`

- [ ] **Step 3: Write memory/structured.py**

```python
# memory/structured.py
from __future__ import annotations
import sqlite3
import time
from typing import Any


class StructuredStore:
    def __init__(self, db_path: str = "~/.aria/structured.db") -> None:
        import os
        db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                deadline TEXT,
                status TEXT DEFAULT 'pending',
                created_at REAL DEFAULT (unixepoch('now'))
            );
            CREATE TABLE IF NOT EXISTS commitments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                to_whom TEXT,
                fulfilled INTEGER DEFAULT 0,
                deadline_epoch REAL,
                created_at REAL DEFAULT (unixepoch('now'))
            );
            CREATE TABLE IF NOT EXISTS key_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE,
                created_at REAL DEFAULT (unixepoch('now'))
            );
        """)
        self._conn.commit()

    def add_task(self, text: str, deadline: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO tasks (text, deadline) VALUES (?, ?)", (text, deadline)
        )
        self._conn.commit()
        return cur.lastrowid

    def get_tasks(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM tasks").fetchall()
        return [dict(r) for r in rows]

    def update_task_status(self, task_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?", (status, task_id)
        )
        self._conn.commit()

    def add_commitment(
        self, text: str, to_whom: str | None = None, deadline_epoch: float | None = None
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO commitments (text, to_whom, deadline_epoch) VALUES (?, ?, ?)",
            (text, to_whom, deadline_epoch),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_commitments(self, fulfilled: bool | None = None) -> list[dict]:
        if fulfilled is None:
            rows = self._conn.execute("SELECT * FROM commitments").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM commitments WHERE fulfilled = ?", (int(fulfilled),)
            ).fetchall()
        return [dict(r) for r in rows]

    def fulfill_commitment(self, commitment_id: int) -> None:
        self._conn.execute(
            "UPDATE commitments SET fulfilled = 1 WHERE id = ?", (commitment_id,)
        )
        self._conn.commit()

    def add_key_fact(self, text: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO key_facts (text) VALUES (?)", (text,)
        )
        self._conn.commit()

    def get_key_facts(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM key_facts").fetchall()]

    def get_at_risk_commitments(self, within_hours: float = 3.0) -> list[dict]:
        cutoff = time.time() + within_hours * 3600
        rows = self._conn.execute(
            "SELECT * FROM commitments WHERE fulfilled = 0 AND deadline_epoch IS NOT NULL AND deadline_epoch <= ?",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_memory.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add memory/structured.py tests/test_memory.py
git commit -m "feat(memory): SQLite store online — tasks, commitments, facts, all persisted like that one embarrassing thing you said in 2019"
```

---

## Task 2: memory/vector.py — ChromaDB + Ebbinghaus

**Files:**
- Create: `memory/vector.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory.py`:

```python
import math
from memory.vector import VectorStore, ebbinghaus_retention


def test_ebbinghaus_retention_fresh():
    # R at t=0 should be 1.0 (just learned)
    r = ebbinghaus_retention(elapsed_days=0.0, stability=1.0)
    assert r == pytest.approx(1.0, abs=0.01)


def test_ebbinghaus_retention_decays():
    r_now = ebbinghaus_retention(elapsed_days=0.0, stability=1.0)
    r_later = ebbinghaus_retention(elapsed_days=2.0, stability=1.0)
    assert r_later < r_now


def test_ebbinghaus_higher_stability_slower_decay():
    r_low = ebbinghaus_retention(elapsed_days=5.0, stability=1.0)
    r_high = ebbinghaus_retention(elapsed_days=5.0, stability=30.0)
    assert r_high > r_low


def test_vector_store_add_and_query(tmp_path):
    store = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        purge_threshold=0.2,
        initial_stability_days=1.0,
        important_stability_days=30.0,
    )
    store.add("the user is working on a Python project", important=False)
    results = store.query("Python project", k=1)
    assert len(results) == 1
    assert "Python" in results[0]["text"]


def test_vector_store_query_returns_k_results(tmp_path):
    store = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        purge_threshold=0.2,
        initial_stability_days=1.0,
        important_stability_days=30.0,
    )
    for i in range(5):
        store.add(f"context chunk number {i}", important=False)
    results = store.query("context chunk", k=3)
    assert len(results) == 3


def test_vector_store_important_items_high_stability(tmp_path):
    store = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        purge_threshold=0.2,
        initial_stability_days=1.0,
        important_stability_days=30.0,
    )
    store.add("critical deadline: demo on Friday", important=True)
    results = store.query("demo deadline", k=1)
    meta = results[0]["metadata"]
    assert meta["stability"] == 30.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_memory.py -k "ebbinghaus or vector_store" -v
```

Expected: `ImportError: cannot import name 'VectorStore'`

- [ ] **Step 3: Write memory/vector.py**

```python
# memory/vector.py
from __future__ import annotations
import math
import time
import uuid
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def ebbinghaus_retention(elapsed_days: float, stability: float) -> float:
    return math.exp(-elapsed_days / stability)


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "~/.aria/chroma",
        purge_threshold: float = 0.2,
        initial_stability_days: float = 1.0,
        important_stability_days: float = 30.0,
    ) -> None:
        import os
        persist_dir = os.path.expanduser(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        self._purge_threshold = purge_threshold
        self._initial_stability = initial_stability_days
        self._important_stability = important_stability_days
        ef = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=persist_dir)
        self._collection = client.get_or_create_collection(
            name="context", embedding_function=ef
        )

    def add(self, text: str, important: bool = False) -> str:
        doc_id = str(uuid.uuid4())
        stability = self._important_stability if important else self._initial_stability
        self._collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[{
                "added_at": time.time(),
                "stability": stability,
                "important": int(important),
                "retrieval_count": 0,
            }],
        )
        return doc_id

    def query(self, text: str, k: int = 5) -> list[dict]:
        results = self._collection.query(
            query_texts=[text],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        # Reinforce stability on retrieval (Ebbinghaus retrieval effect)
        for i, meta in enumerate(metas):
            doc_id = results["ids"][0][i]
            new_stability = meta["stability"] * 1.2
            new_count = meta["retrieval_count"] + 1
            self._collection.update(
                ids=[doc_id],
                metadatas=[{**meta, "stability": new_stability, "retrieval_count": new_count}]
            )
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]

    def purge_forgotten(self) -> int:
        now = time.time()
        all_items = self._collection.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, meta in zip(all_items["ids"], all_items["metadatas"]):
            if meta.get("important"):
                continue
            elapsed_days = (now - meta["added_at"]) / 86400
            retention = ebbinghaus_retention(elapsed_days, meta["stability"])
            if retention < self._purge_threshold:
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_memory.py -k "ebbinghaus or vector_store" -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add memory/vector.py tests/test_memory.py
git commit -m "feat(memory): VectorStore + Ebbinghaus — AI now forgets things on a scientifically optimised schedule, unlike me with passwords"
```

---

## Task 3: memory/knowledge_graph.py — Temporal entity graph

**Files:**
- Create: `memory/knowledge_graph.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory.py`:

```python
from memory.knowledge_graph import KnowledgeGraph, NodeType, EdgeType


def test_add_and_get_node():
    kg = KnowledgeGraph()
    nid = kg.add_node(NodeType.TASK, label="submit report", attrs={"deadline": "Friday"})
    node = kg.get_node(nid)
    assert node["label"] == "submit report"
    assert node["type"] == NodeType.TASK


def test_add_edge_between_nodes():
    kg = KnowledgeGraph()
    task = kg.add_node(NodeType.TASK, label="prepare slides")
    person = kg.add_node(NodeType.PERSON, label="Alice")
    kg.add_edge(task, person, EdgeType.ASSIGNED_TO)
    edges = kg.get_edges(task)
    assert len(edges) == 1
    assert edges[0]["edge_type"] == EdgeType.ASSIGNED_TO
    assert edges[0]["target"] == person


def test_subgraph_for_entities():
    kg = KnowledgeGraph()
    kg.add_node(NodeType.FACT, label="project deadline is next Monday")
    kg.add_node(NodeType.TASK, label="write unit tests")
    kg.add_node(NodeType.PERSON, label="Bob")
    result = kg.subgraph(["deadline", "Monday"])
    assert any("deadline" in n["label"] for n in result)


def test_at_risk_commitments():
    import time
    kg = KnowledgeGraph()
    now = time.time()
    cid = kg.add_node(NodeType.COMMITMENT, label="send invoice",
                      attrs={"deadline_epoch": now + 3600, "fulfilled": False})
    kg.add_node(NodeType.COMMITMENT, label="book flights",
                attrs={"deadline_epoch": now + 86400, "fulfilled": False})
    at_risk = kg.get_at_risk_commitments(within_hours=2)
    assert len(at_risk) == 1
    assert at_risk[0]["label"] == "send invoice"


def test_contradicts_edge():
    kg = KnowledgeGraph()
    old = kg.add_node(NodeType.FACT, label="meeting is at 2pm")
    new = kg.add_node(NodeType.FACT, label="meeting is at 3pm")
    kg.add_edge(new, old, EdgeType.CONTRADICTS)
    edges = kg.get_edges(new)
    assert any(e["edge_type"] == EdgeType.CONTRADICTS for e in edges)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_memory.py -k "knowledge_graph or node or subgraph or contradicts or at_risk_commit" -v
```

Expected: `ImportError: cannot import name 'KnowledgeGraph'`

- [ ] **Step 3: Write memory/knowledge_graph.py**

```python
# memory/knowledge_graph.py
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_memory.py -k "knowledge_graph or node or subgraph or contradicts or at_risk_commit" -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add memory/knowledge_graph.py tests/test_memory.py
git commit -m "feat(memory): KnowledgeGraph alive — it knows who owes whom what, and it never forgets. HR is terrified."
```

---

## Task 4: memory/episodic.py — 30-min session summaries

**Files:**
- Create: `memory/episodic.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory.py`:

```python
import time
from memory.episodic import EpisodicMemory, ContextWindow


def test_add_chunk_and_get_recent():
    mem = EpisodicMemory(window_seconds=120, summary_interval_minutes=30)
    mem.add_chunk("user opened VS Code and started editing main.py", source="screen")
    recent = mem.get_recent(seconds=60)
    assert len(recent) == 1
    assert "VS Code" in recent[0]["text"]


def test_get_recent_filters_old_chunks():
    mem = EpisodicMemory(window_seconds=1, summary_interval_minutes=30)
    mem.add_chunk("old chunk", source="screen")
    time.sleep(1.1)
    mem.add_chunk("new chunk", source="mic")
    recent = mem.get_recent(seconds=1)
    assert len(recent) == 1
    assert "new" in recent[0]["text"]


def test_context_window_extractive_summary():
    mem = EpisodicMemory(window_seconds=120, summary_interval_minutes=30)
    chunks = [
        "user is writing Python code",
        "VS Code is the active app",
        "user mentioned deadline on Friday",
    ]
    for c in chunks:
        mem.add_chunk(c, source="screen")
    summary = mem.extractive_summary(max_sentences=2)
    assert isinstance(summary, str)
    assert len(summary) > 0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_memory.py -k "episodic or context_window" -v
```

Expected: `ImportError: cannot import name 'EpisodicMemory'`

- [ ] **Step 3: Write memory/episodic.py**

```python
# memory/episodic.py
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextWindow:
    text: str
    source: str
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    def __init__(self, window_seconds: int = 120, summary_interval_minutes: int = 30) -> None:
        self._window_seconds = window_seconds
        self._summary_interval = summary_interval_minutes * 60
        self._chunks: list[ContextWindow] = []
        self._summaries: list[dict] = []
        self._last_summary_at: float = time.time()

    def add_chunk(self, text: str, source: str = "screen") -> None:
        self._chunks.append(ContextWindow(text=text, source=source))
        now = time.time()
        if now - self._last_summary_at >= self._summary_interval:
            summary = self.extractive_summary()
            self._summaries.append({"text": summary, "created_at": now})
            self._chunks = [c for c in self._chunks
                            if now - c.timestamp < self._window_seconds]
            self._last_summary_at = now

    def get_recent(self, seconds: int | None = None) -> list[dict]:
        cutoff = time.time() - (seconds or self._window_seconds)
        return [
            {"text": c.text, "source": c.source, "timestamp": c.timestamp}
            for c in self._chunks
            if c.timestamp >= cutoff
        ]

    def get_summaries(self, limit: int = 5) -> list[dict]:
        return self._summaries[-limit:]

    def extractive_summary(self, max_sentences: int = 5) -> str:
        if not self._chunks:
            return ""
        sentences = [c.text for c in self._chunks]
        # Deduplicate preserving order
        seen: set[str] = set()
        unique = [s for s in sentences if not (s in seen or seen.add(s))]
        return " ".join(unique[:max_sentences])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_memory.py -k "episodic or context_window" -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add memory/episodic.py tests/test_memory.py
git commit -m "feat(memory): EpisodicMemory summarises your last 30 minutes — a harsh but fair mirror of your productivity"
```

---

## Task 5: memory/classifier.py — LLM output router

**Files:**
- Create: `memory/classifier.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory.py`:

```python
import tempfile
import pytest
from unittest.mock import MagicMock
from memory.classifier import MemoryClassifier


@pytest.fixture
def classifier(tmp_path):
    structured = MagicMock()
    vector = MagicMock()
    kg = MagicMock()
    return MemoryClassifier(structured=structured, vector=vector, kg=kg), structured, vector, kg


def test_classifier_routes_task(classifier):
    clf, structured, vector, kg = classifier
    clf.route({"type": "task", "text": "write the report", "deadline": "Friday"})
    structured.add_task.assert_called_once_with("write the report", deadline="Friday")
    vector.add.assert_called_once()


def test_classifier_routes_commitment(classifier):
    clf, structured, vector, kg = classifier
    clf.route({"type": "commitment", "text": "send slides to Alice", "to_whom": "Alice"})
    structured.add_commitment.assert_called_once_with("send slides to Alice", to_whom="Alice")


def test_classifier_routes_fact(classifier):
    clf, structured, vector, kg = classifier
    clf.route({"type": "fact", "text": "user prefers dark mode"})
    structured.add_key_fact.assert_called_once_with("user prefers dark mode")
    vector.add.assert_called_once()


def test_classifier_routes_context_chunk(classifier):
    clf, structured, vector, kg = classifier
    clf.route({"type": "context", "text": "user is in a meeting"})
    vector.add.assert_called_once_with("user is in a meeting", important=False)
    structured.add_task.assert_not_called()


def test_classifier_routes_contradiction(classifier):
    clf, structured, vector, kg = classifier
    clf.route({
        "type": "contradiction",
        "text": "meeting moved to 3pm",
        "contradicts": "meeting is at 2pm"
    })
    assert kg.add_node.call_count == 2
    kg.add_edge.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_memory.py -k "classifier" -v
```

Expected: `ImportError: cannot import name 'MemoryClassifier'`

- [ ] **Step 3: Write memory/classifier.py**

```python
# memory/classifier.py
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory.structured import StructuredStore
    from memory.vector import VectorStore
    from memory.knowledge_graph import KnowledgeGraph, NodeType, EdgeType


class MemoryClassifier:
    def __init__(
        self,
        structured: StructuredStore,
        vector: VectorStore,
        kg: KnowledgeGraph,
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
            self._structured.add_commitment(
                text, to_whom=llm_json.get("to_whom"),
            )

        elif t == "fact":
            self._structured.add_key_fact(text)
            self._vector.add(text, important=True)

        elif t == "contradiction":
            from memory.knowledge_graph import NodeType, EdgeType
            new_id = self._kg.add_node(NodeType.FACT, label=text)
            old_id = self._kg.add_node(NodeType.FACT, label=llm_json.get("contradicts", ""))
            self._kg.add_edge(new_id, old_id, EdgeType.CONTRADICTS)

        else:  # "context" — default
            self._vector.add(text, important=False)
```

- [ ] **Step 4: Run full memory test suite**

```bash
pytest tests/test_memory.py -v --tb=short
```

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add memory/classifier.py tests/test_memory.py
git commit -m "feat(memory): MemoryClassifier routes LLM output like a very opinionated post office — tasks here, facts there, chaos filed under 'context'"
```

---

## Task 6: Update build/status.json + push

- [ ] **Step 1: Update build/status.json mnemon entry**

```json
"mnemon": {
  "state": "active",
  "task": "All memory modules complete. Requesting CIPHER review + alerting NEXUS to fire ORACLE.",
  "progress": 100,
  "last_update": "<ISO timestamp>",
  "log": [
    "structured.py complete — tasks, commitments, key_facts, at_risk",
    "vector.py complete — ChromaDB + Ebbinghaus decay + retrieval reinforcement",
    "knowledge_graph.py complete — temporal entity graph, contradicts edges",
    "episodic.py complete — 30min summaries, extractive fallback",
    "classifier.py complete — routes task/commitment/fact/contradiction/context",
    "All tests passing"
  ]
}
```

- [ ] **Step 2: Request ARIA password for push**

Present to Nisarg:
```
GIT OPERATION REQUEST
Operation: git push origin feat/memory
Why: MNEMON complete — all memory modules with passing tests. Also: ORACLE can now start.
```

- [ ] **Step 3: Push (after ARIA received)**

```bash
git checkout -b feat/memory
git push origin feat/memory
```

---

## Self-Review

**Spec coverage:**
- ✅ `memory/structured.py` — tasks, commitments, key_facts, at_risk → Task 1
- ✅ `memory/vector.py` — ChromaDB, Ebbinghaus, retrieval reinforcement → Task 2
- ✅ `memory/knowledge_graph.py` — nodes, edges, subgraph, contradicts → Task 3
- ✅ `memory/episodic.py` — 30-min summaries, extractive fallback → Task 4
- ✅ `memory/classifier.py` — routes all LLM JSON types → Task 5
- ✅ ORACLE trigger notification via status.json → Task 6

**Type consistency:**
- `StructuredStore.add_task(text, deadline=)` consistent in classifier and tests ✓
- `VectorStore.add(text, important=bool)` consistent across classifier and vector tests ✓
- `KnowledgeGraph.add_node(NodeType.X, label=, attrs=)` consistent ✓
- `MemoryClassifier.route(dict)` input shape consistent across all route tests ✓
