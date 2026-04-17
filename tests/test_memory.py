import math
import time as time_mod
import pytest
from unittest.mock import MagicMock

from memory.structured import StructuredStore
from memory.knowledge_graph import KnowledgeGraph, NodeType, EdgeType
from memory.episodic import EpisodicMemory, ContextWindow
from memory.classifier import MemoryClassifier


# ─── StructuredStore ──────────────────────────────────────────────────────────

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


# ─── VectorStore + Ebbinghaus ─────────────────────────────────────────────────

def test_ebbinghaus_retention_fresh():
    from memory.vector import ebbinghaus_retention
    r = ebbinghaus_retention(elapsed_days=0.0, stability=1.0)
    assert r == pytest.approx(1.0, abs=0.01)


def test_ebbinghaus_retention_decays():
    from memory.vector import ebbinghaus_retention
    r_now = ebbinghaus_retention(elapsed_days=0.0, stability=1.0)
    r_later = ebbinghaus_retention(elapsed_days=2.0, stability=1.0)
    assert r_later < r_now


def test_ebbinghaus_higher_stability_slower_decay():
    from memory.vector import ebbinghaus_retention
    r_low = ebbinghaus_retention(elapsed_days=5.0, stability=1.0)
    r_high = ebbinghaus_retention(elapsed_days=5.0, stability=30.0)
    assert r_high > r_low


def test_vector_store_add_and_query(tmp_path):
    from memory.vector import VectorStore
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
    from memory.vector import VectorStore
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
    from memory.vector import VectorStore
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


# ─── KnowledgeGraph ───────────────────────────────────────────────────────────

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
    kg = KnowledgeGraph()
    now = time_mod.time()
    kg.add_node(NodeType.COMMITMENT, label="send invoice",
                attrs={"deadline_epoch": now + 3600, "fulfilled": False})
    kg.add_node(NodeType.COMMITMENT, label="book flights",
                attrs={"deadline_epoch": now + 86400, "fulfilled": False})
    at_risk = kg.get_at_risk_commitments(within_hours=2)
    assert len(at_risk) == 1
    assert at_risk[0]["label"] == "send invoice"


def test_contradicts_edge():
    kg = KnowledgeGraph()
    old = kg.add_node(NodeType.FACT, label="meeting is at 2pm")
    new_node = kg.add_node(NodeType.FACT, label="meeting is at 3pm")
    kg.add_edge(new_node, old, EdgeType.CONTRADICTS)
    edges = kg.get_edges(new_node)
    assert any(e["edge_type"] == EdgeType.CONTRADICTS for e in edges)


# ─── EpisodicMemory ───────────────────────────────────────────────────────────

def test_add_chunk_and_get_recent():
    mem = EpisodicMemory(window_seconds=120, summary_interval_minutes=30)
    mem.add_chunk("user opened VS Code and started editing main.py", source="screen")
    recent = mem.get_recent(seconds=60)
    assert len(recent) == 1
    assert "VS Code" in recent[0]["text"]


def test_get_recent_filters_old_chunks():
    mem = EpisodicMemory(window_seconds=1, summary_interval_minutes=30)
    mem.add_chunk("old chunk", source="screen")
    time_mod.sleep(1.1)
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


# ─── MemoryClassifier ─────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
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
