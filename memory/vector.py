from __future__ import annotations
import math
import os
import time
import uuid
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction



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
        persist_dir = os.path.expanduser(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        self._purge_threshold = purge_threshold
        self._initial_stability = initial_stability_days
        self._important_stability = important_stability_days
        ef = DefaultEmbeddingFunction()
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
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_texts=[text],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
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
