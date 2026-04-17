from __future__ import annotations
import time
from dataclasses import dataclass, field


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
            self._chunks = [c for c in self._chunks if now - c.timestamp < self._window_seconds]
            self._last_summary_at = now

    def get_recent(self, seconds: int | None = None) -> list[dict]:
        cutoff = time.time() - (seconds or self._window_seconds)
        return [
            {"text": c.text, "source": c.source, "timestamp": c.timestamp}
            for c in self._chunks if c.timestamp >= cutoff
        ]

    def get_summaries(self, limit: int = 5) -> list[dict]:
        return self._summaries[-limit:]

    def extractive_summary(self, max_sentences: int = 5) -> str:
        if not self._chunks:
            return ""
        sentences = [c.text for c in self._chunks]
        seen: set[str] = set()
        unique = [s for s in sentences if not (s in seen or seen.add(s))]
        return " ".join(unique[:max_sentences])
