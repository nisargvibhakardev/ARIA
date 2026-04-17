from __future__ import annotations
from config import EnergyConfig


class EnergyScheduler:
    def __init__(self, config: EnergyConfig) -> None:
        self._config = config
        self._avg_novelty: float = 0.0
        self._focus: float = 0.5

    def update(self, novelty_score: float, focus_level: float) -> None:
        self._avg_novelty = 0.7 * self._avg_novelty + 0.3 * novelty_score
        self._focus = focus_level

    def next_interval(self) -> int:
        raw = self._config.max_interval_seconds - self._avg_novelty * 100.0
        raw = max(self._config.min_interval_seconds, raw)
        dampened = raw + (self._config.max_interval_seconds - raw) * self._focus * 0.5
        return int(min(self._config.max_interval_seconds,
                       max(self._config.min_interval_seconds, dampened)))
