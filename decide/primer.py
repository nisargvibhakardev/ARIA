from __future__ import annotations
from typing import TYPE_CHECKING
from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from config import LLMConfig

_SYSTEM_PROMPT = (
    'You are ARIA, an assistant. Respond in English only. '
    'Reply with JSON only: {"say":true/false,"message":"max 20 words","importance":0.0-1.0,"reason":"5 words"}'
)


def _normalized_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return Levenshtein.distance(a, b) / max_len


class LLMPrimer:
    """Pre-warm Ollama KV cache with rolling transcript while final STT runs."""

    def __init__(self, config) -> None:
        self._config = config
        self._partial_text: str = ""
        self._stream = None

    def on_rolling_transcript(self, partial_text: str) -> None:
        if not self._config.primer_enabled:
            return
        if self._stream is not None:
            return  # already priming

        self._partial_text = partial_text
        import ollama
        client = ollama.Client(timeout=30)
        try:
            self._stream = client.chat(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f'The user just said: "{partial_text}"'},
                ],
                options={"temperature": 0.2, "num_predict": self._config.num_predict},
                keep_alive=self._config.keep_alive,
                stream=True,
            )
        except Exception:
            self._stream = None

    def on_speech_detected(self, final_text: str) -> object | None:
        """
        Called when final transcript is ready.
        Returns stream if divergence is low (caller drains it), else None (caller re-runs LLM).
        """
        if not self._config.primer_enabled or self._stream is None:
            self._stream = None
            return None

        divergence = _normalized_distance(self._partial_text, final_text)
        stream = self._stream
        self._stream = None
        self._partial_text = ""

        if divergence > self._config.primer_divergence_threshold:
            # Diverged too far — drain/discard the warm stream
            try:
                for _ in stream:
                    pass
            except Exception:
                pass
            return None

        return stream

    def reset(self) -> None:
        self._stream = None
        self._partial_text = ""
