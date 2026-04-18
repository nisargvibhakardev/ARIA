from __future__ import annotations
import queue
import threading
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from process.stt import STTEngine
    from core.event_queue import EventQueue
    from capture.done_word import DoneWordDetector

from core.events import Event, EventType


class RollingTranscriber:
    def __init__(
        self,
        stt_engine,
        event_queue,
        done_detector,
        flush_callback: Callable[[], None],
        confidence_gate_logprob: float = -0.8,
        noise_speech_prob_max: float = 0.6,
    ) -> None:
        self._stt = stt_engine
        self._eq = event_queue
        self._done = done_detector
        self._flush_callback = flush_callback
        self._logprob_floor = confidence_gate_logprob
        self._noise_prob_max = noise_speech_prob_max

        self._chunk_queue: queue.Queue[bytes] = queue.Queue()
        self._lock = threading.Lock()
        self._rolling_text = ""
        self._thread: threading.Thread | None = None
        self._running = False

    @property
    def rolling_text(self) -> str:
        with self._lock:
            return self._rolling_text

    def push(self, audio_chunk: bytes) -> None:
        self._chunk_queue.put_nowait(audio_chunk)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._chunk_queue.put_nowait(b"")  # sentinel to unblock queue.get

    def reset(self) -> None:
        with self._lock:
            self._rolling_text = ""

    def _loop(self) -> None:
        while self._running:
            try:
                chunk = self._chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not chunk:
                break
            self._process_chunk(chunk)

    def _process_chunk(self, audio_chunk: bytes) -> None:
        with self._lock:
            prompt = self._rolling_text

        result = self._stt.transcribe(audio_chunk, initial_prompt=prompt)
        text = result.get("text", "").strip()

        # Confidence gate: reject noise chunks
        if (result.get("no_speech_prob", 0.0) > self._noise_prob_max or
                result.get("avg_logprob", 0.0) < self._logprob_floor):
            return

        if not text:
            return

        with self._lock:
            self._rolling_text = (self._rolling_text + " " + text).strip()
            current = self._rolling_text

        self._eq.put_nowait(Event(
            type=EventType.ROLLING_TRANSCRIPT,
            data={"partial_text": current, "confidence": result.get("avg_logprob", 0.0)},
        ))

        # Done-word check on newly appended text only (avoid re-detecting on old text).
        # Scan word-windows so "open the door pineapple" triggers on the "pineapple" window.
        if self._contains_done_word(text):
            self._flush_callback()

    def _contains_done_word(self, text: str) -> bool:
        """Return True if any word-window within text matches the done word."""
        words = text.split()
        window_size = self._done._word_count + 1  # ±1 for split variants
        for size in range(1, window_size + 1):
            for i in range(len(words) - size + 1):
                chunk = " ".join(words[i : i + size])
                if self._done.check(chunk):
                    return True
        return False
