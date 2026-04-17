from __future__ import annotations
import numpy as np
from config import IdleUnloadConfig, WhisperConfig
from process.loader import IdleTimer


class STTEngine:
    def __init__(self, idle_config: IdleUnloadConfig, whisper_config: WhisperConfig) -> None:
        self._idle_config = idle_config
        self._whisper_config = whisper_config
        self._model = None
        self._timer = IdleTimer(idle_config.whisper_minutes * 60)

    def _load(self):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            "small",
            device=self._whisper_config.device,
            compute_type=self._whisper_config.compute_type,
        )
        self._timer.reset()

    def transcribe(self, audio_bytes: bytes) -> dict:
        if self._model is None:
            self._load()
        self._timer.reset()
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        segments, info = self._model.transcribe(audio, beam_size=5)
        text = "".join(s.text for s in segments).strip()
        return {"text": text, "language": info.language}

    def check_idle(self) -> None:
        if self._model is not None and self._timer.is_expired():
            del self._model
            self._model = None
