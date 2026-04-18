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
            self._whisper_config.model,
            device=self._whisper_config.device,
            compute_type=self._whisper_config.compute_type,
        )
        self._timer.reset()

    def transcribe(self, audio_bytes: bytes, initial_prompt: str = "") -> dict:
        if self._model is None:
            self._load()
        self._timer.reset()
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments_iter, info = self._model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=False,
            initial_prompt=initial_prompt if initial_prompt else None,
        )
        segments = list(segments_iter)
        text = "".join(s.text for s in segments).strip()
        avg_logprob = (
            sum(s.avg_logprob for s in segments) / len(segments) if segments else 0.0
        )
        no_speech_prob = max((s.no_speech_prob for s in segments), default=0.0)
        return {
            "text": text,
            "language": info.language,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
        }

    def check_idle(self) -> None:
        if self._model is not None and self._timer.is_expired():
            del self._model
            self._model = None
