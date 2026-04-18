from __future__ import annotations
import subprocess
import tempfile
import os
from config import IdleUnloadConfig
from process.loader import IdleTimer


class TTS:
    def __init__(self, config: IdleUnloadConfig) -> None:
        self._config = config
        self._model = None
        self._timer = IdleTimer(config.tts_minutes * 60)

    def _load(self):
        from piper import PiperVoice
        self._model = PiperVoice.load(self._model_path)
        self._timer.reset()

    def _generate_wav(self, text: str) -> bytes:
        if self._model is None:
            self._load()
        self._timer.reset()
        import io
        import wave
        chunks = list(self._model.synthesize(text))
        if not chunks:
            return b""
        sample_rate = chunks[0].sample_rate
        sample_width = chunks[0].sample_width
        channels = chunks[0].sample_channels
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return buf.getvalue()

    @property
    def _model_path(self) -> str:
        return os.path.expanduser("~/.aria/piper/en_US-lessac-medium.onnx")

    @property
    def available(self) -> bool:
        return os.path.exists(self._model_path)

    def speak(self, text: str) -> None:
        if not self.available:
            return  # no voice model installed — silent, no error
        try:
            wav_data = self._generate_wav(text)
        except Exception:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            tmp_path = f.name
        try:
            self._play(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _play(self, wav_path: str) -> None:
        try:
            subprocess.run(["aplay", "-q", wav_path], check=True, timeout=30)
        except (FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run(["paplay", wav_path], check=True, timeout=30)

    def check_idle(self) -> None:
        if self._model is not None and self._timer.is_expired():
            del self._model
            self._model = None
