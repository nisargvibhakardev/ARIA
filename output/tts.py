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
        model_path = os.path.expanduser("~/.aria/piper/en_US-lessac-medium.onnx")
        self._model = PiperVoice.load(model_path)
        self._timer.reset()

    def _generate_wav(self, text: str) -> bytes:
        if self._model is None:
            self._load()
        self._timer.reset()
        import io
        buf = io.BytesIO()
        with self._model.output_to_wav_file(buf):
            self._model.synthesize(text)
        return buf.getvalue()

    def speak(self, text: str) -> None:
        wav_data = self._generate_wav(text)
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
