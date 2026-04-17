from __future__ import annotations
import threading
import numpy as np

SAMPLE_RATE = 16000


class PushToTalk:
    def __init__(self) -> None:
        self.is_recording = False
        self._frames: list[bytes] = []
        self._stream = None
        self._lock = threading.Lock()

    def toggle(self) -> bytes | None:
        with self._lock:
            if self.is_recording:
                return self.stop()
            else:
                self._start()
                return None

    def _start(self) -> None:
        import sounddevice as sd  # lazy: PortAudio only needed at runtime
        self._frames = []
        self.is_recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            callback=self._callback
        )
        self._stream.start()

    def stop(self) -> bytes | None:
        if not self.is_recording:
            return None
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return b"".join(self._frames) if self._frames else None

    def _callback(self, indata, frames, time_info, status) -> None:
        self._frames.append(indata.copy().tobytes())
