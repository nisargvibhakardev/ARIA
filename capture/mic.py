from __future__ import annotations
import threading
from typing import TYPE_CHECKING
import numpy as np
import webrtcvad

from core.events import Event, EventType
if TYPE_CHECKING:
    from core.event_queue import EventQueue
    from config import MicConfig

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2


def is_speech_frame(vad: webrtcvad.Vad, frame: bytes, sample_rate: int) -> bool:
    return vad.is_speech(frame, sample_rate)


class MicWatcher:
    def __init__(self, queue: EventQueue, config: MicConfig) -> None:
        self._queue = queue
        self._config = config
        self._vad = webrtcvad.Vad(config.vad_aggressiveness)
        self._running = False
        self._thread: threading.Thread | None = None
        self._speech_buffer: list[bytes] = []
        self._in_speech = False
        self._silence_count = 0
        self._SILENCE_THRESHOLD = 10

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        import sounddevice as sd  # lazy: PortAudio only needed at runtime
        buffer = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        idx = 0

        def callback(indata, frames, time_info, status):
            nonlocal idx
            samples = indata[:, 0].astype(np.int16)
            for s in samples:
                buffer[idx] = s
                idx += 1
                if idx == FRAME_SAMPLES:
                    self._process_frame(buffer.tobytes())
                    idx = 0

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='int16', blocksize=FRAME_SAMPLES,
                            callback=callback):
            while self._running:
                sd.sleep(100)

    def _process_frame(self, frame: bytes) -> None:
        speech = is_speech_frame(self._vad, frame, SAMPLE_RATE)
        if speech:
            self._in_speech = True
            self._silence_count = 0
            self._speech_buffer.append(frame)
        elif self._in_speech:
            self._silence_count += 1
            if self._silence_count >= self._SILENCE_THRESHOLD:
                audio = b"".join(self._speech_buffer)
                self._queue.put_nowait(Event(
                    type=EventType.SPEECH_DETECTED,
                    data={"audio_bytes": audio, "sample_rate": SAMPLE_RATE}
                ))
                self._speech_buffer.clear()
                self._in_speech = False
                self._silence_count = 0
