from __future__ import annotations
import threading
from typing import TYPE_CHECKING
import numpy as np
import webrtcvad

from core.events import Event, EventType
from capture.done_word import DoneWordDetector
from capture.eot_detector import EOTDetector
from capture.rolling_transcriber import RollingTranscriber

if TYPE_CHECKING:
    from core.event_queue import EventQueue
    from config import MicConfig
    from process.stt import STTEngine

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480
FRAME_BYTES = FRAME_SAMPLES * 2


def is_speech_frame(vad: webrtcvad.Vad, frame: bytes, sample_rate: int) -> bool:
    return vad.is_speech(frame, sample_rate)


class MicWatcher:
    def __init__(self, queue: EventQueue, config: MicConfig,
                 stt_engine=None) -> None:
        self._queue = queue
        self._config = config
        self._stt = stt_engine
        self._vad = webrtcvad.Vad(config.vad_aggressiveness)

        self._done_detector = DoneWordDetector(
            config.done_word, config.done_word_phoneme_tolerance
        )
        self._eot = EOTDetector(
            threshold=config.eot_probability_threshold,
            hard_cutoff_frames=config.eot_hard_cutoff_frames,
        )
        self._rolling_transcriber = RollingTranscriber(
            stt_engine=stt_engine,
            event_queue=queue,
            done_detector=self._done_detector,
            flush_callback=self._trigger_flush,
            confidence_gate_logprob=config.confidence_gate_logprob,
            noise_speech_prob_max=config.noise_speech_prob_max,
        )

        self._speech_buffer: list[bytes] = []
        self._buffer_lock = threading.Lock()
        self._in_speech = False
        self._frames_since_chunk = 0
        self._flush_event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._flush_thread: threading.Thread | None = None

    def set_stt_engine(self, stt_engine) -> None:
        self._stt = stt_engine
        self._rolling_transcriber._stt = stt_engine

    def _trigger_flush(self) -> None:
        self._flush_event.set()

    def start(self) -> None:
        self._running = True
        self._rolling_transcriber.start()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._flush_event.set()  # unblock flush loop
        self._rolling_transcriber.stop()

    def _flush_loop(self) -> None:
        while self._running:
            triggered = self._flush_event.wait(timeout=0.1)
            if not triggered:
                continue
            self._flush_event.clear()
            if not self._running:
                break
            self._do_flush()

    def _do_flush(self) -> None:
        with self._buffer_lock:
            if not self._speech_buffer:
                return
            audio = b"".join(self._speech_buffer)
            self._speech_buffer.clear()
            self._in_speech = False
            self._frames_since_chunk = 0
        self._eot.reset()
        rolling = self._rolling_transcriber.rolling_text
        self._rolling_transcriber.reset()

        if self._stt is None:
            return

        result = self._stt.transcribe(audio, initial_prompt=rolling)

        # Confidence gate on full audio
        if (result.get("no_speech_prob", 0.0) > self._config.noise_speech_prob_max or
                result.get("avg_logprob", 0.0) < self._config.confidence_gate_logprob):
            print(
                f"[mic] confidence gate: rejected "
                f"(logprob={result.get('avg_logprob', 0.0):.2f})",
                flush=True,
            )
            return

        text = result.get("text", "").strip()
        if not text:
            return

        # Strip done-word from final transcript
        text = self._done_detector.strip(text)
        if not text:
            return

        self._queue.put_nowait(Event(
            type=EventType.SPEECH_DETECTED,
            data={
                "audio_bytes": audio,
                "text": text,
                "confidence": result.get("avg_logprob", 0.0),
                "sample_rate": SAMPLE_RATE,
            },
        ))

    def _loop(self) -> None:
        import sounddevice as sd
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
                            dtype="int16", blocksize=FRAME_SAMPLES,
                            callback=callback):
            while self._running:
                sd.sleep(100)

    def _process_frame(self, frame: bytes) -> None:
        speech = is_speech_frame(self._vad, frame, SAMPLE_RATE)

        snapshot = None
        should_push_chunk = False
        was_in_speech = False

        with self._buffer_lock:
            if speech:
                self._in_speech = True
                self._speech_buffer.append(frame)
                self._frames_since_chunk += 1
                if self._frames_since_chunk >= self._config.chunk_frames:
                    snapshot = b"".join(self._speech_buffer)
                    self._frames_since_chunk = 0
                    should_push_chunk = True
            else:
                was_in_speech = self._in_speech
                if was_in_speech:
                    self._speech_buffer.append(frame)

        # All slow operations happen outside the lock
        if speech or was_in_speech:
            self._eot.update([frame])

        if should_push_chunk:
            self._rolling_transcriber.push(snapshot)
        elif was_in_speech and not speech:
            if self._eot.is_done() or self._eot.hard_cutoff_reached():
                self._trigger_flush()
