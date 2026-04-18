from __future__ import annotations
import asyncio
import os
import sys
import threading
import time

from PyQt6.QtWidgets import QApplication

from config import Config
from core.event_queue import EventQueue
from core.events import Event, EventType

from capture.screen import ScreenWatcher
from capture.mic import MicWatcher
from process.ocr import OCREngine
from process.stt import STTEngine
from process.scene import SceneParser
from memory.vector import VectorStore
from memory.structured import StructuredStore
from memory.knowledge_graph import KnowledgeGraph
from memory.episodic import EpisodicMemory
from memory.classifier import MemoryClassifier
from decide.agent import DecisionAgent
from decide.primer import LLMPrimer
from decide.energy import EnergyScheduler
from decide.calibrator import InterruptCalibrator
from output.overlay import Overlay
from output.tts import TTS
from output import monitor_writer
from hotkey import HotkeyListener


def _apply_nice(level: int) -> None:
    try:
        os.nice(level)
    except (PermissionError, AttributeError):
        pass


class ARIAPipeline:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._queue = EventQueue()

        self._vector = VectorStore(
            purge_threshold=config.memory.ebbinghaus_purge_threshold,
            initial_stability_days=config.memory.initial_stability_days,
            important_stability_days=config.memory.important_stability_days,
        )
        self._structured = StructuredStore()
        self._kg = KnowledgeGraph()
        self._episodic = EpisodicMemory(
            window_seconds=config.memory.context_window_seconds,
            summary_interval_minutes=config.episodic.summary_interval_minutes,
        )

        class _Memory:
            pass
        mem = _Memory()
        mem.vector = self._vector
        mem.structured = self._structured
        mem.kg = self._kg
        mem.episodic = self._episodic

        self._classifier = MemoryClassifier(self._structured, self._vector, self._kg)
        self._decision_agent = DecisionAgent(memory=mem, config=config.llm)
        self._primer = LLMPrimer(config.llm)
        self._energy = EnergyScheduler(config.energy)
        self._calibrator = InterruptCalibrator(config.calibrator)

        self._ocr = OCREngine(config.idle_unload)
        self._stt = STTEngine(config.idle_unload, config.whisper)
        self._scene_parser = SceneParser()

        self._overlay: Overlay | None = None
        self._tts = TTS(config.idle_unload)

        self._screen_watcher = ScreenWatcher(self._queue, config.screen)
        self._mic_watcher = MicWatcher(self._queue, config.mic, stt_engine=self._stt)
        self._hotkey = HotkeyListener(self._queue, config.hotkey)

        self._last_decision_at: float = 0.0
        self._running = False
        self._hotkey_lock: asyncio.Lock | None = None

    def set_overlay(self, overlay: Overlay) -> None:
        self._overlay = overlay

    def start_capture(self) -> None:
        if self._config.screen.enabled:
            self._screen_watcher.start()
        else:
            print("[ARIA] screen capture disabled — OCR off", flush=True)
        self._mic_watcher.start()
        self._hotkey.start()

    async def run(self) -> None:
        print("[ARIA] pipeline run() entered — waiting for events", flush=True)
        self._hotkey_lock = asyncio.Lock()
        self._running = True
        while self._running:
            event = await self._queue.get()
            asyncio.create_task(self._handle_event(event))
            self._queue.task_done()

    async def _handle_event(self, event: Event) -> None:
        print(f"[ARIA] handling event: {event.type}", flush=True)
        if event.type == EventType.SCREEN_CHANGED:
            await self._process_screen(event)
        elif event.type == EventType.SPEECH_DETECTED:
            await self._process_speech(event)
        elif event.type == EventType.HOTKEY_PRESSED:
            await self._handle_hotkey()
        elif event.type == EventType.ROLLING_TRANSCRIPT:
            await self._process_rolling_transcript(event)

    async def _process_rolling_transcript(self, event: Event) -> None:
        partial = event.data.get("partial_text", "")
        if self._overlay:
            self._overlay.show_partial(partial)
        self._primer.on_rolling_transcript(partial)

    async def _process_screen(self, event: Event) -> None:
        screenshot = event.data["screenshot"]
        window_title = event.data.get("window_title", "")
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._ocr.extract, screenshot)
        monitor_writer.update_capture(vad="silence", diff_pct=0.0, ocr_loaded=True)
        monitor_writer.update_process(
            ocr_idle_countdown=self._ocr._timer._timeout - (time.monotonic() - self._ocr._timer._last_reset),
            whisper_loaded=self._stt._model is not None,
            last_text_preview=text[:80],
        )
        if not text.strip():
            return
        print(f"[ARIA] OCR [{window_title[:40] or 'unknown'}]: \"{text[:120].strip()}\"", flush=True)
        scene = self._scene_parser.parse(window_title, text)
        self._episodic.add_chunk(text, source="screen")
        self._vector.add(text)
        monitor_writer.update_memory(
            chroma_docs=self._vector._collection.count(),
            kg_nodes=len(self._kg._nodes),
            tasks=len(self._structured.get_tasks()),
            commitments=len(self._structured.get_commitments()),
        )

        now = time.monotonic()
        interval = self._energy.next_interval()
        if now - self._last_decision_at < interval:
            return
        self._last_decision_at = now

        recent = self._episodic.get_recent(seconds=120)
        result = await loop.run_in_executor(
            None, self._decision_agent.evaluate, {"recent": recent, "scene": scene}
        )
        monitor_writer.update_decide(
            interval=self._energy.next_interval(),
            generator_verdict=result is not None,
            critic_verdict=False,
            final_say=result is not None,
            calibrator_conf=0.0,
            reason=result["reason"] if result else "no interrupt",
        )
        if result:
            await self._interrupt(result)

    async def _process_speech(self, event: Event) -> None:
        audio = event.data["audio_bytes"]
        audio_ms = round(len(audio) / 32)  # 16kHz 16-bit = 32 bytes/ms
        print(f"[ARIA] STT: transcribing {audio_ms}ms of audio...", flush=True)
        loop = asyncio.get_event_loop()
        pre_text = event.data.get("text", "")
        if pre_text:
            text = pre_text
            print(f"[ARIA] STT: using pre-transcribed text ({audio_ms}ms): \"{text}\"", flush=True)
        else:
            transcript = await loop.run_in_executor(None, self._stt.transcribe, audio)
            text = transcript.get("text", "").strip()
            if not text:
                print("[ARIA] STT: no speech detected in audio chunk", flush=True)
                return
            print(f"[ARIA] STT: \"{text}\"", flush=True)

        # Notify primer that final text is ready
        primer_stream = self._primer.on_speech_detected(text)
        if primer_stream is not None:
            # Warm stream available but DecisionAgent doesn't consume streams yet — drain it
            try:
                for _ in primer_stream:
                    pass
            except Exception:
                pass

        self._episodic.add_chunk(text, source="mic")
        self._vector.add(text)

        # Respond to what was said
        if self._overlay:
            self._overlay.show_thinking()
        recent = self._episodic.get_recent(seconds=120)
        result = await loop.run_in_executor(
            None, self._decision_agent.evaluate, {"recent": recent, "scene": None, "source": "speech"}
        )
        if result:
            print(f"[ARIA] speech response: \"{result['message']}\"", flush=True)
            await self._interrupt(result)
        else:
            if self._overlay:
                self._overlay.hide_message()

    async def _handle_hotkey(self) -> None:
        if self._hotkey_lock is None or self._hotkey_lock.locked():
            print("[ARIA] hotkey ignored — LLM already running", flush=True)
            return
        async with self._hotkey_lock:
            print("[ARIA] hotkey received — showing thinking overlay", flush=True)
            if self._overlay:
                self._overlay.show_thinking()
            recent = self._episodic.get_recent(seconds=120)
            recent_texts = [c.get("text", "")[:80] for c in recent[-5:]]
            print(f"[ARIA] context window ({len(recent)} chunks):", flush=True)
            for i, t in enumerate(recent_texts):
                print(f"  [{i+1}] {t!r}", flush=True)
            print("[ARIA] calling LLM...", flush=True)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._decision_agent.evaluate, {"recent": recent, "scene": None}
            )
            if result:
                print(f"[ARIA] response: \"{result['message']}\" (importance={result['importance']:.2f})", flush=True)
                await self._interrupt(result)
            else:
                print("[ARIA] response: say=false — showing idle message", flush=True)
                if self._overlay:
                    self._overlay.show_message("Nothing urgent right now.", "low", "Hotkey query")

    async def _interrupt(self, result: dict) -> None:
        if self._overlay:
            self._overlay.show_message(
                result["message"],
                "high" if result["importance"] > 0.7 else "low",
                result["reason"],
            )
        monitor_writer.update_output(
            overlay_visible=True,
            tts_state="speaking",
            last_message=result["message"],
        )
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._tts.speak, result["message"])

    def stop(self) -> None:
        self._running = False
        self._screen_watcher.stop()
        self._mic_watcher.stop()
        self._hotkey.stop()


def _run_pipeline(pipeline: ARIAPipeline, loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        print("[ARIA] pipeline loop starting", flush=True)
        loop.run_until_complete(pipeline.run())
    except Exception as e:
        import traceback
        print(f"[ARIA] pipeline CRASHED: {e}", flush=True)
        traceback.print_exc()


def main() -> None:
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _apply_nice(10)
    config = Config.from_yaml("config.yaml")

    app = QApplication(sys.argv)

    pipeline = ARIAPipeline(config)

    overlay = Overlay(config.overlay, on_engage=lambda: None)
    pipeline.set_overlay(overlay)

    loop = asyncio.new_event_loop()

    pipeline_thread = threading.Thread(
        target=_run_pipeline, args=(pipeline, loop), daemon=True
    )
    pipeline_thread.start()

    # Pre-warm the model so first hotkey press isn't a cold load
    def _warm():
        import ollama
        try:
            ollama.chat(model=config.llm.model,
                        messages=[{"role": "user", "content": "hi"}],
                        options={"num_predict": 1},
                        keep_alive=config.llm.keep_alive)
            print("[ARIA] model warmed up", flush=True)
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()

    pipeline.start_capture()

    exit_code = app.exec()
    pipeline.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
