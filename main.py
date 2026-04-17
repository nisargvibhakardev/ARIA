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
from decide.energy import EnergyScheduler
from decide.calibrator import InterruptCalibrator
from output.overlay import Overlay
from output.tts import TTS
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
        self._energy = EnergyScheduler(config.energy)
        self._calibrator = InterruptCalibrator(config.calibrator)

        self._ocr = OCREngine(config.idle_unload)
        self._stt = STTEngine(config.idle_unload, config.whisper)
        self._scene_parser = SceneParser()

        self._overlay: Overlay | None = None
        self._tts = TTS(config.idle_unload)

        self._screen_watcher = ScreenWatcher(self._queue, config.screen)
        self._mic_watcher = MicWatcher(self._queue, config.mic)
        self._hotkey = HotkeyListener(self._queue, config.hotkey)

        self._last_decision_at: float = 0.0
        self._running = False

    def set_overlay(self, overlay: Overlay) -> None:
        self._overlay = overlay

    def start_capture(self) -> None:
        self._screen_watcher.start()
        self._mic_watcher.start()
        self._hotkey.start()

    async def run(self) -> None:
        self._running = True
        while self._running:
            event = await self._queue.get()
            await self._handle_event(event)
            self._queue.task_done()

    async def _handle_event(self, event: Event) -> None:
        if event.type == EventType.SCREEN_CHANGED:
            await self._process_screen(event)
        elif event.type == EventType.SPEECH_DETECTED:
            await self._process_speech(event)
        elif event.type == EventType.HOTKEY_PRESSED:
            await self._handle_hotkey()

    async def _process_screen(self, event: Event) -> None:
        screenshot = event.data["screenshot"]
        window_title = event.data.get("window_title", "")
        text = self._ocr.extract(screenshot)
        if not text.strip():
            return
        scene = self._scene_parser.parse(window_title, text)
        self._episodic.add_chunk(text, source="screen")
        self._vector.add(text)

        now = time.monotonic()
        interval = self._energy.next_interval()
        if now - self._last_decision_at < interval:
            return
        self._last_decision_at = now

        recent = self._episodic.get_recent(seconds=120)
        result = self._decision_agent.evaluate({"recent": recent, "scene": scene})
        if result:
            await self._interrupt(result)

    async def _process_speech(self, event: Event) -> None:
        audio = event.data["audio_bytes"]
        transcript = self._stt.transcribe(audio)
        text = transcript.get("text", "").strip()
        if not text:
            return
        self._episodic.add_chunk(text, source="mic")
        self._vector.add(text)

    async def _handle_hotkey(self) -> None:
        recent = self._episodic.get_recent(seconds=120)
        result = self._decision_agent.evaluate({"recent": recent, "scene": None})
        if result:
            await self._interrupt(result)
        elif self._overlay:
            self._overlay.show_message(
                "Nothing urgent right now.", importance="low", reason="Hotkey query"
            )

    async def _interrupt(self, result: dict) -> None:
        if self._overlay:
            self._overlay.show_message(
                result["message"],
                importance="high" if result["importance"] > 0.7 else "low",
                reason=result["reason"],
            )
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._tts.speak, result["message"])

    def stop(self) -> None:
        self._running = False
        self._screen_watcher.stop()
        self._mic_watcher.stop()
        self._hotkey.stop()


def _run_pipeline(pipeline: ARIAPipeline) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(pipeline.run())


def main() -> None:
    _apply_nice(10)
    config = Config.from_yaml("config.yaml")

    app = QApplication(sys.argv)

    pipeline = ARIAPipeline(config)

    overlay = Overlay(config.overlay, on_engage=lambda: None)
    pipeline.set_overlay(overlay)

    pipeline_thread = threading.Thread(
        target=_run_pipeline, args=(pipeline,), daemon=True
    )
    pipeline_thread.start()
    pipeline.start_capture()

    exit_code = app.exec()
    pipeline.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
