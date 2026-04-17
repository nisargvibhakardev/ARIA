from __future__ import annotations
import json
import time
from pathlib import Path

MONITOR_FILE = Path.home() / ".aria" / "monitor.json"


def _write(data: dict) -> None:
    MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if MONITOR_FILE.exists():
        with open(MONITOR_FILE) as f:
            try:
                current = json.load(f)
            except json.JSONDecodeError:
                pass
    current.update(data)
    current["_updated_at"] = time.time()
    with open(MONITOR_FILE, "w") as f:
        json.dump(current, f)


def update_capture(vad: str, diff_pct: float, ocr_loaded: bool) -> None:
    _write({"capture": {"vad": vad, "diff_pct": round(diff_pct, 3),
                         "ocr_loaded": ocr_loaded}})


def update_process(ocr_idle_countdown: float, whisper_loaded: bool,
                   last_text_preview: str) -> None:
    _write({"process": {"ocr_idle_countdown": round(ocr_idle_countdown, 1),
                         "whisper_loaded": whisper_loaded,
                         "last_text": last_text_preview[:80]}})


def update_memory(chroma_docs: int, kg_nodes: int, tasks: int,
                  commitments: int) -> None:
    _write({"memory": {"chroma_docs": chroma_docs, "kg_nodes": kg_nodes,
                        "tasks": tasks, "commitments": commitments}})


def update_decide(interval: int, generator_verdict: bool,
                  critic_verdict: bool, final_say: bool,
                  calibrator_conf: float, reason: str) -> None:
    _write({"decide": {
        "interval": interval,
        "generator": generator_verdict,
        "critic_suppress": critic_verdict,
        "final_say": final_say,
        "calibrator_conf": round(calibrator_conf, 2),
        "reason": reason,
    }})


def update_output(overlay_visible: bool, tts_state: str,
                  last_message: str) -> None:
    _write({"output": {"overlay_visible": overlay_visible,
                        "tts_state": tts_state,
                        "last_message": last_message}})


def record_interrupt(message: str, generator_reasoning: str,
                     critic_reasoning: str, suppressed: bool) -> None:
    entry = {
        "message": message,
        "generator": generator_reasoning,
        "critic": critic_reasoning,
        "suppressed": suppressed,
        "timestamp": time.time(),
    }
    current = {}
    if MONITOR_FILE.exists():
        with open(MONITOR_FILE) as f:
            try:
                current = json.load(f)
            except json.JSONDecodeError:
                pass
    replays = current.get("interrupt_replays", [])
    replays.append(entry)
    replays = replays[-10:]
    current["interrupt_replays"] = replays
    MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_FILE, "w") as f:
        json.dump(current, f)
