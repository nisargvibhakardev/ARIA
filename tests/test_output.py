import subprocess
import pytest
from unittest.mock import patch, MagicMock
from output.tts import TTS
from config import IdleUnloadConfig


@pytest.fixture
def tts():
    return TTS(IdleUnloadConfig(tts_minutes=2.0))


def test_tts_uses_aplay_by_default(tts):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch("output.tts.TTS._generate_wav", return_value=b"\xff\xfe" + b"\x00" * 98):
            tts.speak("hello world")
    calls = [str(c) for c in mock_run.call_args_list]
    assert any("aplay" in c for c in calls)


def test_tts_falls_back_to_paplay(tts):
    def run_side_effect(cmd, **kwargs):
        if cmd[0] == "aplay":
            raise FileNotFoundError("aplay not found")
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=run_side_effect):
        with patch("output.tts.TTS._generate_wav", return_value=b"\xff\xfe" + b"\x00" * 98):
            tts.speak("hello world")


def test_tts_idle_unloads_after_timeout():
    import time
    tts = TTS(IdleUnloadConfig(tts_minutes=0.0005))
    with patch("output.tts.TTS._generate_wav", return_value=b"\xff\xfe" + b"\x00" * 98):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            tts.speak("test")
    time.sleep(0.05)
    tts.check_idle()
    assert tts._model is None

from unittest.mock import MagicMock, patch
from hotkey import HotkeyListener
from core.events import EventType


def test_hotkey_listener_fires_event():
    from core.event_queue import EventQueue
    q = EventQueue()
    listener = HotkeyListener(queue=q, hotkey_str="ctrl+shift+space")
    fired_events = []

    original_put = q.put_nowait
    def capture(event):
        fired_events.append(event)
    q.put_nowait = capture

    listener._on_activate()
    assert len(fired_events) == 1
    assert fired_events[0].type == EventType.HOTKEY_PRESSED


def test_hotkey_listener_stop_does_not_raise():
    from core.event_queue import EventQueue
    q = EventQueue()
    listener = HotkeyListener(queue=q, hotkey_str="ctrl+shift+space")
    listener.stop()
