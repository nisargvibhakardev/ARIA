import hashlib
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from capture.screen import ScreenWatcher, pixel_hash, diff_ratio


def test_pixel_hash_returns_string():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pixel_hash(frame)
    assert isinstance(result, str)
    assert len(result) == 64  # sha256 hex


def test_pixel_hash_same_frame_same_hash():
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
    assert pixel_hash(frame) == pixel_hash(frame.copy())


def test_pixel_hash_different_frames_different_hash():
    frame_a = np.zeros((50, 50, 3), dtype=np.uint8)
    frame_b = np.ones((50, 50, 3), dtype=np.uint8) * 255
    assert pixel_hash(frame_a) != pixel_hash(frame_b)


def test_diff_ratio_identical_frames():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert diff_ratio(frame, frame.copy()) == 0.0


def test_diff_ratio_completely_different():
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = np.ones((100, 100, 3), dtype=np.uint8) * 255
    ratio = diff_ratio(frame_a, frame_b)
    assert ratio > 0.9


def test_diff_ratio_within_threshold():
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = frame_a.copy()
    frame_b[0, 0] = [255, 255, 255]
    ratio = diff_ratio(frame_a, frame_b)
    assert ratio < 0.15


from capture.mic import MicWatcher, is_speech_frame


def test_is_speech_frame_silent():
    vad = MagicMock()
    vad.is_speech.return_value = False
    silent_audio = bytes(960)
    assert is_speech_frame(vad, silent_audio, 16000) is False


def test_is_speech_frame_speech():
    vad = MagicMock()
    vad.is_speech.return_value = True
    audio = bytes(960)
    assert is_speech_frame(vad, audio, 16000) is True


def test_mic_watcher_initialises():
    from core.event_queue import EventQueue
    from config import MicConfig
    import asyncio
    loop = asyncio.new_event_loop()
    q = EventQueue()
    cfg = MicConfig(vad_aggressiveness=2)
    watcher = MicWatcher(q, cfg)
    assert watcher is not None
    loop.close()


from process.loader import IdleTimer, ComponentLoader


def test_idle_timer_not_expired_immediately():
    timer = IdleTimer(timeout_seconds=60)
    timer.reset()
    assert not timer.is_expired()


def test_idle_timer_expires():
    timer = IdleTimer(timeout_seconds=0.05)
    timer.reset()
    time.sleep(0.1)
    assert timer.is_expired()


def test_idle_timer_reset_restarts_countdown():
    timer = IdleTimer(timeout_seconds=0.05)
    timer.reset()
    time.sleep(0.03)
    timer.reset()
    time.sleep(0.03)
    assert not timer.is_expired()


def test_component_loader_loads_on_get():
    loaded = []
    loader = ComponentLoader(
        load_fn=lambda: loaded.append(1) or "component",
        unload_fn=lambda c: None,
        timeout_seconds=60,
    )
    result = loader.get()
    assert result == "component"
    assert loaded == [1]


def test_component_loader_reuses_loaded_component():
    calls = []
    loader = ComponentLoader(
        load_fn=lambda: calls.append(1) or "comp",
        unload_fn=lambda c: None,
        timeout_seconds=60,
    )
    loader.get()
    loader.get()
    assert len(calls) == 1


def test_component_loader_unloads_on_expiry():
    unloaded = []
    loader = ComponentLoader(
        load_fn=lambda: "comp",
        unload_fn=lambda c: unloaded.append(c),
        timeout_seconds=0.05,
    )
    loader.get()
    time.sleep(0.1)
    loader.check_idle()
    assert unloaded == ["comp"]


from process.scene import SceneParser, SceneContext, detect_app, detect_task_type


def test_detect_app_vscode():
    assert detect_app("Visual Studio Code — main.py") == "vscode"


def test_detect_app_chrome():
    assert detect_app("Google Chrome — GitHub") == "chrome"


def test_detect_app_slack():
    assert detect_app("Slack — #general") == "slack"


def test_detect_app_terminal():
    assert detect_app("Terminal — bash") == "terminal"


def test_detect_app_unknown():
    assert detect_app("Some Random App") == "unknown"


def test_detect_task_type_coding():
    assert detect_task_type("vscode", "def test_foo(): ...") == "coding"


def test_detect_task_type_browsing():
    assert detect_task_type("chrome", "Wikipedia article") == "browsing"


def test_detect_task_type_communicating():
    assert detect_task_type("slack", "hey can you review this") == "communicating"


def test_scene_context_has_expected_fields():
    ctx = SceneContext(
        app="vscode", task_type="coding",
        entities=["main.py", "EventQueue"],
        focus_level=0.8, delta={"app_switch": False},
        raw_text="some text"
    )
    assert ctx.app == "vscode"
    assert ctx.focus_level == 0.8
    assert "main.py" in ctx.entities


from unittest.mock import patch as upatch
from process.ocr import OCREngine
from process.stt import STTEngine
from config import IdleUnloadConfig, WhisperConfig


def test_ocr_engine_deduplicates_identical_frames():
    cfg = IdleUnloadConfig(ocr_minutes=3.0)
    engine = OCREngine(cfg)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with upatch("pytesseract.image_to_string", return_value="hello world") as mock_ocr:
        result1 = engine.extract(frame)
        result2 = engine.extract(frame)
    assert mock_ocr.call_count == 1
    assert result1 == "hello world"
    assert result2 == "hello world"


def test_ocr_engine_processes_different_frames():
    cfg = IdleUnloadConfig(ocr_minutes=3.0)
    engine = OCREngine(cfg)
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = np.ones((100, 100, 3), dtype=np.uint8) * 128
    with upatch("pytesseract.image_to_string", side_effect=["hello world today", "something different here"]) as mock_ocr:
        r1 = engine.extract(frame_a)
        r2 = engine.extract(frame_b)
    assert mock_ocr.call_count == 2
    assert r1 == "hello world today"
    assert r2 == "something different here"


def test_stt_engine_transcribes(monkeypatch):
    cfg_idle = IdleUnloadConfig(whisper_minutes=3.0)
    cfg_whisper = WhisperConfig(device="cpu", compute_type="int8")
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " hello there"
    mock_model.transcribe.return_value = ([mock_segment], MagicMock(language="en"))
    with upatch("faster_whisper.WhisperModel", return_value=mock_model):
        engine = STTEngine(cfg_idle, cfg_whisper)
        audio = np.zeros(16000, dtype=np.float32).tobytes()
        result = engine.transcribe(audio)
    assert result["text"] == "hello there"
    assert result["language"] == "en"


from capture.push_to_talk import PushToTalk


def test_push_to_talk_toggle():
    ptt = PushToTalk()
    assert not ptt.is_recording
    # Patch _start so we don't need PortAudio hardware in tests
    mock_stream = MagicMock()
    def fake_start(self_inner):
        self_inner._frames = []
        self_inner.is_recording = True
        self_inner._stream = mock_stream
    with upatch.object(PushToTalk, "_start", fake_start):
        ptt.toggle()
        assert ptt.is_recording
        ptt.toggle()
    assert not ptt.is_recording


def test_push_to_talk_stop_when_not_recording():
    ptt = PushToTalk()
    ptt.stop()
    assert not ptt.is_recording


from capture.done_word import DoneWordDetector

def test_done_word_exact_match():
    d = DoneWordDetector("pineapple", tolerance=2)
    assert d.check("pineapple") is True

def test_done_word_split_variant():
    d = DoneWordDetector("pineapple", tolerance=2)
    assert d.check("pine apple") is True

def test_done_word_initial_consonant_confusion():
    # "find apple" has phoneme dist=2 from pineapple
    d = DoneWordDetector("pineapple", tolerance=2)
    assert d.check("find apple") is True

def test_done_word_syllable_shift():
    d = DoneWordDetector("pineapple", tolerance=2)
    assert d.check("pie napple") is True

def test_done_word_false_negative_apple():
    d = DoneWordDetector("pineapple", tolerance=2)
    assert d.check("I want an apple") is False

def test_done_word_false_negative_unrelated():
    d = DoneWordDetector("pineapple", tolerance=2)
    assert d.check("hello world") is False

def test_done_word_strips_from_text():
    d = DoneWordDetector("pineapple", tolerance=2)
    result = d.strip("please do that pineapple")
    assert "pineapple" not in result.lower()
    assert "please do that" in result

def test_done_word_strips_split_variant():
    d = DoneWordDetector("pineapple", tolerance=2)
    result = d.strip("open the door pine apple")
    assert result.strip() == "open the door"


from capture.eot_detector import EOTDetector
import numpy as np

def _make_frames(n_frames: int, amplitude: float) -> list[bytes]:
    """Generate n_frames of 30ms PCM audio at given amplitude."""
    samples_per_frame = 480  # 30ms at 16kHz
    frames = []
    for _ in range(n_frames):
        pcm = (np.random.randn(samples_per_frame) * amplitude * 32767).astype(np.int16)
        frames.append(pcm.tobytes())
    return frames

def test_eot_high_energy_mid_utterance():
    det = EOTDetector(threshold=0.7, hard_cutoff_frames=100)
    samples_per_frame = 480
    frames = []
    for i in range(20):
        amp = 0.1 + i * 0.02  # rising amplitude 0.1 → 0.48
        pcm = (np.random.randn(samples_per_frame) * amp * 32767).astype(np.int16)
        frames.append(pcm.tobytes())
    det.update(frames)
    assert det.probability() < 0.7

def test_eot_falling_energy_end_of_utterance():
    det = EOTDetector(threshold=0.7, hard_cutoff_frames=100)
    # Build frames with rapidly decaying energy
    samples_per_frame = 480
    frames = []
    for i in range(20):
        amp = max(0.0, 0.5 - i * 0.04)  # decays to 0
        pcm = (np.random.randn(samples_per_frame) * amp * 32767).astype(np.int16)
        frames.append(pcm.tobytes())
    det.update(frames)
    assert det.probability() > 0.7

def test_eot_hard_cutoff_triggers():
    det = EOTDetector(threshold=0.7, hard_cutoff_frames=5)
    frames = _make_frames(5, amplitude=0.3)
    det.update(frames)
    assert det.hard_cutoff_reached() is True

def test_eot_hard_cutoff_not_triggered_early():
    det = EOTDetector(threshold=0.7, hard_cutoff_frames=100)
    frames = _make_frames(10, amplitude=0.3)
    det.update(frames)
    assert det.hard_cutoff_reached() is False

def test_eot_reset_clears_state():
    det = EOTDetector(threshold=0.7, hard_cutoff_frames=5)
    frames = _make_frames(5, amplitude=0.3)
    det.update(frames)
    assert det.hard_cutoff_reached() is True
    det.reset()
    assert det.hard_cutoff_reached() is False
    assert det.probability() < 0.7


def test_stt_engine_passes_initial_prompt(monkeypatch):
    from process.stt import STTEngine
    from config import IdleUnloadConfig, WhisperConfig
    from unittest.mock import MagicMock
    from unittest.mock import patch as upatch

    cfg_idle = IdleUnloadConfig(whisper_minutes=3.0)
    cfg_whisper = WhisperConfig(device="cpu", compute_type="int8")
    mock_model = MagicMock()
    mock_seg = MagicMock()
    mock_seg.text = " hello"
    mock_seg.avg_logprob = -0.3
    mock_seg.no_speech_prob = 0.1
    mock_model.transcribe.return_value = ([mock_seg], MagicMock(language="en"))

    with upatch("faster_whisper.WhisperModel", return_value=mock_model):
        engine = STTEngine(cfg_idle, cfg_whisper)
        import numpy as np
        audio = np.zeros(16000, dtype=np.float32).tobytes()
        result = engine.transcribe(audio, initial_prompt="test context")

    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs.get("initial_prompt") == "test context"
    assert result["text"] == "hello"
    assert result["avg_logprob"] == -0.3
    assert result["no_speech_prob"] == 0.1


from capture.rolling_transcriber import RollingTranscriber
import queue, threading

def _make_mock_stt(text=" hello", avg_logprob=-0.3, no_speech_prob=0.1):
    from unittest.mock import MagicMock
    stt = MagicMock()
    stt.transcribe.return_value = {
        "text": text, "language": "en",
        "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob,
    }
    return stt

def test_rolling_transcriber_emits_event_on_clean_chunk():
    from core.event_queue import EventQueue
    from core.events import EventType
    from capture.done_word import DoneWordDetector
    import asyncio, numpy as np

    loop = asyncio.new_event_loop()
    eq = EventQueue()
    stt = _make_mock_stt(" hello world", avg_logprob=-0.3, no_speech_prob=0.1)
    done_detector = DoneWordDetector("pineapple", tolerance=2)
    flush_called = threading.Event()

    rt = RollingTranscriber(
        stt_engine=stt,
        event_queue=eq,
        done_detector=done_detector,
        flush_callback=lambda: flush_called.set(),
        confidence_gate_logprob=-0.8,
        noise_speech_prob_max=0.6,
    )
    rt.start()

    chunk = np.zeros(533 * 480, dtype=np.int16).tobytes()
    rt.push(chunk)
    import time; time.sleep(0.3)
    rt.stop()

    assert rt.rolling_text == "hello world"
    loop.close()

def test_rolling_transcriber_rejects_noisy_chunk():
    from core.event_queue import EventQueue
    from capture.done_word import DoneWordDetector
    import numpy as np

    eq = EventQueue()
    stt = _make_mock_stt(" background noise", avg_logprob=-1.5, no_speech_prob=0.8)
    done_detector = DoneWordDetector("pineapple", tolerance=2)

    rt = RollingTranscriber(
        stt_engine=stt, event_queue=eq,
        done_detector=done_detector, flush_callback=lambda: None,
        confidence_gate_logprob=-0.8, noise_speech_prob_max=0.6,
    )
    rt.start()
    chunk = np.zeros(480 * 10, dtype=np.int16).tobytes()
    rt.push(chunk)
    import time; time.sleep(0.3)
    rt.stop()

    assert rt.rolling_text == ""

def test_rolling_transcriber_triggers_flush_on_done_word():
    from core.event_queue import EventQueue
    from capture.done_word import DoneWordDetector
    import numpy as np

    eq = EventQueue()
    stt = _make_mock_stt(" open the door pineapple", avg_logprob=-0.2, no_speech_prob=0.05)
    done_detector = DoneWordDetector("pineapple", tolerance=2)
    flush_called = threading.Event()

    rt = RollingTranscriber(
        stt_engine=stt, event_queue=eq,
        done_detector=done_detector, flush_callback=lambda: flush_called.set(),
        confidence_gate_logprob=-0.8, noise_speech_prob_max=0.6,
    )
    rt.start()
    chunk = np.zeros(480 * 20, dtype=np.int16).tobytes()
    rt.push(chunk)
    assert flush_called.wait(timeout=2.0), "flush not triggered within 2s"
    rt.stop()


def test_mic_watcher_v2_initialises():
    from core.event_queue import EventQueue
    from config import MicConfig
    eq = EventQueue()
    cfg = MicConfig(vad_aggressiveness=2, chunk_frames=533)
    watcher = MicWatcher(eq, cfg)
    assert watcher is not None

def test_mic_watcher_v2_has_rolling_transcriber():
    from core.event_queue import EventQueue
    from config import MicConfig
    from capture.rolling_transcriber import RollingTranscriber
    eq = EventQueue()
    cfg = MicConfig()
    watcher = MicWatcher(eq, cfg)
    assert isinstance(watcher._rolling_transcriber, RollingTranscriber)

def test_mic_watcher_v2_has_eot_detector():
    from core.event_queue import EventQueue
    from config import MicConfig
    from capture.eot_detector import EOTDetector
    eq = EventQueue()
    cfg = MicConfig()
    watcher = MicWatcher(eq, cfg)
    assert isinstance(watcher._eot, EOTDetector)
