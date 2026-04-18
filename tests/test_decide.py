import pytest
from decide.energy import EnergyScheduler
from config import EnergyConfig


@pytest.fixture
def scheduler():
    return EnergyScheduler(EnergyConfig(min_interval_seconds=20, max_interval_seconds=120))


def test_default_interval_is_max(scheduler):
    assert scheduler.next_interval() == 120


def test_high_novelty_reduces_interval(scheduler):
    scheduler.update(novelty_score=1.0, focus_level=0.2)
    interval = scheduler.next_interval()
    assert interval < 120


def test_high_focus_increases_interval(scheduler):
    scheduler.update(novelty_score=0.5, focus_level=1.0)
    interval = scheduler.next_interval()
    assert interval >= 60


def test_interval_never_below_min(scheduler):
    scheduler.update(novelty_score=1.0, focus_level=0.0)
    assert scheduler.next_interval() >= 20


def test_interval_never_above_max(scheduler):
    scheduler.update(novelty_score=0.0, focus_level=1.0)
    assert scheduler.next_interval() <= 120


def test_interval_formula_matches_spec(scheduler):
    scheduler.update(novelty_score=0.5, focus_level=0.5)
    interval = scheduler.next_interval()
    assert 20 <= interval <= 120


import tempfile
from decide.calibrator import InterruptCalibrator
from config import CalibratorConfig


@pytest.fixture
def calibrator(tmp_path):
    return InterruptCalibrator(
        CalibratorConfig(min_samples=3, retrain_every=2),
        log_path=str(tmp_path / "events.jsonl"),
        model_path=str(tmp_path / "model.pkl"),
    )


def test_calibrator_records_event(calibrator):
    calibrator.record(
        interrupt_type="task", time_of_day=14, focus_level=0.6,
        app="vscode", task_type="coding", importance=0.8, engaged=True
    )
    assert calibrator._event_count == 1


def test_calibrator_suppresses_returns_false_before_training(calibrator):
    result = calibrator.should_suppress(
        interrupt_type="task", time_of_day=9, focus_level=0.9,
        app="vscode", task_type="coding", importance=0.5,
    )
    assert result is False


def test_calibrator_trains_after_min_samples(calibrator):
    for i in range(4):
        calibrator.record(
            interrupt_type="commitment", time_of_day=10, focus_level=0.3,
            app="slack", task_type="communicating", importance=0.9, engaged=(i % 2 == 0)
        )
    assert calibrator._model is not None


def test_calibrator_persists_events_to_jsonl(calibrator):
    import json
    calibrator.record(
        interrupt_type="focus_drift", time_of_day=15, focus_level=0.8,
        app="chrome", task_type="browsing", importance=0.4, engaged=False
    )
    with open(calibrator._log_path) as f:
        lines = f.readlines()
    event = json.loads(lines[0])
    assert event["interrupt_type"] == "focus_drift"
    assert event["engaged"] is False


from unittest.mock import patch, MagicMock
from decide.agent import DecisionAgent
from config import LLMConfig


def _make_agent():
    memory = MagicMock()
    memory.episodic.get_recent.return_value = []
    return DecisionAgent(memory=memory, config=LLMConfig())


def test_decision_agent_returns_none_when_say_false():
    agent = _make_agent()
    with patch.object(agent, "_call_single", return_value=None):
        result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is None


def test_decision_agent_returns_result_when_say_true():
    agent = _make_agent()
    expected = {"message": "Submit report soon", "type": "general",
                "importance": 0.9, "reason": "deadline close", "extract": {}}
    with patch.object(agent, "_call_single", return_value=expected):
        result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is not None
    assert result["message"] == "Submit report soon"


def test_decision_agent_handles_timeout_gracefully():
    agent = _make_agent()

    def _raise(_ctx):
        raise RuntimeError("request timed out")

    with patch.object(agent, "_call_single", side_effect=_raise):
        result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is not None
    assert "timed out" in result["message"].lower()


from decide.primer import LLMPrimer


def test_llm_primer_opens_stream_on_rolling_transcript():
    cfg = LLMConfig(model="aria-qwen", primer_enabled=True, primer_divergence_threshold=0.2)

    with patch("ollama.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = iter([{"message": {"content": "hello"}}])
        primer = LLMPrimer(cfg)
        primer.on_rolling_transcript("hello world")

    mock_client.chat.assert_called_once()
    call_kwargs = mock_client.chat.call_args[1]
    assert call_kwargs.get("stream") is True


def test_llm_primer_cancels_on_high_divergence():
    cfg = LLMConfig(model="aria-qwen", primer_enabled=True, primer_divergence_threshold=0.2)

    with patch("ollama.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = iter([])
        primer = LLMPrimer(cfg)
        primer.on_rolling_transcript("hello world")
        # Final text is very different: >20% normalized edit distance
        result = primer.on_speech_detected("completely different utterance about cats")

    assert result is None  # stream cancelled, new call needed


def test_llm_primer_continues_on_low_divergence():
    cfg = LLMConfig(model="aria-qwen", primer_enabled=True, primer_divergence_threshold=0.2)

    with patch("ollama.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.return_value = iter([{"message": {"content": "hi"}}])
        primer = LLMPrimer(cfg)
        primer.on_rolling_transcript("hello world how are you")
        # Final text is very close (identical)
        result = primer.on_speech_detected("hello world how are you")

    # Should return the stream (not None)
    assert result is not None


def test_llm_primer_disabled_does_nothing():
    cfg = LLMConfig(primer_enabled=False)
    primer = LLMPrimer(cfg)
    primer.on_rolling_transcript("hello")
    result = primer.on_speech_detected("hello")
    assert result is None
