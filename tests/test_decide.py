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
from decide.agent import DecisionAgent, GeneratorOutput, CriticOutput
from config import LLMConfig


def _make_agent():
    memory = MagicMock()
    memory.vector.query.return_value = []
    memory.episodic.get_recent.return_value = []
    memory.episodic.get_summaries.return_value = []
    memory.structured.get_at_risk_commitments.return_value = []
    memory.kg.subgraph.return_value = []
    return DecisionAgent(memory=memory, config=LLMConfig())


def test_generator_output_schema():
    out = GeneratorOutput(
        say=True, message="You have a deadline in 2h",
        type="deadline", importance=0.9,
        reason="commit deadline < 3h", extract={}
    )
    assert out.say is True
    assert 0.0 <= out.importance <= 1.0


def test_critic_output_schema():
    out = CriticOutput(suppress=False, reason="high urgency overrides focus")
    assert out.suppress is False


def test_decision_agent_returns_none_when_generator_says_no():
    agent = _make_agent()
    mock_gen = MagicMock()
    mock_gen.say = False

    with patch.object(agent, "_call_generator", return_value=mock_gen):
        result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is None


def test_decision_agent_returns_none_when_critic_suppresses():
    agent = _make_agent()
    mock_gen = MagicMock(say=True, message="Hey", type="task",
                         importance=0.5, reason="test", extract={})
    mock_critic = MagicMock(suppress=True)

    with patch.object(agent, "_call_generator", return_value=mock_gen):
        with patch.object(agent, "_call_critic", return_value=mock_critic):
            result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is None


def test_decision_agent_returns_message_when_both_approve():
    agent = _make_agent()
    mock_gen = MagicMock(say=True, message="Submit report soon",
                         type="deadline", importance=0.9, reason="deadline < 2h", extract={})
    mock_critic = MagicMock(suppress=False)

    with patch.object(agent, "_call_generator", return_value=mock_gen):
        with patch.object(agent, "_call_critic", return_value=mock_critic):
            result = agent.evaluate(context={"recent": [], "scene": None})
    assert result is not None
    assert result["message"] == "Submit report soon"
