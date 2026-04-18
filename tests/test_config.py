# tests/test_config.py
import textwrap
import pytest
from config import Config


def test_default_config():
    cfg = Config()
    assert cfg.screen.interval_seconds == 3.0
    assert cfg.screen.diff_threshold == 0.15
    assert cfg.mic.vad_aggressiveness == 2
    assert cfg.memory.rolling_days == 7
    assert cfg.llm.model == "llama3.1:8b"
    assert cfg.hotkey == "ctrl+shift+space"
    assert cfg.overlay.auto_dismiss_seconds == 8


def test_from_yaml_overrides_values(tmp_path):
    yaml_content = textwrap.dedent("""
        screen:
          interval_seconds: 5.0
          diff_threshold: 0.20
        llm:
          model: llama3.2:3b
        hotkey: ctrl+shift+a
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    cfg = Config.from_yaml(str(config_file))
    assert cfg.screen.interval_seconds == 5.0
    assert cfg.screen.diff_threshold == 0.20
    assert cfg.llm.model == "llama3.2:3b"
    assert cfg.hotkey == "ctrl+shift+a"
    assert cfg.mic.vad_aggressiveness == 2  # unset = default


def test_from_yaml_missing_file_returns_defaults():
    cfg = Config.from_yaml("/nonexistent/path/config.yaml")
    assert cfg.screen.interval_seconds == 3.0


def test_ebbinghaus_thresholds():
    cfg = Config()
    assert cfg.memory.initial_stability_days == 1.0
    assert cfg.memory.important_stability_days == 30.0
    assert cfg.memory.ebbinghaus_purge_threshold == 0.2


def test_idle_unload_minutes():
    cfg = Config()
    assert cfg.idle_unload.ocr_minutes == 3.0
    assert cfg.idle_unload.whisper_minutes == 3.0
    assert cfg.idle_unload.tts_minutes == 2.0


def test_mic_config_new_fields():
    from config import MicConfig
    cfg = MicConfig()
    assert cfg.done_word == "pineapple"
    assert cfg.done_word_phoneme_tolerance == 2
    assert cfg.chunk_frames == 533
    assert cfg.eot_probability_threshold == 0.7
    assert cfg.eot_hard_cutoff_frames == 50
    assert cfg.confidence_gate_logprob == -0.8
    assert cfg.noise_speech_prob_max == 0.6

def test_llm_config_primer_fields():
    from config import LLMConfig
    cfg = LLMConfig()
    assert cfg.primer_enabled is True
    assert cfg.primer_divergence_threshold == 0.2
