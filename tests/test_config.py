# tests/test_config.py
import textwrap
import tempfile
import os
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
