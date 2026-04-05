import os
import pytest
from applied_scientist.config import Config, SystemConfig, TuningConfig


def test_config_from_yaml(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("""
system:
  time_budget: 900
  n_gpus: 2
tuning:
  enabled: false
llm:
  explorer:
    backend: anthropic
    model: claude-haiku-4-5-20251001
  critic:
    backend: anthropic
    model: claude-sonnet-4-6
  builder:
    backend: anthropic
    model: claude-sonnet-4-6
  orchestrator:
    backend: anthropic
    model: claude-haiku-4-5-20251001
compute:
  backend: local
task:
  module: tasks/test
paths:
  workspace: ./ws
""")
    config = Config.from_yaml(str(config_yaml))
    assert config.system.time_budget == 900
    assert config.system.n_gpus == 2
    assert config.tuning.enabled is False
    assert config.llm["explorer"].model == "claude-haiku-4-5-20251001"
    assert config.task["module"] == "tasks/test"


def test_config_defaults():
    s = SystemConfig()
    assert s.time_budget == 1800
    assert s.early_stop_fraction == 0.4
    t = TuningConfig()
    assert t.enabled is True
    assert t.max_iterations == 15


def test_env_var_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODEL", "my-model")
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("""
system: {}
tuning: {}
llm:
  explorer:
    backend: anthropic
    model: ${TEST_MODEL}
  critic:
    backend: anthropic
    model: ${TEST_MODEL}
  builder:
    backend: anthropic
    model: ${TEST_MODEL}
  orchestrator:
    backend: anthropic
    model: ${TEST_MODEL}
compute:
  backend: local
task: {}
paths: {}
""")
    config = Config.from_yaml(str(config_yaml))
    assert config.llm["explorer"].model == "my-model"
