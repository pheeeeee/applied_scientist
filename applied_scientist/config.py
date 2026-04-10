from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import yaml


@dataclass
class SystemConfig:
    time_budget: int = 1800
    n_gpus: int = 1
    baseline_seeds: int = 3
    eval_samples: int = 100
    early_stop_fraction: float = 0.4
    max_spec_review_rounds: int = 3
    queue_throttle_high: int = 10
    queue_throttle_low: int = 5
    timeout_multiplier: float = 2.0
    cost_alert_threshold: float | None = None


@dataclass
class TuningConfig:
    enabled: bool = True
    max_iterations: int = 15
    min_iterations: int = 3


@dataclass
class LLMAgentConfig:
    backend: str
    model: str
    base_url: str | None = None


@dataclass
class ComputeConfig:
    backend: str = "local"
    slurm: dict = field(default_factory=dict)


@dataclass
class PathsConfig:
    workspace: str = "./workspace"
    results: str = "./workspace/results"
    configs: str = "./workspace/configs"
    checkpoints: str = "./workspace/results/checkpoints"
    logs: str = "./workspace/results/logs"


@dataclass
class InterfaceConfig:
    backend: str = "terminal"  # "terminal", "slack", "telegram"


@dataclass
class Config:
    system: SystemConfig
    tuning: TuningConfig
    llm: dict[str, LLMAgentConfig]
    compute: ComputeConfig
    task: dict
    paths: PathsConfig
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        """Load from YAML, resolve env vars (${VAR}), apply defaults."""
        with open(path) as f:
            raw = f.read()

        def resolve_env(match):
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        raw = re.sub(r'\$\{(\w+)\}', resolve_env, raw)
        data = yaml.safe_load(raw)

        # Resolve relative paths against the config file's directory
        config_dir = os.path.dirname(os.path.abspath(path))
        project_root = os.path.dirname(config_dir)  # configs/ -> project root
        paths_data = data.get("paths", {})
        for key, val in paths_data.items():
            if isinstance(val, str) and not os.path.isabs(val):
                paths_data[key] = os.path.normpath(os.path.join(project_root, val))

        return cls(
            system=SystemConfig(**data.get("system", {})),
            tuning=TuningConfig(**data.get("tuning", {})),
            llm={k: LLMAgentConfig(**v) for k, v in data.get("llm", {}).items()},
            compute=ComputeConfig(**data.get("compute", {})),
            task=data.get("task", {}),
            paths=PathsConfig(**paths_data),
            interface=InterfaceConfig(**data.get("interface", {})),
        )
