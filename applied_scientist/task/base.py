from __future__ import annotations

from abc import ABC, abstractmethod


class TaskAdapter(ABC):
    """Bridge between Applied Scientist (generic) and a specific ML task.
    You write this once before launching the system. Agents call it but never modify it."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name."""

    @property
    @abstractmethod
    def metric(self) -> tuple[str, str]:
        """(metric_name, direction). direction is 'higher' or 'lower'."""

    @property
    @abstractmethod
    def domain_context(self) -> str:
        """Domain description for agents (Explorer for paper search, Builder for implementation)."""

    @property
    @abstractmethod
    def code_map(self) -> str:
        """Complete reference for Builder. Must contain enough detail that Builder can
        implement architectures WITHOUT reading source files in most cases.

        Include:
        - Every editable file with its full path
        - Key classes, functions, and their signatures
        - How components connect (what calls what)
        - Framework-specific patterns (e.g., how to register a custom model in RLlib)
        - Files that must NOT be edited (adapter.py, evaluate.py)
        - Where to add new files if needed
        - Observation/action space details with dimensions
        - Hardware constraints
        """

    @property
    def progress_file(self) -> str:
        """Filename for periodic progress updates during training.
        Training script should write JSON: {"step": int, "<metric>": float, ...}
        Default: 'progress.json'"""
        return "progress.json"

    @abstractmethod
    def train(self, config: dict, seed: int, time_budget: int,
              log_path: str, checkpoint_dir: str) -> dict:
        """Run training. Return dict of metrics including primary metric.
        Training script MUST write progress_file periodically.
        Return {"status": "crash", "error": "..."} on failure."""

    @abstractmethod
    def evaluate(self, checkpoint_dir: str, n_eval_samples: int) -> dict:
        """Evaluate checkpoint. Return dict of metrics."""

    def get_baseline_spec(self) -> dict | None:
        """Return a baseline experiment spec dict, or None for system default."""
        return None

    def get_random_baseline(self) -> dict | None:
        """Return metrics for an untrained/random agent. Called once at startup.
        No training needed — zero GPU cost. Provides early stopping floor.
        Return None to skip random baseline.

        Example: {"win_rate": 0.50, "avg_reward": 0.0}"""
        return None

    def get_tunable_defaults(self) -> dict:
        """Return default tunable hyperparameters and search spaces for this task.
        Explorer-defined spec.tunable_hyperparameters override these.
        Return empty dict to use only spec-defined hyperparameters.

        Format: {"param_name": {"type": "log_uniform"|"uniform"|"discrete"|"factor",
                                 "low": float, "high": float,
                                 "values": list,
                                 "factors": list}}

        Example:
            {"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
             "train_batch_size": {"type": "factor", "factors": [0.5, 1, 2, 4]},
             "gamma": {"type": "discrete", "values": [0.99, 0.995, 0.999]}}
        """
        return {}
