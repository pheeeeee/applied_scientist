from __future__ import annotations

import json
import os


class HyperparameterTuner:
    """Bayesian optimization for hyperparameter tuning after each experiment."""

    def __init__(self, task_adapter, job_runner, config):
        self.task = task_adapter
        self.runner = job_runner
        self.config = config

    def compute_budget(self, queue_depth: int, result,
                       best_metric: float | None, training_still_improving: bool) -> int:
        """Compute dynamic tuning iteration budget."""
        base = self.config.tuning.max_iterations

        throttle = self.config.system.queue_throttle_high
        queue_factor = 1.0 if queue_depth == 0 else max(0.2, 1.0 - queue_depth / throttle)

        time_factor = min(1.0, self.config.system.time_budget / 1800)

        if result.metric_value is not None and best_metric is not None and best_metric > 0:
            promise_factor = result.metric_value / best_metric
        else:
            promise_factor = 0.5

        improvement_factor = 1.0 if training_still_improving else 0.5

        n = round(base * queue_factor * time_factor * promise_factor * improvement_factor)
        return max(self.config.tuning.min_iterations, min(n, self.config.tuning.max_iterations))

    def get_search_space(self, spec) -> dict:
        """Merge task defaults with spec-specific overrides."""
        space = dict(self.task.get_tunable_defaults())
        if spec.tunable_hyperparameters:
            space.update(spec.tunable_hyperparameters)
        return space

    def _suggest_params(self, trial, search_space: dict, base_config: dict) -> dict:
        """Use Optuna trial to suggest hyperparameters."""
        params = dict(base_config)
        for name, spec in search_space.items():
            if spec["type"] == "log_uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif spec["type"] == "uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "discrete":
                params[name] = trial.suggest_categorical(name, spec["values"])
            elif spec["type"] == "factor":
                base_val = base_config.get(name, 1)
                factor = trial.suggest_categorical(f"{name}_factor", spec["factors"])
                params[name] = int(base_val * factor) if isinstance(base_val, int) else base_val * factor
        return params

    def tune(self, spec, n_iterations: int, log_dir: str) -> dict | None:
        """Run Bayesian optimization. Returns best params dict or None."""
        try:
            import optuna
        except ImportError:
            return None

        search_space = self.get_search_space(spec)
        if not search_space:
            return None

        metric_name, direction = self.task.metric
        optuna_direction = "maximize" if direction == "higher" else "minimize"
        study = optuna.create_study(direction=optuna_direction)
        tuning_budget = self.config.system.time_budget // 3

        def objective(trial):
            params = self._suggest_params(trial, search_space, spec.training_config)
            trial_dir = os.path.join(log_dir, f"tune_trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            ckpt_dir = os.path.join(trial_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                result = self.task.train(
                    config={**spec.task_config, **params},
                    seed=42,
                    time_budget=tuning_budget,
                    log_path=trial_dir,
                    checkpoint_dir=ckpt_dir,
                )
                value = result.get(metric_name)
                if value is None:
                    return float("-inf") if direction == "higher" else float("inf")
                return value
            except Exception:
                return float("-inf") if direction == "higher" else float("inf")

        study.optimize(objective, n_trials=n_iterations, timeout=self.config.system.time_budget)

        if study.best_trial:
            return study.best_params
        return None
