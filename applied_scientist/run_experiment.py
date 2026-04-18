"""Standardized training wrapper. Called by compute backends (SLURM/local).
Builder's _build_train_command generates: python -m applied_scientist.run_experiment --spec ... --seed ...
This script loads the task adapter, reads the spec, calls adapter.train(), and prints results as JSON."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback


def load_task_adapter(module_path: str):
    """Load TaskAdapter from a module directory."""
    from applied_scientist.task.base import TaskAdapter
    spec = importlib.util.spec_from_file_location(
        "task_adapter", os.path.join(module_path, "adapter.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, TaskAdapter) and obj is not TaskAdapter:
            return obj()
    raise ValueError(f"No TaskAdapter subclass found in {module_path}/adapter.py")


def main():
    parser = argparse.ArgumentParser(description="Run a single training experiment")
    parser.add_argument("--spec", required=True, help="Path to experiment spec YAML")
    parser.add_argument("--task-module", required=True, help="Path to task adapter module")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-budget", type=int, required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    adapter = load_task_adapter(args.task_module)

    from applied_scientist.core.spec import ExperimentSpec
    experiment_spec = ExperimentSpec.from_yaml(args.spec)

    config = {**experiment_spec.task_config, **experiment_spec.training_config}

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    try:
        result = adapter.train(
            config=config,
            seed=args.seed,
            time_budget=args.time_budget,
            log_path=args.log_path,
            checkpoint_dir=args.checkpoint_dir,
        )
    except Exception as e:
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        sys.stderr.flush()
        result = {
            "status": "crash",
            "error": f"{type(e).__name__}: {e}",
            "traceback": tb,
        }

    # Persist results to file so they survive even if the builder agent is not alive
    results_file = os.path.join(args.log_path, "results.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
