"""Task Adapter template. Copy this directory and implement for your task."""

from applied_scientist.task.base import TaskAdapter


class MyTaskAdapter(TaskAdapter):

    @property
    def name(self) -> str:
        """Human-readable task name."""
        return "My Task"

    @property
    def metric(self) -> tuple[str, str]:
        """(metric_name, direction). direction is 'higher' or 'lower'."""
        return ("metric_name", "higher")

    @property
    def domain_context(self) -> str:
        """Description for Explorer (literature search) and Builder (implementation).
        Be specific: name frameworks, observation/action spaces, constraints."""
        return "Describe your task here."

    @property
    def code_map(self) -> str:
        """Complete reference for Builder. Must contain enough detail that Builder
        can implement architectures WITHOUT reading source files in most cases.

        Include: every editable file, key classes/functions, how components connect,
        framework-specific patterns, files NOT to edit, where to add new files."""
        return "## Files\n- model.py: edit this\n- train.py: DO NOT EDIT"

    def train(self, config: dict, seed: int, time_budget: int,
              log_path: str, checkpoint_dir: str) -> dict:
        """Run one training experiment. Return dict of metrics.
        Must write progress.json periodically for early stopping.
        Return {"status": "crash", "error": "..."} on failure."""
        raise NotImplementedError("Implement training for your task")

    def evaluate(self, checkpoint_dir: str, n_eval_samples: int) -> dict:
        """Evaluate a trained checkpoint. Return dict of metrics."""
        raise NotImplementedError("Implement evaluation for your task")
