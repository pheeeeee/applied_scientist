"""Compute backends (pluggable). Registry provides get_runner(config)."""

from __future__ import annotations

from applied_scientist.compute.base import JobRunner, JobStatus


def get_runner(config, workspace: str | None = None) -> JobRunner:
    """Instantiate compute backend from config.

    Args:
        config: ComputeConfig dataclass with .backend and .slurm attributes.
        workspace: Optional workspace path for script persistence and log discovery.
    """
    backend = config.backend
    if backend == "slurm":
        from applied_scientist.compute.slurm import SLURMRunner
        kwargs = dict(config.slurm)
        if workspace:
            kwargs["workspace"] = workspace
        return SLURMRunner(**kwargs)
    elif backend == "local":
        from applied_scientist.compute.local import LocalRunner
        return LocalRunner()
    else:
        raise ValueError(f"Unknown compute backend: {backend}. Available: slurm, local")
