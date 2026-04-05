"""Compute backends (pluggable). Registry provides get_runner(config)."""

from __future__ import annotations

from applied_scientist.compute.base import JobRunner, JobStatus


def get_runner(config) -> JobRunner:
    """Instantiate compute backend from config.
    config is a ComputeConfig dataclass with .backend and .slurm attributes."""
    backend = config.backend
    if backend == "slurm":
        from applied_scientist.compute.slurm import SLURMRunner
        return SLURMRunner(**config.slurm)
    elif backend == "local":
        from applied_scientist.compute.local import LocalRunner
        return LocalRunner()
    else:
        raise ValueError(f"Unknown compute backend: {backend}. Available: slurm, local")
