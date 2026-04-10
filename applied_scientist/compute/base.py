from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JobStatus:
    job_id: str
    state: str                         # "pending", "running", "completed", "failed", "cancelled"
    exit_code: int | None = None
    elapsed_seconds: float | None = None


class JobRunner(ABC):
    @abstractmethod
    def submit(self, command: str, job_name: str,
               resources: dict | None = None,
               scripts_dir: str | None = None) -> str:
        """Submit a job. Return job ID."""

    @abstractmethod
    def status(self, job_id: str) -> JobStatus:
        """Return current job status."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel a running job."""

    @abstractmethod
    def get_log(self, job_id: str, tail: int = 50) -> str:
        """Return last N lines of job output."""
