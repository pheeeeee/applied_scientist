from __future__ import annotations

import os
import subprocess
import time
import uuid

from applied_scientist.compute.base import JobRunner, JobStatus


class LocalRunner(JobRunner):
    """Local compute backend: subprocess with GPU assignment."""

    def __init__(self):
        self._processes: dict[str, dict] = {}  # job_id -> {process, log_path, start_time}

    def submit(self, command: str, job_name: str,
               resources: dict | None = None) -> str:
        job_id = str(uuid.uuid4())[:8]
        log_path = f"{job_name}_{job_id}.log"

        env = os.environ.copy()
        if resources and "gpu_id" in resources:
            env["CUDA_VISIBLE_DEVICES"] = str(resources["gpu_id"])

        log_file = open(log_path, "w")
        process = subprocess.Popen(
            command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
            env=env,
        )
        self._processes[job_id] = {
            "process": process,
            "log_path": log_path,
            "log_file": log_file,
            "start_time": time.time(),
        }
        return job_id

    def status(self, job_id: str) -> JobStatus:
        if job_id not in self._processes:
            return JobStatus(job_id=job_id, state="failed")

        info = self._processes[job_id]
        process = info["process"]
        elapsed = time.time() - info["start_time"]
        rc = process.poll()

        if rc is None:
            return JobStatus(job_id=job_id, state="running",
                             elapsed_seconds=elapsed)
        elif rc == 0:
            return JobStatus(job_id=job_id, state="completed",
                             exit_code=rc, elapsed_seconds=elapsed)
        else:
            return JobStatus(job_id=job_id, state="failed",
                             exit_code=rc, elapsed_seconds=elapsed)

    def cancel(self, job_id: str) -> None:
        if job_id in self._processes:
            self._processes[job_id]["process"].terminate()

    def get_log(self, job_id: str, tail: int = 50) -> str:
        if job_id not in self._processes:
            return f"No log for job {job_id}"
        log_path = self._processes[job_id]["log_path"]
        if not os.path.exists(log_path):
            return ""
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-tail:])
