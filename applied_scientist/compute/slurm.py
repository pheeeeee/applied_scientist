from __future__ import annotations

import os
import subprocess
import tempfile

from applied_scientist.compute.base import JobRunner, JobStatus


class SLURMRunner(JobRunner):
    """SLURM compute backend: sbatch, sacct, scancel."""

    def __init__(self, partition: str = "gpu", gres: str = "gpu:1",
                 mem: str = "48G", time: str = "01:00:00",
                 setup_commands: list[str] | None = None):
        self.partition = partition
        self.gres = gres
        self.mem = mem
        self.time = time
        self.setup_commands = setup_commands or []

    def submit(self, command: str, job_name: str,
               resources: dict | None = None) -> str:
        """Submit via sbatch --parsable. Returns job ID string."""
        # Write temporary batch script
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --gres={self.gres}",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --output=slurm-%j.out",
        ]
        for cmd in self.setup_commands:
            script_lines.append(cmd)
        script_lines.append(command)

        fd, script_path = tempfile.mkstemp(suffix=".sh")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(script_lines) + "\n")

        result = subprocess.run(
            ["sbatch", "--parsable", script_path],
            capture_output=True, text=True)
        os.unlink(script_path)

        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")
        return result.stdout.strip()

    def status(self, job_id: str) -> JobStatus:
        """Check via sacct."""
        result = subprocess.run(
            ["sacct", "-j", job_id,
             "--format=State,ExitCode,Elapsed", "--noheader", "--parsable2"],
            capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            return JobStatus(job_id=job_id, state="pending")

        line = result.stdout.strip().split("\n")[0]
        parts = line.split("|")
        state_str = parts[0].strip() if parts else "UNKNOWN"
        exit_code = None
        elapsed = None

        if len(parts) > 1:
            try:
                exit_code = int(parts[1].split(":")[0])
            except (ValueError, IndexError):
                pass
        if len(parts) > 2:
            elapsed = self._parse_elapsed(parts[2])

        state_map = {
            "PENDING": "pending", "RUNNING": "running",
            "COMPLETED": "completed", "FAILED": "failed",
            "CANCELLED": "cancelled", "TIMEOUT": "failed",
            "NODE_FAIL": "failed", "PREEMPTED": "cancelled",
        }
        state = state_map.get(state_str, "pending")
        return JobStatus(job_id=job_id, state=state, exit_code=exit_code,
                         elapsed_seconds=elapsed)

    def cancel(self, job_id: str) -> None:
        subprocess.run(["scancel", job_id], capture_output=True)

    def get_log(self, job_id: str, tail: int = 50) -> str:
        """Read SLURM output file."""
        import glob
        pattern = f"slurm-{job_id}.out"
        matches = glob.glob(pattern)
        if not matches:
            return f"No log file found for job {job_id}"
        with open(matches[0]) as f:
            lines = f.readlines()
        return "".join(lines[-tail:])

    def _parse_elapsed(self, elapsed_str: str) -> float | None:
        """Parse SLURM elapsed time format (HH:MM:SS or D-HH:MM:SS) to seconds."""
        try:
            parts = elapsed_str.strip().split("-")
            if len(parts) == 2:
                days = int(parts[0])
                time_part = parts[1]
            else:
                days = 0
                time_part = parts[0]
            h, m, s = time_part.split(":")
            return days * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
        except (ValueError, IndexError):
            return None
