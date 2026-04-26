from __future__ import annotations

import os
import subprocess
from datetime import datetime

from applied_scientist.compute.base import JobRunner, JobStatus


class SLURMRunner(JobRunner):
    """SLURM compute backend: sbatch, sacct, scancel.

    Generates sbatch scripts with proper environment setup and submits jobs.
    Scripts are saved to workspace/.scripts/ for debugging.
    """

    def __init__(self, partition: str = "gpu", gres: str = "gpu:1",
                 mem: str = "48G", time: str = "01:00:00",
                 setup_commands: list[str] | None = None,
                 account: str | None = None,
                 workspace: str | None = None):
        """
        Args:
            partition: SLURM partition name (e.g., "gpu-v100")
            gres: GPU resource specification (e.g., "gpu:V100:1")
            mem: Memory allocation (e.g., "48G")
            time: Walltime limit (e.g., "02:00:00")
            setup_commands: Shell commands to run before training (conda activate, etc.)
            account: SLURM account for billing (required on some clusters, e.g., "gts-mypi")
            workspace: Path to workspace directory (for script persistence and logs)
        """
        self.partition = partition
        self.gres = gres
        self.mem = mem
        self.time = time
        self.setup_commands = setup_commands or []
        self.account = account
        self.workspace = workspace

        # Create scripts directory if workspace is provided
        if self.workspace:
            self._scripts_dir = os.path.join(self.workspace, ".scripts")
            os.makedirs(self._scripts_dir, exist_ok=True)
        else:
            self._scripts_dir = None

    def submit(self, command: str, job_name: str,
               resources: dict | None = None,
               log_dir: str | None = None) -> str:
        """Submit via sbatch --parsable. Returns job ID string.

        Args:
            command: The command to execute
            job_name: Name for the SLURM job
            resources: Optional resource overrides (not yet implemented)
            log_dir: Directory for SLURM output file (defaults to CWD)
        """
        # Determine output path
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            output_path = os.path.join(log_dir, "slurm-%j.out")
        else:
            output_path = "slurm-%j.out"

        # Build script header
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --gres={self.gres}",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --output={output_path}",
            f"#SBATCH --error={output_path}",
        ]

        # Add account if specified (required on PACE and similar clusters)
        if self.account:
            script_lines.append(f"#SBATCH --account={self.account}")

        # Add diagnostic header
        script_lines.extend([
            "",
            "# === Job Info ===",
            'echo "=========================================="',
            'echo "Job ID: $SLURM_JOB_ID"',
            'echo "Job Name: $SLURM_JOB_NAME"',
            'echo "Node: $SLURMD_NODENAME"',
            'echo "Started: $(date)"',
            'echo "=========================================="',
            "",
            "# === Environment Setup ===",
        ])

        # Add setup commands
        for cmd in self.setup_commands:
            script_lines.append(cmd)

        # Add verification steps
        script_lines.extend([
            "",
            "# === Verify Environment ===",
            'echo "Python: $(which python)"',
            'echo "Conda env: $CONDA_DEFAULT_ENV"',
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU info unavailable'",
            "",
            "# === Run Command ===",
            "set -e  # Exit on error",
        ])

        script_lines.append(command)

        # Add completion marker
        script_lines.extend([
            "",
            'echo "=========================================="',
            'echo "Finished: $(date)"',
            'echo "Exit code: $?"',
            'echo "=========================================="',
        ])

        script_content = "\n".join(script_lines) + "\n"

        # Save script to workspace for debugging (or use temp file)
        if self._scripts_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_path = os.path.join(self._scripts_dir, f"{job_name}_{timestamp}.sh")
            with open(script_path, "w") as f:
                f.write(script_content)
        else:
            import tempfile
            fd, script_path = tempfile.mkstemp(suffix=".sh")
            with os.fdopen(fd, "w") as f:
                f.write(script_content)

        # Submit job
        result = subprocess.run(
            ["sbatch", "--parsable", script_path],
            capture_output=True, text=True)

        # Clean up temp file (but keep workspace scripts for debugging)
        if not self._scripts_dir:
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

    def get_log(self, job_id: str, log_dir: str | None = None, tail: int = 50) -> str:
        """Read SLURM output file.

        Args:
            job_id: The SLURM job ID
            log_dir: Directory where log file is located (if known)
            tail: Number of lines to return from end of file
        """
        import glob

        # Search in log_dir first, then CWD
        search_paths = []
        if log_dir:
            search_paths.append(os.path.join(log_dir, f"slurm-{job_id}.out"))
        search_paths.append(f"slurm-{job_id}.out")

        # Also search workspace if available
        if self.workspace:
            search_paths.append(
                os.path.join(self.workspace, "results", "logs", "*", f"slurm-{job_id}.out"))

        for pattern in search_paths:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0]) as f:
                    lines = f.readlines()
                return "".join(lines[-tail:])

        return f"No log file found for job {job_id}. Searched: {search_paths}"

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
