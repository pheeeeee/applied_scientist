"""
Debugger Agent — monitors failed jobs, diagnoses errors, suggests fixes.

The Debugger watches for job failures reported by Builder, reads SLURM logs,
identifies error patterns, and either applies automatic fixes or escalates
with a diagnosis to Builder/Orchestrator.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from applied_scientist.agents.base import BaseAgent
from applied_scientist.core.message_bus import AgentMessage
from applied_scientist.core.event_logger import EventLogger


@dataclass
class DiagnosisResult:
    """Result of diagnosing a job failure."""
    error_type: str                    # e.g., "import_error", "oom", "timeout"
    error_message: str                 # The actual error message found
    root_cause: str                    # Human-readable explanation
    severity: str                      # "auto_fixable", "needs_config_change", "needs_code_fix", "unknown"
    suggested_fix: Optional[str]       # Specific fix suggestion
    fix_config: Optional[dict]         # Config changes to apply (if auto_fixable)


# Known error patterns and their diagnoses
ERROR_PATTERNS = [
    # Import/Module errors
    {
        "pattern": r"ModuleNotFoundError: No module named ['\"](\w+)['\"]",
        "type": "import_error",
        "severity": "needs_config_change",
        "root_cause": "Python module '{0}' is not installed or not on PYTHONPATH",
        "fix": "Add the module to the conda environment or add its path to PYTHONPATH in setup_commands",
    },
    {
        "pattern": r"ModuleNotFoundError: No module named ['\"]applied_scientist['\"]",
        "type": "pythonpath_error",
        "severity": "auto_fixable",
        "root_cause": "applied_scientist package not on PYTHONPATH in compute environment",
        "fix": "Add 'export PYTHONPATH=/path/to/applied-scientist:$PYTHONPATH' to setup_commands",
    },
    {
        "pattern": r"ImportError: libcudart\.so\.(\d+\.\d+): cannot open shared object file",
        "type": "cuda_library_error",
        "severity": "needs_config_change",
        "root_cause": "CUDA {0} libraries not loaded. Need to load CUDA module before running.",
        "fix": "Add 'module load cuda/{0}' to setup_commands before conda activate",
    },

    # Memory errors
    {
        "pattern": r"CUDA out of memory\. Tried to allocate ([\d.]+ \w+)",
        "type": "cuda_oom",
        "severity": "needs_code_fix",
        "root_cause": "GPU memory exhausted trying to allocate {0}",
        "fix": "Reduce train_batch_size or model size in the experiment spec",
    },
    {
        "pattern": r"RuntimeError: CUDA error: out of memory",
        "type": "cuda_oom",
        "severity": "needs_code_fix",
        "root_cause": "GPU out of memory during operation",
        "fix": "Reduce batch size, use gradient checkpointing, or request larger GPU",
    },
    {
        "pattern": r"MemoryError|Cannot allocate memory|Killed.*oom",
        "type": "cpu_oom",
        "severity": "needs_config_change",
        "root_cause": "System ran out of CPU memory",
        "fix": "Increase mem allocation in SLURM config (e.g., mem: 64G)",
    },

    # SLURM errors
    {
        "pattern": r"sbatch: error:.*Invalid account",
        "type": "slurm_account_error",
        "severity": "needs_config_change",
        "root_cause": "SLURM account not specified or invalid",
        "fix": "Add 'account: gts-yourPI' to the slurm config section",
    },
    {
        "pattern": r"slurmstepd:.*CANCELLED.*TIME",
        "type": "slurm_timeout",
        "severity": "needs_config_change",
        "root_cause": "Job exceeded SLURM walltime limit",
        "fix": "Increase 'time' in SLURM config or reduce time_budget",
    },
    {
        "pattern": r"DUE TO NODE FAILURE",
        "type": "node_failure",
        "severity": "auto_fixable",
        "root_cause": "The compute node failed during execution (hardware issue)",
        "fix": "Automatic retry recommended - this is a transient infrastructure failure",
    },

    # Conda/environment errors
    {
        "pattern": r"conda.*EnvironmentNameNotFound.*['\"](\w+)['\"]",
        "type": "conda_env_missing",
        "severity": "needs_config_change",
        "root_cause": "Conda environment '{0}' does not exist on compute node",
        "fix": "Create the conda environment on compute nodes or fix the env name in setup_commands",
    },
    {
        "pattern": r"CommandNotFoundError: Your shell has not been properly configured",
        "type": "conda_init_error",
        "severity": "needs_config_change",
        "root_cause": "Conda not initialized in the shell",
        "fix": "Add 'source /path/to/conda.sh' before 'conda activate' in setup_commands",
    },

    # Ray/RLlib specific
    {
        "pattern": r"ray\.exceptions\.RaySystemError",
        "type": "ray_error",
        "severity": "needs_code_fix",
        "root_cause": "Ray system error - possibly worker crash or resource exhaustion",
        "fix": "Check Ray logs, reduce num_workers, or increase resources",
    },
    {
        "pattern": r"No such file or directory: ['\"](.+checkpoint.+)['\"]",
        "type": "checkpoint_missing",
        "severity": "needs_code_fix",
        "root_cause": "Checkpoint file not found at '{0}'",
        "fix": "Ensure checkpoint path is correct and checkpoint was saved properly",
    },

    # Generic Python errors
    {
        "pattern": r"TypeError: (\w+)\(\) got an unexpected keyword argument ['\"](\w+)['\"]",
        "type": "api_mismatch",
        "severity": "needs_code_fix",
        "root_cause": "Function {0} received unexpected argument '{1}' - likely API version mismatch",
        "fix": "Check library versions match expected API, update code to match installed version",
    },
    {
        "pattern": r"FileNotFoundError: \[Errno 2\] No such file or directory: ['\"](.+)['\"]",
        "type": "file_not_found",
        "severity": "needs_code_fix",
        "root_cause": "File not found: '{0}'",
        "fix": "Ensure the file exists and path is correct. Check if path is relative vs absolute.",
    },
]


class DebuggerAgent(BaseAgent):
    """Monitors failed jobs, diagnoses errors, and coordinates fixes.

    The Debugger is notified of failures by Builder and can:
    1. Parse SLURM logs to identify error patterns
    2. Diagnose root causes using known patterns + LLM reasoning
    3. Auto-retry transient failures (node failures, rate limits)
    4. Suggest config changes for environment issues
    5. Escalate to Builder for code-level fixes
    6. Record error patterns to knowledge base for future reference
    """

    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 results_tracker, job_runner, system_logger: EventLogger,
                 config, workspace: str):
        super().__init__("debugger", llm, tools, system_prompt, message_bus, cost_tracker)
        self.results = results_tracker
        self.runner = job_runner
        self.system_logger = system_logger
        self.config = config
        self.workspace = workspace

        # Track failures to avoid infinite retry loops
        self.failure_counts: dict[str, int] = {}  # experiment_name -> failure count
        self.max_retries = 2

        # Cache of recent diagnoses
        self.diagnosis_cache: dict[str, DiagnosisResult] = {}

    def run(self):
        """Main loop. Waits for failure notifications and processes them.

        Unlike other agents, Debugger has a longer sleep interval since job
        failures are relatively rare events. It only activates when Builder
        posts a 'job_failed' message to its inbox.
        """
        while not self._stopped:
            try:
                messages = self._handle_inbox()
                # Sleep longer if no messages (failures are rare)
                # Sleep shorter after processing to catch rapid failures
                if messages:
                    time.sleep(2)
                else:
                    time.sleep(30)  # Check every 30 seconds when idle
            except Exception as e:
                self._alert("error", f"Debugger error: {e}")
                time.sleep(30)

    def _handle_inbox(self) -> int:
        """Process failure notifications from Builder. Returns count of messages processed."""
        messages = self._process_inbox()
        for msg in messages:
            if msg.type == "job_failed":
                self._handle_job_failure(msg)
            elif msg.type == "diagnose_request":
                self._handle_diagnose_request(msg)
            elif msg.type == "command":
                action = msg.payload.get("action", "")
                if action == "clear_failure_counts":
                    self.failure_counts = {}
                    self.system_logger.log(event="debugger_failure_counts_cleared")
        return len(messages)

    def _handle_job_failure(self, msg: AgentMessage):
        """Diagnose a failed job and decide on action."""
        experiment_name = msg.payload.get("experiment_name", "unknown")
        job_id = msg.payload.get("job_id")
        log_dir = msg.payload.get("log_dir")
        error_summary = msg.payload.get("error_summary", "")

        self.system_logger.log(
            event="debugger_received_failure",
            experiment=experiment_name,
            job_id=job_id
        )

        # Get full log content
        log_content = self._read_log(job_id, log_dir)

        # Pattern-based diagnosis first (fast, no LLM cost)
        diagnosis = self._pattern_diagnosis(log_content, error_summary)

        # If pattern matching fails, use LLM for deeper analysis
        if diagnosis is None or diagnosis.error_type == "unknown":
            diagnosis = self._llm_diagnosis(experiment_name, log_content, error_summary)

        self.diagnosis_cache[experiment_name] = diagnosis

        # Log the diagnosis
        self.system_logger.log(
            event="debugger_diagnosis",
            experiment=experiment_name,
            error_type=diagnosis.error_type,
            severity=diagnosis.severity,
            root_cause=diagnosis.root_cause,
            suggested_fix=diagnosis.suggested_fix
        )

        # Decide on action based on severity
        self._take_action(experiment_name, diagnosis, msg.payload)

    def _read_log(self, job_id: str, log_dir: str) -> str:
        """Read SLURM log file content."""
        if not job_id:
            return ""

        # Try log_dir first
        if log_dir:
            log_path = os.path.join(log_dir, f"slurm-{job_id}.out")
            if os.path.exists(log_path):
                try:
                    with open(log_path) as f:
                        return f.read()
                except IOError:
                    pass

        # Fallback to runner's get_log
        try:
            return self.runner.get_log(job_id, log_dir=log_dir, tail=200)
        except Exception:
            return ""

    def _pattern_diagnosis(self, log_content: str, error_summary: str) -> Optional[DiagnosisResult]:
        """Try to diagnose using known error patterns."""
        combined = f"{error_summary}\n{log_content}"

        for pattern_info in ERROR_PATTERNS:
            match = re.search(pattern_info["pattern"], combined, re.IGNORECASE | re.MULTILINE)
            if match:
                # Format root cause and fix with captured groups
                groups = match.groups() if match.groups() else []
                root_cause = pattern_info["root_cause"]
                fix = pattern_info["fix"]

                for i, g in enumerate(groups):
                    root_cause = root_cause.replace(f"{{{i}}}", str(g))
                    fix = fix.replace(f"{{{i}}}", str(g))

                return DiagnosisResult(
                    error_type=pattern_info["type"],
                    error_message=match.group(0),
                    root_cause=root_cause,
                    severity=pattern_info["severity"],
                    suggested_fix=fix,
                    fix_config=None
                )

        return None

    def _llm_diagnosis(self, experiment_name: str, log_content: str, error_summary: str) -> DiagnosisResult:
        """Use LLM to diagnose unknown errors."""
        # Truncate log to avoid context overflow
        log_excerpt = log_content[-8000:] if len(log_content) > 8000 else log_content

        prompt = f"""Analyze this failed experiment and diagnose the root cause.

Experiment: {experiment_name}
Error summary: {error_summary}

Log content (last portion):
```
{log_excerpt}
```

Respond with a JSON object:
{{
    "error_type": "brief_category",
    "error_message": "the actual error line",
    "root_cause": "explanation of why it failed",
    "severity": "auto_fixable|needs_config_change|needs_code_fix|unknown",
    "suggested_fix": "specific actionable fix"
}}

Severity guide:
- auto_fixable: transient failures (node crash, network blip) - safe to retry
- needs_config_change: environment/SLURM config issue - needs user to fix config
- needs_code_fix: bug in training code - needs Builder to fix code
- unknown: can't determine - needs manual investigation
"""

        try:
            response = self.chat(prompt)
            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return DiagnosisResult(
                    error_type=data.get("error_type", "unknown"),
                    error_message=data.get("error_message", ""),
                    root_cause=data.get("root_cause", "Unable to determine"),
                    severity=data.get("severity", "unknown"),
                    suggested_fix=data.get("suggested_fix"),
                    fix_config=None
                )
        except (json.JSONDecodeError, Exception) as e:
            self.system_logger.log(event="debugger_llm_parse_error", error=str(e))

        return DiagnosisResult(
            error_type="unknown",
            error_message=error_summary,
            root_cause="Could not determine root cause from logs",
            severity="unknown",
            suggested_fix="Manual investigation required. Check the SLURM log file.",
            fix_config=None
        )

    def _take_action(self, experiment_name: str, diagnosis: DiagnosisResult, payload: dict):
        """Take action based on diagnosis severity."""

        # Track failure count
        self.failure_counts[experiment_name] = self.failure_counts.get(experiment_name, 0) + 1

        if diagnosis.severity == "auto_fixable":
            # Transient failure - retry if under limit
            if self.failure_counts[experiment_name] <= self.max_retries:
                self._notify_retry(experiment_name, diagnosis, payload)
            else:
                self._escalate(experiment_name, diagnosis,
                              f"Exceeded max retries ({self.max_retries}) for transient failure")

        elif diagnosis.severity == "needs_config_change":
            # Config issue - notify user via orchestrator
            self._escalate_to_user(experiment_name, diagnosis)

        elif diagnosis.severity == "needs_code_fix":
            # Code bug - notify Builder
            self._escalate_to_builder(experiment_name, diagnosis, payload)

        else:
            # Unknown - escalate to user
            self._escalate_to_user(experiment_name, diagnosis)

    def _notify_retry(self, experiment_name: str, diagnosis: DiagnosisResult, payload: dict):
        """Tell Builder to retry the experiment."""
        self.bus.post(AgentMessage(
            sender="debugger",
            recipient="builder",
            type="retry_experiment",
            payload={
                "experiment_name": experiment_name,
                "spec_path": payload.get("spec_path"),
                "seed": payload.get("seed"),
                "reason": f"Auto-retry: {diagnosis.root_cause}",
                "attempt": self.failure_counts[experiment_name]
            }
        ))
        self.system_logger.log(
            event="debugger_requested_retry",
            experiment=experiment_name,
            attempt=self.failure_counts[experiment_name],
            reason=diagnosis.root_cause
        )

    def _escalate_to_builder(self, experiment_name: str, diagnosis: DiagnosisResult, payload: dict):
        """Escalate code fix to Builder."""
        self.bus.post(AgentMessage(
            sender="debugger",
            recipient="builder",
            type="fix_required",
            payload={
                "experiment_name": experiment_name,
                "spec_path": payload.get("spec_path"),
                "error_type": diagnosis.error_type,
                "error_message": diagnosis.error_message,
                "root_cause": diagnosis.root_cause,
                "suggested_fix": diagnosis.suggested_fix
            }
        ))
        self.system_logger.log(
            event="debugger_escalated_to_builder",
            experiment=experiment_name,
            error_type=diagnosis.error_type
        )

    def _escalate_to_user(self, experiment_name: str, diagnosis: DiagnosisResult):
        """Escalate to user via Orchestrator."""
        message = f"""
**Job Failed: {experiment_name}**

**Error Type:** {diagnosis.error_type}
**Root Cause:** {diagnosis.root_cause}

**Error Message:**
```
{diagnosis.error_message[:500]}
```

**Suggested Fix:** {diagnosis.suggested_fix}
"""
        self.bus.post(AgentMessage(
            sender="debugger",
            recipient="orchestrator",
            type="alert",
            payload={
                "alert_type": "job_failure",
                "experiment_name": experiment_name,
                "message": message
            }
        ))
        self.system_logger.log(
            event="debugger_escalated_to_user",
            experiment=experiment_name,
            error_type=diagnosis.error_type
        )

    def _escalate(self, experiment_name: str, diagnosis: DiagnosisResult, reason: str):
        """Generic escalation with custom reason."""
        self.bus.post(AgentMessage(
            sender="debugger",
            recipient="orchestrator",
            type="alert",
            payload={
                "alert_type": "job_failure",
                "experiment_name": experiment_name,
                "message": f"{reason}\n\nDiagnosis: {diagnosis.root_cause}\nFix: {diagnosis.suggested_fix}"
            }
        ))

    def _handle_diagnose_request(self, msg: AgentMessage):
        """Handle a request to diagnose a specific job (from Orchestrator)."""
        job_id = msg.payload.get("job_id")
        log_dir = msg.payload.get("log_dir")

        log_content = self._read_log(job_id, log_dir)
        diagnosis = self._pattern_diagnosis(log_content, "")

        if diagnosis is None:
            diagnosis = self._llm_diagnosis(f"job_{job_id}", log_content, "")

        # Send diagnosis back to requester
        self.bus.post(AgentMessage(
            sender="debugger",
            recipient=msg.sender,
            type="diagnosis_result",
            payload={
                "job_id": job_id,
                "error_type": diagnosis.error_type,
                "root_cause": diagnosis.root_cause,
                "severity": diagnosis.severity,
                "suggested_fix": diagnosis.suggested_fix
            }
        ))

    def get_diagnosis(self, experiment_name: str) -> Optional[DiagnosisResult]:
        """Get cached diagnosis for an experiment."""
        return self.diagnosis_cache.get(experiment_name)

    def get_failure_stats(self) -> dict:
        """Get failure statistics."""
        return {
            "total_failures": sum(self.failure_counts.values()),
            "experiments_failed": len(self.failure_counts),
            "failure_counts": dict(self.failure_counts),
            "cached_diagnoses": len(self.diagnosis_cache)
        }
