from __future__ import annotations

import json
import os
import re
import time

from applied_scientist.agents.base import BaseAgent
from applied_scientist.core.message_bus import (
    AgentMessage, PRIORITY_CODE_REVIEW, PRIORITY_PLAN_REVIEW,
    PRIORITY_INSIGHT_REVIEW, PRIORITY_SUGGESTION, PRIORITY_RERANK,
)
from applied_scientist.core.spec import ExperimentSpec
from applied_scientist.core.results_tracker import ExperimentResult
from applied_scientist.core.gpu_pool import GPUSlot
from applied_scientist.core.event_logger import EventLogger
from applied_scientist.core.utils import atomic_write


class BuilderAgent(BaseAgent):
    """GPU pool manager: implement, validate, submit, monitor, insight."""

    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 priority_queue, knowledge_base, results_tracker,
                 gpu_pool, job_runner, task_adapter, system_logger: EventLogger,
                 config, approve_plans: bool = False, approve_submit: bool = False):
        super().__init__("builder", llm, tools, system_prompt, message_bus, cost_tracker)
        self.queue = priority_queue
        self.kb = knowledge_base
        self.results = results_tracker
        self.pool = gpu_pool
        self.runner = job_runner
        self.task = task_adapter
        self.system_logger = system_logger
        self.config = config
        self.paused = False
        self.pending_pairs: dict[str, list[str]] = {}
        self._last_validation_error: str = ""
        self._consecutive_failures: int = 0
        self._last_failure_error: str = ""
        self._edited_files: set[str] = set()
        # Human approval flags
        self.approve_plans = approve_plans
        self.approve_submit = approve_submit

    def run(self):
        """Main loop. Manages GPU pool with non-blocking reviews."""
        self._fill_free_slots()

        while not self._stopped:
            try:
                if self.paused:
                    time.sleep(5)
                    continue

                self._handle_inbox()
                self._monitor_running_slots()
                self._fill_free_slots()
                self._persist_pairs()

                time.sleep(10)

            except Exception as e:
                self._alert("error", f"Builder error: {e}")
                time.sleep(10)

    def _handle_inbox(self):
        """Process Critic and Orchestrator responses."""
        for msg in self._process_inbox():
            if msg.type == "code_rejected":
                self._handle_code_rejection(msg)
            elif msg.type == "code_approved":
                pass  # job already submitted optimistically
            elif msg.type in ("plan_approved", "plan_rejected"):
                self.bus.post(msg)  # re-inject for _wait_for_message to consume
            elif msg.type == "insight_approved":
                self.kb.add_insight(msg.payload["experiment"], msg.payload["insight"])
            elif msg.type == "insight_rejected":
                pass  # optionally revise, but don't block
            elif msg.type == "suggestion_triaged":
                self._handle_triaged_suggestion(msg)
            elif msg.type == "clarification_response":
                pass
            elif msg.type == "command":
                action = msg.payload.get("action", "")
                if action == "pause":
                    self.paused = True
                    self.system_logger.log(event="builder_paused", reason="user_command")
                elif action == "resume":
                    self.paused = False
                    self._consecutive_failures = 0
                    self.system_logger.log(event="builder_resumed", reason="user_command")

    def _monitor_running_slots(self):
        """Check all running slots for completion, failure, timeout, early stop."""
        for slot in self.pool.get_running():
            status = self.runner.status(slot.job_id)

            if status.state == "completed":
                self._handle_completion(slot)
            elif status.state == "failed":
                self._handle_failure(slot)
            elif status.state in ("cancelled",):
                self.pool.release(slot.slot_id)
            elif status.state == "running":
                if self._check_early_stop(slot):
                    self.runner.cancel(slot.job_id)
                    self._handle_early_stop(slot)
                elif self._check_timeout(slot):
                    self.runner.cancel(slot.job_id)
                    self._handle_timeout(slot)

        for slot in self.pool.get_timed_out(
                self.config.system.time_budget * self.config.system.timeout_multiplier):
            self.runner.cancel(slot.job_id)
            self._handle_timeout(slot)

    def _check_early_stop(self, slot: GPUSlot) -> bool:
        """Check progress file for early stopping signal."""
        elapsed = time.time() - slot.submitted_at
        if elapsed < self.config.system.time_budget * self.config.system.early_stop_fraction:
            return False

        progress_path = os.path.join(slot.log_dir, self.task.progress_file)
        if not os.path.exists(progress_path):
            return False

        try:
            with open(progress_path) as f:
                progress = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False

        metric_name = self.task.metric[0]
        current = progress.get(metric_name)
        if current is None:
            return False

        baselines = self.results.get_baselines()
        if not baselines:
            return False

        baseline_values = [r.metric_value for r in baselines if r.metric_value is not None]
        if not baseline_values:
            return False

        best_baseline = max(baseline_values)
        threshold = best_baseline * 0.1
        if self.task.metric[1] == "higher":
            return current < threshold
        else:
            return current > best_baseline * 10 if best_baseline > 0 else False

    def _check_timeout(self, slot: GPUSlot) -> bool:
        elapsed = time.time() - slot.submitted_at
        return elapsed > self.config.system.time_budget * self.config.system.timeout_multiplier

    def _fill_free_slots(self):
        """Fill all free GPU slots with top-ranked specs."""
        while True:
            slot = self.pool.get_free_slot()
            if slot is None:
                break
            item = self.queue.pop()
            if item is None:
                break
            spec, score = item

            try:
                self._implement_and_submit(spec, slot)
            except Exception as e:
                self._alert("error", f"Failed to implement {spec.name}: {e}")
                self.pool.release(slot.slot_id)

    def _implement_and_submit(self, spec: ExperimentSpec, slot: GPUSlot):
        """Plan -> Critic review -> Implement -> Validate -> Submit.

        Non-baseline experiments get self-contained directories under
        workspace/experiments/<name>/. Baseline experiments use the adapter's train.py.
        """
        self._track_pair(spec)
        self._edited_files.clear()

        is_baseline = spec.category == "baseline"
        seeds = self.config.system.baseline_seeds if is_baseline else 1

        # ── Baseline: unchanged flow (uses adapter's train.py) ──
        if is_baseline:
            if seeds > 1:
                self._run_baseline_seeds(spec, slot, seeds)
            else:
                log_dir = os.path.join(self.config.paths.logs, spec.name)
                os.makedirs(log_dir, exist_ok=True)
                command = self._build_train_command(spec, seed=1)
                scripts_dir = os.path.join(self.config.paths.workspace, "scripts")
                job_id = self.runner.submit(command, spec.name, scripts_dir=scripts_dir)
                self.pool.assign(slot.slot_id, job_id, spec.name, log_dir)
                self.system_logger.log(event="experiment_submitted",
                                       spec=spec.name, job_id=job_id)
            self.reset_conversation()
            return

        # ── Non-baseline: self-contained experiment directory flow ──

        # Step 1: Create experiment directory and save artifacts
        experiment_dir = self._make_experiment_dir(spec)
        spec.to_yaml(os.path.join(experiment_dir, "spec.yaml"))
        self._save_queue_snapshot(experiment_dir)

        # Step 2: Write implementation plan (LLM chat)
        plan_text = self._write_implementation_plan(spec, experiment_dir)

        # Step 2.5: Human approval of plan (if --approve-plans flag)
        if self.approve_plans:
            plan_path = os.path.join(experiment_dir, "PLAN.md")
            if not self._request_human_approval("plan", spec.name, plan_path):
                self._alert("info", f"Plan for {spec.name} rejected by human")
                self.pool.release(slot.slot_id)
                self.reset_conversation()
                return

        # Step 3: Critic reviews plan (BLOCKING)
        approved = self._wait_for_plan_review(spec, plan_text, experiment_dir)
        if not approved:
            self._alert("error", f"Plan for {spec.name} rejected after max rounds")
            self.pool.release(slot.slot_id)
            self.reset_conversation()
            return

        # Step 4a: Read reference code (baseline train.py) before implementing
        spec_content = spec.to_yaml()
        code_map = self.task.code_map if self.task.code_map else ""
        rel_dir = os.path.relpath(experiment_dir, self.config.paths.workspace)
        task_module = self.config.task.get("module", "")

        self.chat(f"Read the approved plan: read_file {rel_dir}/PLAN.md")
        if task_module:
            self.chat(
                f"Read the working baseline training script as reference. "
                f"Study how it handles CLI args, progress.json, Ray init, and "
                f"final JSON output — your train.py must follow the same contract.\n\n"
                f"read_file {task_module}/train.py"
            )

        # Step 4b: Implement all files
        self.chat(
            f"## Implement the experiment from the approved plan\n\n"
            f"```yaml\n{spec_content}\n```\n\n"
            f"## Task context\n{self.task.domain_context}\n\n"
            f"## Reference code map\n{code_map}\n\n"
            f"Implement ALL files listed in the PLAN.md into the directory:\n"
            f"  {rel_dir}/\n\n"
            f"Requirements:\n"
            f"- train.py MUST accept: --seed, --time-budget, --log-path, --checkpoint-dir\n"
            f"- train.py MUST write progress.json to --log-path periodically\n"
            f"- train.py MUST print final result as JSON on the last line\n"
            f"- All imports must be self-contained (no imports from tasks/examples/ models)\n"
            f"- Environment: Python 3.8, Ray 1.4.0, PyTorch 1.8.1\n\n"
            f"Implement ONE FILE AT A TIME. For each file:\n"
            f"1. Write the complete file using write_file\n"
            f"2. Review what you wrote — check imports, signatures, and connections\n"
            f"3. Move to the next file\n\n"
            f"After writing ALL files, read back train.py and verify it matches the plan."
        )

        # Track edited files
        for msg in self.conversation:
            if msg.tool_calls:
                for call in msg.tool_calls:
                    if call["name"] in ("edit_file", "write_file"):
                        path = call["arguments"].get("path",
                               call["arguments"].get("file_path", ""))
                        if path:
                            self._edited_files.add(path)

        # Step 5: Validate with iterative fixing (up to 3 attempts)
        max_fix_attempts = 3
        for attempt in range(max_fix_attempts):
            if self._validate_experiment_dir(spec, experiment_dir):
                break
            error = self._last_validation_error
            if attempt == max_fix_attempts - 1:
                self._record_validation_fail(spec, slot)
                self.reset_conversation()
                return
            self.chat(
                f"Pre-flight validation failed (attempt {attempt + 1}/{max_fix_attempts}):\n"
                f"{error}\n\n"
                f"Read the failing file, diagnose the root cause, and fix it. "
                f"Do not guess — read the actual code first."
            )

        # Step 6: Code review (non-blocking)
        diff = self._get_git_diff()
        if self._is_structural_change(diff):
            self.bus.post(AgentMessage(
                sender="builder", recipient="critic",
                type="code_review_request",
                payload={"spec_name": spec.name, "diff": diff},
                priority=PRIORITY_CODE_REVIEW,
            ))

        # Step 7: Git commit
        self.chat("Commit the changes with git_commit tool.")

        # Step 7.5: Human approval before GPU submission (if --approve-submit flag)
        if self.approve_submit:
            train_path = os.path.join(experiment_dir, "train.py")
            print(f"\nReady to submit {spec.name} to GPU.")
            print(f"  Experiment dir: {experiment_dir}")
            print(f"  Time budget: {self.config.system.time_budget}s")
            if not self._request_human_approval("submit", spec.name, train_path):
                self._alert("info", f"Submission of {spec.name} rejected by human")
                self.pool.release(slot.slot_id)
                self.reset_conversation()
                return

        # Step 8: Submit to SLURM (logs go inside experiment directory)
        log_dir = os.path.join(experiment_dir, "logs")
        command = self._build_train_command_experiment_dir(spec, experiment_dir, seed=1)
        job_id = self.runner.submit(command, spec.name, scripts_dir=experiment_dir)
        self.pool.assign(slot.slot_id, job_id, spec.name, log_dir)

        self.system_logger.log(event="experiment_submitted", spec=spec.name,
                               job_id=job_id, experiment_dir=experiment_dir)
        self.reset_conversation()

    # ── New methods for self-contained experiment directories ──

    def _make_experiment_dir(self, spec: ExperimentSpec) -> str:
        """Create and return path to workspace/experiments/<spec.name>/.

        Creates the full directory structure:
            experiments/<name>/
            ├── logs/           # training logs, progress.json, slurm output
            ├── checkpoints/    # model weights
            └── errors/         # validation errors, crash tracebacks
        """
        exp_dir = os.path.join(self.config.paths.workspace, "experiments", spec.name)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "errors"), exist_ok=True)
        return exp_dir

    def _save_queue_snapshot(self, experiment_dir: str):
        """Save current queue state when pulling an experiment for debugging."""
        snapshot_path = os.path.join(experiment_dir, "queue_snapshot.yaml")
        try:
            import yaml
            items = self.queue.peek(20)  # top 20 for context
            snapshot = [{
                "name": spec.name,
                "score": score,
                "description": spec.description,
                "category": spec.category,
            } for spec, score in items]
            atomic_write(snapshot_path, yaml.dump(snapshot, default_flow_style=False))
        except Exception:
            pass  # non-critical, don't fail experiment

    def _write_implementation_plan(self, spec: ExperimentSpec,
                                    experiment_dir: str) -> str:
        """Chat with LLM to create PLAN.md. Returns the plan text."""
        spec_content = spec.to_yaml()
        code_map = self.task.code_map if self.task.code_map else ""
        rel_dir = os.path.relpath(experiment_dir, self.config.paths.workspace)

        plan_text = self.chat(
            f"## Write an implementation plan for this experiment\n\n"
            f"```yaml\n{spec_content}\n```\n\n"
            f"## Task context\n{self.task.domain_context}\n\n"
            f"## Reference code map (existing task structure)\n{code_map}\n\n"
            f"## Instructions\n"
            f"You will implement this as a SELF-CONTAINED experiment directory at:\n"
            f"  {rel_dir}/\n\n"
            f"Write a PLAN.md that lists:\n"
            f"1. **Every file** you will create and its purpose\n"
            f"2. **Key classes and functions** in each file with signatures\n"
            f"3. **How the files connect** (imports, call chains)\n"
            f"4. **train.py contract**: Must accept CLI args --seed, --time-budget, "
            f"--log-path, --checkpoint-dir. Must write progress.json periodically. "
            f"Must print final result as JSON on last line.\n"
            f"5. **Dependencies**: What packages/modules are imported\n"
            f"6. **Environment constraints**: Python 3.8, Ray 1.4.0, PyTorch 1.8.1\n\n"
            f"The experiment directory must be completely self-contained. "
            f"Do NOT import from tasks/examples/ model files. "
            f"You MAY import standard libraries, ray, torch, gym, and soccer_twos.\n\n"
            f"Output ONLY the PLAN.md content. Do not use any tools yet."
        )

        # Write PLAN.md to the experiment directory
        self.chat(
            f"Write the plan to {rel_dir}/PLAN.md using write_file tool:\n\n{plan_text}"
        )

        return plan_text

    def _wait_for_plan_review(self, spec: ExperimentSpec, plan_text: str,
                               experiment_dir: str) -> bool:
        """Submit plan for Critic review, block until response.
        Returns True if approved, False if rejected after max rounds.
        """
        max_rounds = self.config.system.max_spec_review_rounds
        spec_content = spec.to_yaml()
        current_plan = plan_text
        rel_dir = os.path.relpath(experiment_dir, self.config.paths.workspace)

        for round_num in range(1, max_rounds + 1):
            self.bus.post(AgentMessage(
                sender="builder", recipient="critic",
                type="plan_review_request",
                payload={
                    "spec_name": spec.name,
                    "plan_content": current_plan,
                    "spec_content": spec_content,
                    "round": round_num,
                },
                priority=PRIORITY_PLAN_REVIEW,
            ))

            # Block until Critic responds (approval or rejection)
            response = self._wait_for_message(
                ("plan_approved", "plan_rejected"),
                match_fn=lambda m: m.payload.get("spec_name") == spec.name,
                timeout=300.0,
            )

            if response is None:
                self.system_logger.log(
                    event="plan_review_timeout", spec=spec.name, round=round_num)
                if round_num == max_rounds:
                    return True  # force-approve on final timeout
                continue

            if response.type == "plan_approved":
                self.system_logger.log(
                    event="plan_approved", spec=spec.name, round=round_num)
                return True

            # Handle rejection — revise the plan
            feedback = response.payload.get("feedback", "")
            missing = response.payload.get("missing_files", [])
            concerns = response.payload.get("concerns", [])
            self.system_logger.log(
                event="plan_rejected", spec=spec.name, round=round_num,
                feedback=feedback[:200])

            revision_prompt = (
                f"## Revise the implementation plan (round {round_num + 1}/{max_rounds})\n\n"
                f"The Critic rejected the plan with this feedback:\n{feedback}\n\n"
            )
            if missing:
                revision_prompt += f"Missing files: {', '.join(missing)}\n\n"
            if concerns:
                revision_prompt += "Concerns:\n" + "\n".join(
                    f"- {c}" for c in concerns) + "\n\n"
            revision_prompt += (
                "Revise the plan to address ALL feedback. "
                "Output the complete revised PLAN.md content."
            )

            current_plan = self.chat(revision_prompt)
            self.chat(
                f"Update the plan file at {rel_dir}/PLAN.md "
                f"using write_file tool:\n\n{current_plan}"
            )

        self.system_logger.log(
            event="plan_force_approved", spec=spec.name,
            reason="max_review_rounds_exhausted")
        return True

    def _validate_experiment_dir(self, spec: ExperimentSpec,
                                  experiment_dir: str) -> bool:
        """Pre-flight validation for self-contained experiment directory.

        Captures all validation output to logs/validation.log for debugging.
        """
        validation_log_path = os.path.join(experiment_dir, "logs", "validation.log")
        validation_output = []

        train_path = os.path.join(experiment_dir, "train.py")
        if not os.path.exists(train_path):
            self._last_validation_error = f"train.py not found in {experiment_dir}/"
            self._write_validation_log(validation_log_path, [self._last_validation_error])
            return False

        # Syntax check all .py files
        validation_output.append("=== SYNTAX CHECK ===")
        self.chat(
            f"Run a syntax check using run_command: "
            f"python -c \"import py_compile; import glob; "
            f"[py_compile.compile(f, doraise=True) for f in "
            f"glob.glob('{experiment_dir}/*.py')]\""
        )
        cmd_result = self._parse_last_tool_result("run_command")
        validation_output.append(cmd_result or "(no output)")

        if cmd_result is None:
            self._last_validation_error = "Syntax check was not executed (tool not called)"
            self._write_validation_log(validation_log_path, validation_output + [self._last_validation_error])
            return False
        if "Error" in cmd_result or "Traceback" in cmd_result:
            self._last_validation_error = cmd_result
            self._write_validation_log(validation_log_path, validation_output)
            return False

        # 10-second smoke test (runs on head node, keep short)
        validation_output.append("\n=== SMOKE TEST ===")
        smoke_cmd = self._build_train_command_experiment_dir(
            spec, experiment_dir, seed=0, time_budget=10)
        validation_output.append(f"Command: {smoke_cmd}")
        self.chat(f"Run a quick smoke test using run_command:\n{smoke_cmd}")
        cmd_result = self._parse_last_tool_result("run_command")
        validation_output.append(cmd_result or "(no output)")

        if cmd_result is None:
            self._last_validation_error = "Smoke test was not executed (tool not called)"
            self._write_validation_log(validation_log_path, validation_output + [self._last_validation_error])
            return False
        if "Error" in cmd_result or "Traceback" in cmd_result or "crash" in cmd_result.lower():
            self._last_validation_error = cmd_result
            self._write_validation_log(validation_log_path, validation_output)
            return False

        validation_output.append("\n=== VALIDATION PASSED ===")
        self._write_validation_log(validation_log_path, validation_output)
        return True

    def _write_validation_log(self, path: str, lines: list[str]):
        """Write validation output to log file."""
        try:
            content = "\n".join(str(line) for line in lines)
            atomic_write(path, content)
        except Exception:
            pass  # non-critical

    def _request_human_approval(self, checkpoint: str, spec_name: str,
                                 file_path: str | None = None) -> bool:
        """Request human approval at a checkpoint. Returns True if approved.

        Args:
            checkpoint: Name of checkpoint (e.g., "plan", "submit")
            spec_name: Name of the experiment spec
            file_path: Optional path to file for review

        Returns:
            True if approved, False if rejected
        """
        prompt_text = f"\n{'='*60}\n"
        prompt_text += f"APPROVAL REQUIRED: {checkpoint.upper()}\n"
        prompt_text += f"Experiment: {spec_name}\n"
        if file_path:
            prompt_text += f"Review: {file_path}\n"
        prompt_text += f"{'='*60}\n"
        prompt_text += "[a]pprove / [r]eject / [v]iew file? "

        while True:
            try:
                print(prompt_text, end="", flush=True)
                response = input().strip().lower()

                if response in ("a", "approve", "y", "yes"):
                    self.system_logger.log(
                        event="human_approval",
                        checkpoint=checkpoint,
                        spec=spec_name,
                        decision="approved",
                    )
                    print(f"  Approved by human.\n")
                    return True

                elif response in ("r", "reject", "n", "no"):
                    self.system_logger.log(
                        event="human_approval",
                        checkpoint=checkpoint,
                        spec=spec_name,
                        decision="rejected",
                    )
                    print(f"  Rejected by human.\n")
                    return False

                elif response in ("v", "view") and file_path:
                    try:
                        with open(file_path) as f:
                            content = f.read()
                        print(f"\n--- {file_path} ---\n{content}\n--- END ---\n")
                    except Exception as e:
                        print(f"  Error reading file: {e}\n")

                else:
                    print("  Invalid input. Enter 'a' to approve, 'r' to reject, 'v' to view.\n")

            except EOFError:
                # Non-interactive mode - auto-approve
                self.system_logger.log(
                    event="human_approval",
                    checkpoint=checkpoint,
                    spec=spec_name,
                    decision="auto_approved_non_interactive",
                )
                return True

    def _build_train_command_experiment_dir(self, spec: ExperimentSpec,
                                             experiment_dir: str,
                                             seed: int,
                                             time_budget: int | None = None) -> str:
        """Build training command for self-contained experiment directory.

        All outputs go inside the experiment directory:
            experiments/<name>/logs/          - training logs, progress.json
            experiments/<name>/checkpoints/   - model weights
        """
        tb = time_budget or self.config.system.time_budget
        # Keep logs and checkpoints inside experiment directory
        log_dir = os.path.join(experiment_dir, "logs")
        ckpt_dir = os.path.join(experiment_dir, "checkpoints")
        train_script = os.path.join(experiment_dir, "train.py")
        return (
            f"python {train_script} "
            f"--seed {seed} "
            f"--time-budget {tb} "
            f"--log-path {log_dir} "
            f"--checkpoint-dir {ckpt_dir}"
        )

    def _validate_implementation(self, spec: ExperimentSpec) -> bool:
        """Pre-flight: syntax check + 60s smoke test.
        Checks actual tool output, not LLM conversational response."""
        self.chat(
            "Run a syntax check on the modified files using run_command: "
            "python -c 'import applied_scientist'"
        )
        cmd_result = self._parse_last_tool_result("run_command")
        if cmd_result and ("Error" in cmd_result or "Traceback" in cmd_result):
            self._last_validation_error = cmd_result
            return False

        smoke_cmd = self._build_train_command(spec, seed=0, time_budget=60)
        self.chat(f"Run a 60-second smoke test using run_command:\n{smoke_cmd}")
        cmd_result = self._parse_last_tool_result("run_command")
        if cmd_result and ("Error" in cmd_result or "Traceback" in cmd_result
                           or "crash" in cmd_result.lower()):
            self._last_validation_error = cmd_result
            return False
        return True

    def _parse_last_tool_result(self, tool_name: str) -> str | None:
        """Find the last tool result from a specific tool in conversation."""
        for msg in reversed(self.conversation):
            if msg.role == "tool_result" and msg.tool_call_id:
                for prev_msg in self.conversation:
                    if prev_msg.tool_calls:
                        for call in prev_msg.tool_calls:
                            if (call["id"] == msg.tool_call_id and
                                    call["name"] == tool_name):
                                return msg.content
        return None

    def _run_baseline_seeds(self, spec: ExperimentSpec, slot: GPUSlot, seeds: int):
        """Run baseline seeds sequentially on one slot."""
        for seed in range(1, seeds + 1):
            job_name = f"{spec.name}_s{seed}"
            log_dir = os.path.join(self.config.paths.logs, job_name)
            os.makedirs(log_dir, exist_ok=True)
            command = self._build_train_command(spec, seed=seed)
            scripts_dir = os.path.join(self.config.paths.workspace, "scripts")
            job_id = self.runner.submit(command, job_name, scripts_dir=scripts_dir)
            self.pool.assign(slot.slot_id, job_id, job_name, log_dir)
            self._wait_for_slot_completion(slot)
            status = self.runner.status(slot.job_id)
            if status.state == "completed":
                self._handle_completion(slot)
            else:
                self._handle_failure(slot)

    def _handle_completion(self, slot: GPUSlot):
        """Process a completed experiment."""
        name = slot.experiment_name
        job_id = slot.job_id
        self.pool.release(slot.slot_id)
        log = self.runner.get_log(job_id)
        result = self._parse_results(name, log)
        self.results.add(result)

        self.system_logger.log(event="experiment_completed", spec=name,
                               metric=result.metric_value, status=result.status)

        self._consecutive_failures = 0
        self._write_and_submit_insight(name, result)
        self._write_and_submit_suggestion(name, result)
        self._check_paired_completion(name)

        self.bus.post(AgentMessage(
            sender="builder", recipient="critic",
            type="rerank_request", payload={},
            priority=PRIORITY_RERANK,
        ))
        self.reset_conversation()

    def _write_and_submit_insight(self, name: str, result):
        """Non-blocking: write insight, submit to Critic, don't wait."""
        context = self.results.get_summary()
        insight = self.chat(
            f"## Write insight for experiment: {name}\n\n"
            f"Result: metric={result.metric_value}, status={result.status}\n\n"
            f"## Previous results\n{context}\n\n"
            f"Write observations (factual), comparisons to baselines, "
            f"and hypotheses (labeled as such)."
        )
        self.bus.post(AgentMessage(
            sender="builder", recipient="critic",
            type="insight_review_request",
            payload={"insight": insight, "experiment": name},
            priority=PRIORITY_INSIGHT_REVIEW,
        ))

    def _write_and_submit_suggestion(self, name: str, result: ExperimentResult):
        """Write improvement suggestion based on results. Non-blocking."""
        if result.status in ("crash", "validation_fail", "timeout"):
            return
        context = self.results.get_summary()
        suggestion = self.chat(
            f"## Suggest improvements for: {name}\n\n"
            f"Result: metric={result.metric_value}, status={result.status}\n\n"
            f"## All results\n{context}\n\n"
            f"Based on the training behavior and results, suggest ONE specific improvement. "
            f"Be concrete: name the hyperparameter, architectural change, or training strategy."
        )
        if suggestion and len(suggestion.strip()) > 20:
            self.bus.post(AgentMessage(
                sender="builder", recipient="critic",
                type="suggestion_triage_request",
                payload={"suggestion": suggestion, "experiment": name},
                priority=PRIORITY_SUGGESTION,
            ))
        self.reset_conversation()

    def _track_pair(self, spec: ExperimentSpec):
        if spec.control_variant_of:
            self.pending_pairs.setdefault(spec.control_variant_of, []).append(spec.name)

    def _check_paired_completion(self, name: str):
        for parent, variants in list(self.pending_pairs.items()):
            all_names = [parent] + variants
            if name in all_names and all(self.results.is_completed(n) for n in all_names):
                results_map = {n: self.results.get_by_name(n) for n in all_names}
                insight = self.chat(
                    f"## Write comparative insight for paired experiments\n\n"
                    f"Results: {results_map}\n\n"
                    f"Compare these paired experiments. "
                    f"What does the control variant tell us about the key component?"
                )
                self.bus.post(AgentMessage(
                    sender="builder", recipient="critic",
                    type="insight_review_request",
                    payload={"insight": insight, "experiment": f"comparison_{'+'.join(all_names)}"},
                    priority=PRIORITY_INSIGHT_REVIEW,
                ))
                del self.pending_pairs[parent]

    def _record_validation_fail(self, spec: ExperimentSpec, slot: GPUSlot):
        """Record validation failure with 3-tier error handling.

        Saves error artifacts to experiments/<name>/errors/ for debugging.
        """
        error = self._last_validation_error

        # Save error to experiment's errors/ directory
        experiment_dir = os.path.join(
            self.config.paths.workspace, "experiments", spec.name)
        errors_dir = os.path.join(experiment_dir, "errors")
        os.makedirs(errors_dir, exist_ok=True)
        try:
            atomic_write(
                os.path.join(errors_dir, "validation_error.txt"),
                f"Validation failed for {spec.name}\n\n{error}"
            )
        except Exception:
            pass

        builder_fault = any(f in error for f in self._edited_files)

        if not builder_fault:
            self._alert("task_error",
                f"Task folder error during {spec.name}: {error[:300]}\n"
                f"Builder cannot fix this. Please check your task setup.")
            self.paused = True
            self.system_logger.log(event="builder_paused", reason="task_folder_error",
                                   spec=spec.name, error=error[:500])
            self.pool.release(slot.slot_id)
            return

        if error[:200] == self._last_failure_error[:200]:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 1
            self._last_failure_error = error[:200]

        if self._consecutive_failures >= 3:
            self._alert("systemic_error",
                f"3 consecutive failures with same error. Pausing.\n{error[:300]}")
            self.paused = True
            self.system_logger.log(event="builder_paused", reason="repeated_failures",
                                   error=error[:500])

        self.results.add(ExperimentResult(
            name=spec.name, commit="", algorithm=spec.name,
            model="", metric_value=None, metric_std=None, seeds=0,
            training_seconds=0, peak_memory_mb=0,
            status="validation_fail",
            description=f"Pre-flight validation failed: {error[:200]}",
        ))
        self.pool.release(slot.slot_id)
        self._edited_files.clear()

    def _handle_code_rejection(self, msg):
        spec_name = msg.payload["spec_name"]
        feedback = msg.payload["feedback"]
        for slot in self.pool.get_running():
            if slot.experiment_name == spec_name:
                self.runner.cancel(slot.job_id)
                self.pool.release(slot.slot_id)
                self.chat(f"Fix this code based on review:\n{feedback}")
                self.chat("Commit the fix with git_commit tool.")
                experiment_dir = os.path.join(
                    self.config.paths.workspace, "experiments", spec_name)
                spec_path = os.path.join(experiment_dir, "spec.yaml")
                if not os.path.exists(spec_path):
                    spec_path = os.path.join(
                        self.config.paths.configs, "experiments", f"{spec_name}.yaml")
                spec = ExperimentSpec.from_yaml(spec_path)
                new_slot = self.pool.get_free_slot()
                if new_slot:
                    if os.path.isdir(experiment_dir):
                        if self._validate_experiment_dir(spec, experiment_dir):
                            command = self._build_train_command_experiment_dir(
                                spec, experiment_dir, seed=1)
                            log_dir = os.path.join(experiment_dir, "logs")
                            job_id = self.runner.submit(
                                command, spec_name, scripts_dir=experiment_dir)
                            self.pool.assign(new_slot.slot_id, job_id, spec_name, log_dir)
                    else:
                        if self._validate_implementation(spec):
                            command = self._build_train_command(spec, seed=1)
                            log_dir = os.path.join(self.config.paths.logs, spec_name)
                            scripts_dir = os.path.join(self.config.paths.workspace, "scripts")
                            job_id = self.runner.submit(
                                command, spec_name, scripts_dir=scripts_dir)
                            self.pool.assign(new_slot.slot_id, job_id, spec_name, log_dir)
                break
        self.reset_conversation()

    def _handle_triaged_suggestion(self, msg):
        level = msg.payload["level"]
        suggestion = msg.payload["suggestion"]
        if level == "minor":
            pass  # handled in next fill cycle
        elif level == "moderate":
            self.bus.post(AgentMessage(
                sender="builder", recipient="explorer",
                type="suggestion_for_spec",
                payload={"suggestion": suggestion}
            ))
        elif level == "major":
            pass  # already saved to ideas_for_system2.md by Critic

    def _build_train_command(self, spec: ExperimentSpec, seed: int,
                             time_budget: int | None = None) -> str:
        tb = time_budget or self.config.system.time_budget
        spec_path = os.path.join(self.config.paths.configs, "experiments", f"{spec.name}.yaml")
        os.makedirs(os.path.dirname(spec_path), exist_ok=True)
        spec.to_yaml(spec_path)
        log_dir = os.path.join(
            self.config.paths.logs,
            spec.name if seed == 0 else f"{spec.name}_s{seed}",
        )
        ckpt_dir = os.path.join(
            self.config.paths.checkpoints,
            spec.name if seed == 0 else f"{spec.name}_s{seed}",
        )
        task_module = self.config.task.get("module", "")
        return (
            f"python -m applied_scientist.run_experiment "
            f"--spec {spec_path} "
            f"--task-module {task_module} "
            f"--seed {seed} "
            f"--time-budget {tb} "
            f"--log-path {log_dir} "
            f"--checkpoint-dir {ckpt_dir}"
        )

    def _parse_results(self, name: str, log: str) -> ExperimentResult:
        commit = ""
        try:
            for msg in self.conversation:
                if hasattr(msg, "content") and msg.content and "commit" in msg.content.lower():
                    match = re.search(r'[a-f0-9]{7,40}', msg.content)
                    if match:
                        commit = match.group()
                        break
        except Exception:
            pass

        # Try reading persisted results.json first (survives builder restarts)
        metrics = {}
        results_file = os.path.join(self.config.paths.logs, name, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                metrics = {}

        # Fall back to parsing last JSON line from log
        if not metrics:
            try:
                lines = log.strip().split("\n")
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith("{"):
                        metrics = json.loads(line)
                        break
            except (json.JSONDecodeError, IndexError):
                metrics = {}

        metric_name = self.task.metric[0]
        return ExperimentResult(
            name=name, commit=commit,
            algorithm=name.split("_")[0],
            model=self.llm.get_model_id(),
            metric_value=metrics.get(metric_name),
            metric_std=None, seeds=1,
            training_seconds=metrics.get("training_seconds", metrics.get("elapsed_seconds", 0)),
            peak_memory_mb=metrics.get("peak_memory_mb", 0),
            status="keep" if metrics.get(metric_name) is not None else "crash",
            description=metrics.get("description", ""),
            extra_metrics={k: v for k, v in metrics.items()
                          if k not in {metric_name, "training_seconds",
                                       "peak_memory_mb", "description", "status"}},
        )

    def _get_git_diff(self) -> str:
        result = self.chat("Show the current changes using git_diff tool.")
        return result

    def _is_structural_change(self, diff: str) -> bool:
        structural_patterns = [
            "class ", "def ", "import ", "from ",
            "+class ", "+def ", "+import ", "+from ",
        ]
        lines = diff.split("\n")
        added_lines = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
        for line in added_lines:
            for pattern in structural_patterns:
                if pattern in line:
                    return True
        return False

    def _wait_for_slot_completion(self, slot: GPUSlot):
        while not self._stopped:
            status = self.runner.status(slot.job_id)
            if status.state not in ("pending", "running"):
                return
            time.sleep(10)

    def _handle_early_stop(self, slot: GPUSlot):
        name = slot.experiment_name
        progress_path = os.path.join(slot.log_dir, self.task.progress_file)
        partial_metric = None
        try:
            with open(progress_path) as f:
                progress = json.load(f)
                partial_metric = progress.get(self.task.metric[0])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        result = ExperimentResult(
            name=name, commit="", algorithm=name,
            model="", metric_value=partial_metric, metric_std=None, seeds=1,
            training_seconds=time.time() - slot.submitted_at,
            peak_memory_mb=0, status="early_stop",
            description=(
                f"Early stopped: metric={partial_metric} below threshold "
                f"at {self.config.system.early_stop_fraction * 100:.0f}% of budget"
            ),
        )
        self.results.add(result)
        self.pool.release(slot.slot_id)
        self.system_logger.log(
            event="experiment_early_stopped", spec=name, metric=partial_metric,
            reason=f"Below threshold at {self.config.system.early_stop_fraction * 100:.0f}% budget",
        )
        self._write_and_submit_insight(name, result)
        self._consecutive_failures = 0

    def _handle_timeout(self, slot: GPUSlot):
        name = slot.experiment_name
        self.runner.cancel(slot.job_id)
        result = ExperimentResult(
            name=name, commit="", algorithm=name,
            model="", metric_value=None, metric_std=None, seeds=1,
            training_seconds=time.time() - slot.submitted_at,
            peak_memory_mb=0, status="timeout",
            description=f"Exceeded {self.config.system.timeout_multiplier}x time budget",
        )
        self.results.add(result)
        self.pool.release(slot.slot_id)
        self.system_logger.log(event="experiment_timeout", spec=name)

    def _handle_failure(self, slot: GPUSlot):
        name = slot.experiment_name
        log = self.runner.get_log(slot.job_id)
        self.pool.release(slot.slot_id)

        # Save crash info to experiment's errors/ directory
        experiment_dir = os.path.join(
            self.config.paths.workspace, "experiments", name)
        errors_dir = os.path.join(experiment_dir, "errors")
        os.makedirs(errors_dir, exist_ok=True)
        try:
            atomic_write(
                os.path.join(errors_dir, "runtime_error.txt"),
                f"Training crashed for {name}\nJob ID: {slot.job_id}\n\n{log}"
            )
            # Also save just the traceback if we can extract it
            if "Traceback" in log:
                tb_start = log.rfind("Traceback")
                atomic_write(
                    os.path.join(errors_dir, "traceback.txt"),
                    log[tb_start:]
                )
        except Exception:
            pass

        fixable_patterns = [
            "OutOfMemoryError", "CUDA out of memory",
            "RuntimeError", "TypeError", "ValueError",
        ]
        is_fixable = any(p in log for p in fixable_patterns)

        if is_fixable:
            try:
                self.chat(
                    f"Training crashed with this error:\n{log[-2000:]}\n\n"
                    f"Attempt to fix the issue."
                )
                experiment_dir = os.path.join(
                    self.config.paths.workspace, "experiments", name)
                spec_path = os.path.join(experiment_dir, "spec.yaml")
                if not os.path.exists(spec_path):
                    spec_path = os.path.join(
                        self.config.paths.configs, "experiments", f"{name}.yaml")
                spec = ExperimentSpec.from_yaml(spec_path)
                if os.path.isdir(experiment_dir):
                    if self._validate_experiment_dir(spec, experiment_dir):
                        self.chat("Commit the fix with git_commit tool.")
                        new_slot = self.pool.get_free_slot()
                        if new_slot:
                            command = self._build_train_command_experiment_dir(
                                spec, experiment_dir, seed=1)
                            log_dir = os.path.join(experiment_dir, "logs")
                            job_id = self.runner.submit(
                                command, f"{name}_fix", scripts_dir=experiment_dir)
                            self.pool.assign(new_slot.slot_id, job_id, name, log_dir)
                            self.system_logger.log(event="experiment_retry", spec=name)
                            self.reset_conversation()
                            return
                else:
                    if self._validate_implementation(spec):
                        self.chat("Commit the fix with git_commit tool.")
                        new_slot = self.pool.get_free_slot()
                        if new_slot:
                            command = self._build_train_command(spec, seed=1)
                            log_dir = os.path.join(self.config.paths.logs, name)
                            scripts_dir = os.path.join(self.config.paths.workspace, "scripts")
                            job_id = self.runner.submit(
                                command, f"{name}_fix", scripts_dir=scripts_dir)
                            self.pool.assign(new_slot.slot_id, job_id, name, log_dir)
                            self.system_logger.log(event="experiment_retry", spec=name)
                            self.reset_conversation()
                            return
            except Exception:
                pass

        result = ExperimentResult(
            name=name, commit="", algorithm=name,
            model="", metric_value=None, metric_std=None, seeds=1,
            training_seconds=time.time() - (slot.submitted_at or time.time()),
            peak_memory_mb=0, status="crash",
            description=f"Training crashed: {log[-200:]}",
        )
        self.results.add(result)
        self.system_logger.log(event="experiment_crashed", spec=name, error=log[-500:])
        self._write_and_submit_insight(name, result)
        self.reset_conversation()

    def _persist_pairs(self):
        state_path = os.path.join(self.config.paths.workspace, ".state", "pending_pairs.json")
        atomic_write(state_path, json.dumps(self.pending_pairs))

    @classmethod
    def load_pairs(cls, state_path: str) -> dict:
        if os.path.exists(state_path):
            with open(state_path) as f:
                return json.loads(f.read())
        return {}
