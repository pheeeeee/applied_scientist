from __future__ import annotations

import json
import os
import re
import time

from applied_scientist.agents.base import BaseAgent
from applied_scientist.core.message_bus import (
    AgentMessage, PRIORITY_CODE_REVIEW, PRIORITY_INSIGHT_REVIEW,
    PRIORITY_SUGGESTION, PRIORITY_RERANK,
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
                 config):
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
        """Implement, validate, commit, submit. Non-blocking code review."""
        self._track_pair(spec)
        self._edited_files.clear()

        is_baseline = spec.category == "baseline"
        seeds = self.config.system.baseline_seeds if is_baseline else 1

        if is_baseline and seeds > 1:
            self._run_baseline_seeds(spec, slot, seeds)
            return

        spec_content = spec.to_yaml()
        code_map = self.task.code_map if self.task.code_map else ""
        self.chat(
            f"## Implement this experiment\n\n"
            f"```yaml\n{spec_content}\n```\n\n"
            f"## Task context\n{self.task.domain_context}\n\n"
            f"## Code map\n{code_map}\n\n"
            f"Read the current code, implement the architecture from this spec. "
            f"Only modify training code — do not change evaluation or data loading. "
            f"Use edit_file and write_file tools."
        )

        for msg in self.conversation:
            if msg.tool_calls:
                for call in msg.tool_calls:
                    if call["name"] in ("edit_file", "write_file"):
                        path = call["arguments"].get("path", call["arguments"].get("file_path", ""))
                        if path:
                            self._edited_files.add(path)

        if not self._validate_implementation(spec):
            error = self._last_validation_error
            self.chat(f"Pre-flight validation failed:\n{error}\nFix the issue.")
            if not self._validate_implementation(spec):
                self._record_validation_fail(spec, slot)
                self.reset_conversation()
                return

        diff = self._get_git_diff()
        if self._is_structural_change(diff):
            self.bus.post(AgentMessage(
                sender="builder", recipient="critic",
                type="code_review_request",
                payload={"spec_name": spec.name, "diff": diff},
                priority=PRIORITY_CODE_REVIEW,
            ))

        self.chat("Commit the changes with git_commit tool.")

        log_dir = os.path.join(self.config.paths.logs, spec.name)
        os.makedirs(log_dir, exist_ok=True)
        command = self._build_train_command(spec, seed=1)
        job_id = self.runner.submit(command, spec.name)
        self.pool.assign(slot.slot_id, job_id, spec.name, log_dir)

        self.system_logger.log(event="experiment_submitted", spec=spec.name, job_id=job_id)
        self.reset_conversation()

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
            job_id = self.runner.submit(command, job_name)
            self.pool.assign(slot.slot_id, job_id, job_name, log_dir)
            self._wait_for_slot_completion(slot)
            status = self.runner.status(slot.job_id)
            if status.state == "completed":
                self._handle_completion(slot)
            else:
                self._handle_failure(slot)

    def _handle_completion(self, slot: GPUSlot):
        """Process a completed experiment."""
        self.pool.release(slot.slot_id)
        name = slot.experiment_name
        log = self.runner.get_log(slot.job_id)
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
        """Record validation failure with 3-tier error handling."""
        error = self._last_validation_error

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
                spec = ExperimentSpec.from_yaml(
                    os.path.join(self.config.paths.configs, "experiments", f"{spec_name}.yaml"))
                new_slot = self.pool.get_free_slot()
                if new_slot:
                    if self._validate_implementation(spec):
                        command = self._build_train_command(spec, seed=1)
                        log_dir = os.path.join(self.config.paths.logs, spec_name)
                        job_id = self.runner.submit(command, spec_name)
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

        metrics = {}
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
            training_seconds=metrics.get("training_seconds", 0),
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
                spec = ExperimentSpec.from_yaml(
                    os.path.join(self.config.paths.configs, "experiments", f"{name}.yaml"))
                if self._validate_implementation(spec):
                    self.chat("Commit the fix with git_commit tool.")
                    new_slot = self.pool.get_free_slot()
                    if new_slot:
                        command = self._build_train_command(spec, seed=1)
                        log_dir = os.path.join(self.config.paths.logs, name)
                        job_id = self.runner.submit(command, f"{name}_fix")
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
