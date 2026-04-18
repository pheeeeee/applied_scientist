from __future__ import annotations

import importlib.util
import json
import os
import signal
import threading
import time
from datetime import datetime, timezone

import yaml

from applied_scientist.config import Config
from applied_scientist.core.spec import ExperimentSpec
from applied_scientist.core.results_tracker import ExperimentResult, ResultsTracker
from applied_scientist.core.priority_queue import PriorityQueue
from applied_scientist.core.knowledge_base import KnowledgeBase
from applied_scientist.core.gpu_pool import GPUPool
from applied_scientist.core.message_bus import MessageBus, AgentMessage, PRIORITY_SPEC_REVIEW
from applied_scientist.core.cost_tracker import CostTracker
from applied_scientist.core.event_logger import EventLogger
from applied_scientist.agents.explorer import ExplorerAgent
from applied_scientist.agents.critic import CriticAgent
from applied_scientist.agents.builder import BuilderAgent
from applied_scientist.agents.orchestrator import OrchestratorAgent
from applied_scientist.llm import get_backend
from applied_scientist.llm.base import LLMMessage
from applied_scientist.tools import get_tools
from applied_scientist.compute import get_runner
from applied_scientist.tuning import HyperparameterTuner


class AppliedScientistSystem:
    def __init__(self, config_path: str, task_module: str,
                 n_gpus: int = None, job_runner_override: str = None,
                 test_run: bool = False, slack: bool = False,
                 approve_plans: bool = False, approve_submit: bool = False):
        self.test_run = test_run
        self.slack = slack
        self.approve_plans = approve_plans
        self.approve_submit = approve_submit

        # Load config
        self.config = Config.from_yaml(config_path)
        if n_gpus is not None:
            self.config.system.n_gpus = n_gpus
        if job_runner_override:
            self.config.compute.backend = job_runner_override

        # Store task module path in config so Builder can reference it
        self.config.task['module'] = task_module

        # Load task adapter
        self.task = self._load_task_adapter(task_module)

        # Initialize paths
        self._init_workspace()

        # Initialize event loggers
        self.review_logger = EventLogger(
            os.path.join(self.config.paths.results, "review_log.jsonl"))
        self.explorer_logger = EventLogger(
            os.path.join(self.config.paths.results, "explorer_journal.jsonl"))
        self.system_logger = EventLogger(
            os.path.join(self.config.paths.results, "system_events.jsonl"))

        # Initialize core data structures
        state_dir = os.path.join(self.config.paths.workspace, ".state")
        os.makedirs(state_dir, exist_ok=True)
        self.queue = PriorityQueue(
            os.path.join(self.config.paths.configs, "priority_queue.yaml"))
        self.kb = KnowledgeBase(
            os.path.join(self.config.paths.results, "knowledge.md"))
        self.results = ResultsTracker(
            os.path.join(self.config.paths.results, "results.tsv"),
            self.task.metric[0], self.task.metric[1])
        self.pool = GPUPool(self.config.system.n_gpus,
                            os.path.join(state_dir, "gpu_slots.yaml"))
        self.bus = MessageBus(os.path.join(state_dir, "pending_messages.json"))
        self.cost_tracker = CostTracker()

        # CLI flag overrides config
        if not self.slack and self.config.interface.backend == "slack":
            self.slack = True

        # Initialize Slack bridge (if enabled)
        self.slack_io = None
        self.bus_relay = None
        if self.slack:
            from slack_bridge import SlackIO, BusRelay
            self.slack_io = SlackIO.from_env()
            self.slack_io.start()
            if os.environ.get("SLACK_FEED_CHANNEL"):
                self.bus_relay = BusRelay(
                    self.bus,
                    self.slack_io.bridge,
                    feed_channel=os.environ["SLACK_FEED_CHANNEL"],
                )
                self.bus_relay.start()

        # Initialize LLM backends
        self.llms = {}
        for role in ["explorer", "critic", "builder", "orchestrator"]:
            llm_config = self.config.llm[role]
            backend_kwargs = {
                "backend": llm_config.backend,
                "model": llm_config.model,
                "api_key": self._get_api_key(llm_config.backend),
            }
            if llm_config.base_url:
                backend_kwargs["base_url"] = llm_config.base_url
            self.llms[role] = get_backend(backend_kwargs)
            self.cost_tracker.register(role, self.llms[role].get_model_id())

        # Initialize compute backend
        self.runner = get_runner(self.config.compute)

        # Initialize tuner
        self.tuner = HyperparameterTuner(self.task, self.runner, self.config) \
                     if self.config.tuning.enabled else None

        # Build system prompts
        self.prompts = self._render_prompts()

        # Initialize agents
        self.agents = {
            "explorer": ExplorerAgent(
                self.llms["explorer"],
                get_tools("explorer", self.config.paths.workspace),
                self.prompts["explorer"], self.bus, self.cost_tracker,
                self.kb, self.results, self.queue, self.explorer_logger, self.config),
            "critic": CriticAgent(
                self.llms["critic"],
                get_tools("critic", self.config.paths.workspace),
                self.prompts["critic"], self.bus, self.cost_tracker,
                self.queue, self.kb, self.results,
                self.review_logger, self.system_logger, self.config),
            "builder": BuilderAgent(
                self.llms["builder"],
                get_tools("builder", self.config.paths.workspace),
                self.prompts["builder"], self.bus, self.cost_tracker,
                self.queue, self.kb, self.results,
                self.pool, self.runner, self.task,
                self.system_logger, self.config,
                approve_plans=self.approve_plans,
                approve_submit=self.approve_submit),
            "orchestrator": OrchestratorAgent(
                self.llms["orchestrator"],
                get_tools("orchestrator", self.config.paths.workspace),
                self.prompts["orchestrator"], self.bus, self.cost_tracker,
                self.queue, self.kb, self.results, self.pool,
                self.system_logger, {},
                report_fn=self.generate_report,
                io=self.slack_io),
        }
        self.agents["orchestrator"].agents = self.agents

    def _system_output(self, text: str):
        """Output system message. Routes through Slack when available, else print()."""
        if self.slack_io:
            self.slack_io.send(text)
        else:
            print(text)

    def start(self):
        if self.test_run:
            self._run_test()
            return

        self._init_git_repo()
        self._inject_random_baseline()
        self._inject_baseline()
        self._recover_state()
        self._recover_drafts()

        self.threads = {}
        for name in ["explorer", "critic", "builder"]:
            t = threading.Thread(target=self.agents[name].run, name=name, daemon=True)
            t.start()
            self.threads[name] = t

        self._watchdog_thread = threading.Thread(
            target=self._watchdog, name="watchdog", daemon=True)
        self._watchdog_thread.start()

        banner = (
            f"Applied Scientist started.\n"
            f"  Task: {self.task.name}\n"
            f"  Metric: {self.task.metric[0]} ({self.task.metric[1]})\n"
            f"  GPUs: {self.config.system.n_gpus}\n"
            f"  Tuning: {'enabled' if self.config.tuning.enabled else 'disabled'}\n"
            f"  LLMs: Explorer={self.llms['explorer'].get_model_id()}, "
            f"Critic={self.llms['critic'].get_model_id()}, "
            f"Builder={self.llms['builder'].get_model_id()}"
        )
        self._system_output(banner)

        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        try:
            self.agents["orchestrator"].run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        for agent in self.agents.values():
            agent._stopped = True
        self.queue.save()
        self.pool._persist()
        report_path = self.generate_report()

        shutdown_msg = (
            f"Applied Scientist stopped.\n"
            f"  Total cost: ${self.cost_tracker.get_total_cost():.2f}\n"
            f"  Results saved to {self.config.paths.results}"
        )
        if report_path:
            shutdown_msg += f"\n  Report saved to {report_path}.md / .json"

        # Send shutdown message BEFORE stopping the bridge
        self._system_output(shutdown_msg)

        # Stop Slack bridge and relay
        if self.bus_relay:
            self.bus_relay.stop()
        if self.slack_io:
            self.slack_io.stop()

    def generate_report(self) -> str | None:
        """Generate timestamped report combining results with full experiment specs.
        Returns the base path (without extension) or None on failure."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            base_path = os.path.join(self.config.paths.results, f"report_{timestamp}")
            experiments_dir = os.path.join(self.config.paths.configs, "experiments")

            # Build combined data: results + specs
            all_results = self.results.get_all()
            report_entries = []
            for r in all_results:
                entry = {
                    "name": r.name,
                    "algorithm": r.algorithm,
                    "model": r.model,
                    "metric_value": r.metric_value,
                    "metric_std": r.metric_std,
                    "seeds": r.seeds,
                    "training_seconds": r.training_seconds,
                    "peak_memory_mb": r.peak_memory_mb,
                    "status": r.status,
                    "description": r.description,
                    "extra_metrics": r.extra_metrics,
                    "commit": r.commit,
                }
                # Load corresponding spec if it exists
                spec_path = os.path.join(experiments_dir, f"{r.name}.yaml")
                if os.path.exists(spec_path):
                    spec = ExperimentSpec.from_yaml(spec_path)
                    entry["spec"] = {
                        "source_paper": spec.source_paper,
                        "architecture": spec.architecture,
                        "training_config": spec.training_config,
                        "task_config": spec.task_config,
                        "why_it_might_work": spec.why_it_might_work,
                        "resource_estimate": spec.resource_estimate,
                        "category": spec.category,
                        "tags": spec.tags,
                        "tunable_hyperparameters": spec.tunable_hyperparameters,
                    }
                report_entries.append(entry)

            # Sort by metric (best first)
            direction = self.task.metric[1]
            scored = [e for e in report_entries if e["metric_value"] is not None]
            unscored = [e for e in report_entries if e["metric_value"] is None]
            scored.sort(key=lambda e: e["metric_value"],
                        reverse=(direction == "higher"))
            sorted_entries = scored + unscored

            report_data = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "task": self.task.name,
                "metric": {"name": self.task.metric[0], "direction": direction},
                "total_experiments": len(all_results),
                "total_cost_usd": self.cost_tracker.get_total_cost(),
                "best": scored[0] if scored else None,
                "experiments": sorted_entries,
            }

            # Write JSON
            with open(f"{base_path}.json", "w") as f:
                json.dump(report_data, f, indent=2, default=str)

            # Write Markdown
            md_lines = [
                f"# Applied Scientist Report",
                f"",
                f"**Generated:** {report_data['generated_at']}  ",
                f"**Task:** {self.task.name}  ",
                f"**Metric:** {self.task.metric[0]} ({direction})  ",
                f"**Experiments:** {len(all_results)}  ",
                f"**Total LLM cost:** ${report_data['total_cost_usd']:.2f}",
                f"",
            ]
            if scored:
                best = scored[0]
                md_lines += [
                    f"## Best Result",
                    f"",
                    f"**{best['name']}** — {best['metric_value']:.4f}",
                    f"",
                ]

            md_lines += [f"## All Experiments", f""]
            for e in sorted_entries:
                val = f"{e['metric_value']:.4f}" if e['metric_value'] is not None else "N/A"
                md_lines += [
                    f"### {e['name']}",
                    f"",
                    f"- **Metric:** {val} (std: {e['metric_std']})",
                    f"- **Status:** {e['status']}",
                    f"- **Algorithm:** {e['algorithm']}",
                    f"- **Model:** {e['model']}",
                    f"- **Training time:** {e['training_seconds']:.0f}s",
                    f"- **Peak memory:** {e['peak_memory_mb']:.0f} MB",
                    f"- **Description:** {e['description']}",
                ]
                if "spec" in e:
                    s = e["spec"]
                    md_lines += [
                        f"- **Source paper:** {s['source_paper']}",
                        f"- **Why it might work:** {s['why_it_might_work']}",
                        f"- **Category:** {s['category']}",
                        f"- **Architecture:**",
                        f"  ```yaml",
                    ]
                    for line in yaml.dump(s["architecture"], default_flow_style=False).strip().split("\n"):
                        md_lines.append(f"  {line}")
                    md_lines += [
                        f"  ```",
                        f"- **Training config:**",
                        f"  ```yaml",
                    ]
                    for line in yaml.dump(s["training_config"], default_flow_style=False).strip().split("\n"):
                        md_lines.append(f"  {line}")
                    md_lines += [f"  ```"]
                    if s["tunable_hyperparameters"]:
                        md_lines += [
                            f"- **Tunable hyperparameters:**",
                            f"  ```yaml",
                        ]
                        for line in yaml.dump(s["tunable_hyperparameters"], default_flow_style=False).strip().split("\n"):
                            md_lines.append(f"  {line}")
                        md_lines += [f"  ```"]
                md_lines.append("")

            with open(f"{base_path}.md", "w") as f:
                f.write("\n".join(md_lines))

            return base_path
        except Exception as e:
            self.system_logger.log(event="report_generation_failed", error=str(e))
            return None

    def _init_workspace(self):
        dirs = [
            self.config.paths.workspace,
            self.config.paths.results,
            self.config.paths.configs,
            self.config.paths.checkpoints,
            self.config.paths.logs,
            os.path.join(self.config.paths.configs, "drafts"),
            os.path.join(self.config.paths.configs, "experiments"),
            os.path.join(self.config.paths.workspace, ".state"),
            os.path.join(self.config.paths.workspace, "experiments"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _init_git_repo(self):
        git_dir = os.path.join(self.config.paths.workspace, ".git")
        if not os.path.exists(git_dir):
            import subprocess
            subprocess.run(["git", "init"],
                          cwd=self.config.paths.workspace, capture_output=True)
            subprocess.run(["git", "add", "-A"],
                          cwd=self.config.paths.workspace, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial workspace setup"],
                          cwd=self.config.paths.workspace, capture_output=True)

    def _get_api_key(self, backend: str) -> str:
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
        }
        env_var = key_map.get(backend, f"{backend.upper()}_API_KEY")
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(f"Missing API key: set {env_var} environment variable")
        return key

    def _inject_random_baseline(self):
        if self.results.is_completed("random_baseline"):
            return
        metrics = self.task.get_random_baseline()
        if metrics is None:
            return
        metric_name = self.task.metric[0]
        self.results.add(ExperimentResult(
            name="random_baseline", commit="-", algorithm="random",
            model="none", metric_value=metrics.get(metric_name),
            metric_std=None, seeds=0, training_seconds=0, peak_memory_mb=0,
            status="keep", description="Random/untrained agent baseline",
            extra_metrics=metrics,
        ))
        self.system_logger.log(event="random_baseline_recorded", metrics=metrics)

    def _inject_baseline(self):
        if self.results.is_completed("ppo_baseline"):
            return
        if self.queue.contains("ppo_baseline"):
            return
        custom = self.task.get_baseline_spec()
        if custom:
            spec = ExperimentSpec(**custom)
        else:
            spec = ExperimentSpec(
                name="ppo_baseline",
                description="Default PPO with standard hyperparameters",
                source_paper="Schulman et al. 2017 (PPO)",
                architecture={"type": "independent_ppo",
                              "policy": {"structure": "FC, 2 layers, 256 units, ReLU"}},
                why_it_might_work="Standard baseline for comparison",
                task_config={}, training_config={},
                resource_estimate={"memory": "~4GB", "training_time": "Standard"},
                category="baseline",
            )
        spec_path = os.path.join(
            self.config.paths.configs, "experiments", f"{spec.name}.yaml")
        if not os.path.exists(spec_path):
            spec.to_yaml(spec_path)
        self.queue.insert(spec, score=100.0)

    def _recover_state(self):
        """Reconcile persisted state with actual compute backend on restart."""
        state_path = os.path.join(self.config.paths.workspace,
                                  ".state", "gpu_slots.yaml")
        if os.path.exists(state_path):
            saved_pool = GPUPool.from_state(state_path, self.config.system.n_gpus)
            for slot in saved_pool.get_running():
                if slot.job_id:
                    actual = self.runner.status(slot.job_id)
                    if actual.state == "running":
                        self.pool.assign(slot.slot_id, slot.job_id,
                                        slot.experiment_name, slot.log_dir)
                    else:
                        # Job finished/failed while system was down — release the slot
                        self.pool.release(slot.slot_id)

        pairs_path = os.path.join(self.config.paths.workspace,
                                  ".state", "pending_pairs.json")
        if os.path.exists(pairs_path):
            self.agents["builder"].pending_pairs = BuilderAgent.load_pairs(pairs_path)

    def _recover_drafts(self):
        """Re-inject unreviewed drafts from configs/drafts/ into the Critic's inbox.

        On restart, message bus messages from previous runs are lost. This scans
        for draft specs that haven't been approved (not in configs/experiments/)
        and posts spec_review_request for each.
        """
        import glob as glob_mod
        drafts_dir = os.path.join(self.config.paths.configs, "drafts")
        experiments_dir = os.path.join(self.config.paths.configs, "experiments")
        completed_names = set()
        if os.path.isdir(experiments_dir):
            for f in os.listdir(experiments_dir):
                if f.endswith(".yaml"):
                    completed_names.add(f)

        count = 0
        for draft_path in sorted(glob_mod.glob(os.path.join(drafts_dir, "*.yaml"))):
            fname = os.path.basename(draft_path)
            if fname in completed_names:
                continue  # already approved
            self.bus.post(AgentMessage(
                sender="system", recipient="critic",
                type="spec_review_request",
                payload={"draft_path": draft_path, "round": 1},
                priority=PRIORITY_SPEC_REVIEW,
            ))
            count += 1
        if count > 0:
            self.system_logger.log(
                event="drafts_recovered", count=count,
                message=f"Re-injected {count} unreviewed drafts for Critic review")

    def _watchdog(self):
        restart_counts = {name: 0 for name in ["explorer", "critic", "builder"]}
        max_restarts = 3
        while not self.agents["orchestrator"]._stopped:
            time.sleep(30)
            for name in ["explorer", "critic", "builder"]:
                thread = self.threads.get(name)
                if thread and not thread.is_alive():
                    if restart_counts[name] >= max_restarts:
                        self.bus.post(AgentMessage(
                            sender="watchdog", recipient="orchestrator",
                            type="alert",
                            payload={"message":
                                     f"{name} died {max_restarts} times. Not restarting."}
                        ))
                        self.system_logger.log(
                            event="alert", source="watchdog",
                            message=f"{name} died {max_restarts} times")
                        continue
                    restart_counts[name] += 1
                    self.agents[name]._stopped = False
                    self.agents[name].reset_conversation()
                    t = threading.Thread(target=self.agents[name].run,
                                        name=name, daemon=True)
                    t.start()
                    self.threads[name] = t
                    msg = (f"{name} crashed. Restarted "
                           f"({restart_counts[name]}/{max_restarts}).")
                    self.bus.post(AgentMessage(
                        sender="watchdog", recipient="orchestrator",
                        type="alert", payload={"message": msg}
                    ))
                    self.system_logger.log(
                        event="alert", source="watchdog", message=msg)

    def _render_prompts(self) -> dict[str, str]:
        from jinja2 import Environment, FileSystemLoader
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        env = Environment(loader=FileSystemLoader(prompts_dir))
        context = {
            "task_name": self.task.name,
            "task_domain_context": self.task.domain_context,
            "task_code_map": self.task.code_map,
            "metric_name": self.task.metric[0],
            "metric_direction": self.task.metric[1],
            "time_budget": self.config.system.time_budget,
            "max_review_rounds": self.config.system.max_spec_review_rounds,
        }
        return {
            role: env.get_template(f"{role}.md.j2").render(**context)
            for role in ["explorer", "critic", "builder", "orchestrator"]
        }

    def _load_task_adapter(self, module_path: str):
        from applied_scientist.task.base import TaskAdapter
        spec = importlib.util.spec_from_file_location(
            "task_adapter", os.path.join(module_path, "adapter.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, TaskAdapter) and obj is not TaskAdapter:
                return obj()
        raise ValueError(f"No TaskAdapter subclass found in {module_path}/adapter.py")

    def _run_test(self):
        self._system_output("Running test...")
        steps = [
            ("Config", lambda: True),
            ("Task adapter", self._test_task_adapter),
            ("LLM backends", self._test_llm_connectivity),
            ("Compute", self._test_compute),
            ("Workspace", self._test_workspace),
        ]
        results = []
        all_ok = True
        for name, fn in steps:
            try:
                fn()
                results.append(f"  {name:20s} OK")
            except Exception as e:
                results.append(f"  {name:20s} FAILED -- {e}")
                results.append(f"\nTest run failed at: {name}")
                all_ok = False
                break
        if all_ok:
            results.append(f"\nSystem is ready. Run without --test-run to start.")
        self._system_output("\n".join(results))

    def _test_task_adapter(self):
        assert self.task.name, "TaskAdapter.name is empty"
        assert len(self.task.metric) == 2, "TaskAdapter.metric must be (name, direction)"
        assert self.task.metric[1] in ("higher", "lower"), \
            "metric direction must be 'higher' or 'lower'"
        assert self.task.domain_context, "TaskAdapter.domain_context is empty"
        assert self.task.code_map, "TaskAdapter.code_map is empty"

    def _test_llm_connectivity(self):
        for role, llm in self.llms.items():
            response = llm.complete([LLMMessage(role="user", content="Say 'ok'")])
            assert response.content, f"LLM {role} returned empty response"

    def _test_compute(self):
        job_id = self.runner.submit("echo 'test'", "test_job")
        for _ in range(30):
            s = self.runner.status(job_id)
            if s.state == "completed":
                return
            time.sleep(1)
        raise TimeoutError("Test job did not complete in 30 seconds")

    def _test_workspace(self):
        self._init_workspace()
        self._init_git_repo()
