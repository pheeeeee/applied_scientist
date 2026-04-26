from __future__ import annotations

import importlib.util
import os
import threading
import time

from applied_scientist.config import Config
from applied_scientist.core.spec import ExperimentSpec
from applied_scientist.core.results_tracker import ExperimentResult, ResultsTracker
from applied_scientist.core.priority_queue import PriorityQueue
from applied_scientist.core.knowledge_base import KnowledgeBase
from applied_scientist.core.gpu_pool import GPUPool
from applied_scientist.core.message_bus import MessageBus, AgentMessage
from applied_scientist.core.cost_tracker import CostTracker
from applied_scientist.core.event_logger import EventLogger
from applied_scientist.agents.explorer import ExplorerAgent
from applied_scientist.agents.critic import CriticAgent
from applied_scientist.agents.builder import BuilderAgent
from applied_scientist.agents.orchestrator import OrchestratorAgent
from applied_scientist.agents.debugger import DebuggerAgent
from applied_scientist.llm import get_backend
from applied_scientist.llm.base import LLMMessage
from applied_scientist.tools import get_tools
from applied_scientist.compute import get_runner
from applied_scientist.tuning import HyperparameterTuner


class AppliedScientistSystem:
    def __init__(self, config_path: str, task_module: str,
                 n_gpus: int = None, job_runner_override: str = None,
                 test_run: bool = False):
        self.test_run = test_run

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

        # Initialize LLM backends
        self.llms = {}
        for role in ["explorer", "critic", "builder", "orchestrator", "debugger"]:
            # Debugger uses same config as builder if not specified
            if role == "debugger" and role not in self.config.llm:
                llm_config = self.config.llm.get("builder") or self.config.llm["critic"]
            else:
                llm_config = self.config.llm.get(role)
                if llm_config is None:
                    continue
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
        self.runner = get_runner(self.config.compute,
                                 workspace=self.config.paths.workspace)

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
                self.system_logger, self.config),
            "debugger": DebuggerAgent(
                self.llms["debugger"],
                get_tools("debugger", self.config.paths.workspace),
                self.prompts["debugger"], self.bus, self.cost_tracker,
                self.results, self.runner, self.system_logger,
                self.config, self.config.paths.workspace),
            "orchestrator": OrchestratorAgent(
                self.llms["orchestrator"],
                get_tools("orchestrator", self.config.paths.workspace),
                self.prompts["orchestrator"], self.bus, self.cost_tracker,
                self.queue, self.kb, self.results, self.pool,
                self.system_logger, {}),
        }
        self.agents["orchestrator"].agents = self.agents

    def start(self):
        if self.test_run:
            self._run_test()
            return

        self._init_git_repo()
        self._inject_random_baseline()
        self._inject_baseline()
        self._recover_state()

        self.threads = {}
        for name in ["explorer", "critic", "builder", "debugger"]:
            t = threading.Thread(target=self.agents[name].run, name=name, daemon=True)
            t.start()
            self.threads[name] = t

        self._watchdog_thread = threading.Thread(
            target=self._watchdog, name="watchdog", daemon=True)
        self._watchdog_thread.start()

        print(f"Applied Scientist started.")
        print(f"  Task: {self.task.name}")
        print(f"  Metric: {self.task.metric[0]} ({self.task.metric[1]})")
        print(f"  GPUs: {self.config.system.n_gpus}")
        print(f"  Tuning: {'enabled' if self.config.tuning.enabled else 'disabled'}")
        print(f"  LLMs: Explorer={self.llms['explorer'].get_model_id()}, "
              f"Critic={self.llms['critic'].get_model_id()}, "
              f"Builder={self.llms['builder'].get_model_id()}")
        print()

        try:
            self.agents["orchestrator"].run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        for agent in self.agents.values():
            agent._stopped = True
        self.queue.save()
        self.pool._persist()
        print(f"\nApplied Scientist stopped.")
        print(f"  Total cost: ${self.cost_tracker.get_total_cost():.2f}")
        print(f"  Results saved to {self.config.paths.results}")

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
        # claudecode backend uses OAuth subscription auth, not API keys
        if backend == "claudecode":
            return ""
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

    def _watchdog(self):
        restart_counts = {name: 0 for name in ["explorer", "critic", "builder", "debugger"]}
        max_restarts = 3
        while not self.agents["orchestrator"]._stopped:
            time.sleep(30)
            for name in ["explorer", "critic", "builder", "debugger"]:
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
            for role in ["explorer", "critic", "builder", "orchestrator", "debugger"]
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
        print("Running test...")
        steps = [
            ("Config", lambda: True),
            ("Task adapter", self._test_task_adapter),
            ("LLM backends", self._test_llm_connectivity),
            ("Compute", self._test_compute),
            ("Workspace", self._test_workspace),
        ]
        for name, fn in steps:
            try:
                fn()
                print(f"  {name:20s} OK")
            except Exception as e:
                print(f"  {name:20s} FAILED -- {e}")
                print(f"\nTest run failed at: {name}")
                return
        print(f"\nSystem is ready. Run without --test-run to start.")

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
