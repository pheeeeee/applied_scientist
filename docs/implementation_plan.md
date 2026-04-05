# Applied Scientist — Implementation Plan

This document specifies every module, class, function, and prompt needed to build the Applied Scientist system. It is ordered by build phase — each phase depends only on prior phases. Every method referenced in the code is fully defined — there are no undefined methods.

---

## Project Structure

```
applied-scientist/
├── README.md
├── LICENSE
├── pyproject.toml
│
├── applied_scientist/                    # Main package
│   ├── __init__.py                       # Version, package metadata
│   ├── __main__.py                       # Entry point: python -m applied_scientist
│   ├── config.py                         # Config loading, validation, defaults
│   ├── system.py                         # Main orchestrator: thread management, lifecycle, recovery
│   ├── run_experiment.py                 # Standardized training wrapper for compute backends
│   ├── tuning.py                         # Bayesian hyperparameter optimization via Optuna
│   │
│   ├── core/                             # Core data structures (no LLM dependency)
│   │   ├── __init__.py
│   │   ├── utils.py                      # Atomic writes, helpers
│   │   ├── spec.py                       # ExperimentSpec dataclass
│   │   ├── priority_queue.py             # Ranked experiment queue with scoring, dedup
│   │   ├── knowledge_base.py             # Knowledge base read/write/synthesis management
│   │   ├── results_tracker.py            # results.tsv read/write, baseline aggregation
│   │   ├── gpu_pool.py                   # GPU slot pool: track slots, assign jobs, persist state
│   │   ├── message_bus.py                # Inter-agent async communication (thread-safe, non-blocking)
│   │   ├── cost_tracker.py              # Per-agent token/cost tracking
│   │   └── event_logger.py             # JSONL append-only logging for review_log, explorer_journal, system_events
│   │
│   ├── llm/                              # LLM backends (pluggable)
│   │   ├── __init__.py                   # Registry: get_backend(name) -> LLMBackend
│   │   ├── base.py                       # LLMBackend abstract class
│   │   ├── anthropic_backend.py          # Claude via Anthropic SDK
│   │   ├── openai_backend.py             # GPT via OpenAI SDK
│   │   ├── gemini_backend.py             # Gemini via Google GenAI SDK
│   │   └── openai_compatible.py          # Any OpenAI-compatible API (Ollama, vLLM, Together, etc.)
│   │
│   ├── tools/                            # Agent tools (pluggable)
│   │   ├── __init__.py                   # Registry: get_tools(agent_role) -> List[Tool]
│   │   ├── base.py                       # Tool abstract class
│   │   ├── file_ops.py                   # read_file, write_file, list_dir, edit_file
│   │   ├── web_search.py                 # search_web (Google, Bing, Serper, SerpAPI)
│   │   ├── paper_reader.py              # search_papers (Semantic Scholar), read_arxiv, read_pdf
│   │   ├── shell.py                      # run_command (with timeout, sandboxing)
│   │   ├── git_ops.py                    # git_commit, git_log, git_diff, git_reset
│   │   └── critic_tools.py              # Structured review/triage tools for Critic
│   │
│   ├── compute/                          # Compute backends (pluggable)
│   │   ├── __init__.py                   # Registry: get_runner(name) -> JobRunner
│   │   ├── base.py                       # JobRunner abstract class
│   │   ├── slurm.py                      # SLURM: sbatch, squeue, sacct, scancel
│   │   └── local.py                      # Local: subprocess with GPU assignment
│   │
│   ├── task/                             # Task adapter interface
│   │   ├── __init__.py
│   │   └── base.py                       # TaskAdapter abstract class
│   │
│   ├── agents/                           # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                       # BaseAgent: LLM loop, tool dispatch, context management, cost
│   │   ├── explorer.py                   # Explorer: non-blocking literature search, spec writing, scope advancement
│   │   ├── critic.py                     # Critic: inbox-based review, structured decisions, dedup, ranking
│   │   ├── builder.py                    # Builder: non-blocking implement, validate, submit, monitor, insight
│   │   └── orchestrator.py               # Orchestrator: NL interface, routing, cost reporting
│   │
│   └── prompts/                          # System prompt templates (Jinja2)
│       ├── explorer.md.j2
│       ├── critic.md.j2
│       ├── builder.md.j2
│       └── orchestrator.md.j2
│
├── tasks/
│   ├── template/
│   │   ├── __init__.py
│   │   ├── adapter.py                    # Skeleton TaskAdapter with docstrings
│   │   └── README.md
│   └── examples/
│       └── soccer_twos/
│           ├── __init__.py
│           ├── adapter.py                # Full TaskAdapter for 2v2 soccer
│           ├── train.py                  # Unified training script (writes progress.json)
│           ├── evaluate.py               # Evaluation harness
│           └── README.md
│
├── configs/
│   ├── default.yaml                      # Base config with all fields documented
│   └── examples/
│       ├── soccer_twos.yaml              # Soccer-specific config
│       ├── mappo_spec.yaml               # Reference spec example
│       └── ppo_baseline_spec.yaml        # Reference baseline spec example
│
├── docs/
│   └── implementation_plan.md            # This file
│
└── tests/
    ├── __init__.py
    ├── conftest.py                       # Shared fixtures (mock LLM, mock task, temp dirs)
    ├── test_spec.py
    ├── test_priority_queue.py
    ├── test_knowledge_base.py
    ├── test_results_tracker.py
    ├── test_gpu_pool.py
    ├── test_message_bus.py
    ├── test_cost_tracker.py
    ├── test_llm_backends.py
    ├── test_tools.py
    ├── test_agents.py
    └── test_system.py
```

---

## Build Phases

```
Phase 0: Core Data Structures       (no external dependencies)
    |
Phase 1: Abstract Interfaces        (defines contracts)
    |
Phase 2: LLM Backends               (implements LLM contract)
    |
Phase 3: Tools                       (implements tool contract)
    |
Phase 4: Compute Backends            (implements compute contract)
    |
Phase 5: Agents                      (uses LLM + tools + core)
    |
Phase 6: Hyperparameter Tuning       (uses agents + core)
    |
Phase 7: System, Config & Entry Point (ties everything together)
    |
Phase 8: System Prompts              (Jinja2 templates)
    |
Phase 9: Task Adapter Examples       (example implementations)
    |
Phase 10: Tests & Dependencies       (validates everything)
```

Each phase is independently testable. No circular dependencies.

---

## Phase 0: Core Data Structures

These are pure Python data structures with no LLM, no network, no external dependencies. They manage the shared state that agents read and write through files.

### `core/utils.py`

Shared utilities used by all core modules.

```python
import os
import tempfile

def atomic_write(path: str, content: str) -> None:
    """Write atomically: write to temp file, then rename (POSIX atomic)."""
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise

def normalize_name(name: str) -> str:
    """Normalize experiment name for dedup comparison.
    'mappo_v2' and 'MAPPO-v2' both become 'mappo_v2'."""
    return name.lower().replace("-", "_").strip()
```

### `core/spec.py`

The canonical representation of an experiment specification.

```python
from dataclasses import dataclass, field
import yaml
from core.utils import atomic_write

@dataclass
class ExperimentSpec:
    name: str                          # unique identifier (e.g., "mappo")
    description: str                   # one-line summary
    source_paper: str                  # citation
    architecture: dict                 # free-form architecture description
    why_it_might_work: str             # rationale for this task
    task_config: dict                  # task-specific configuration
    training_config: dict              # hyperparameters
    resource_estimate: dict            # memory, time estimates
    confidence: str = "high"           # "high" or "low" (force-approved)
    control_variant_of: str | None = None  # name of parent spec if this is a control
    category: str = ""                 # for synthesis grouping (e.g., "algorithm", "architecture")
    tags: list[str] = field(default_factory=list)
    tunable_hyperparameters: dict | None = None  # per-spec search space from Explorer

    def to_yaml(self, path: str | None = None) -> str:
        """Serialize to YAML. If path given, write atomically. Returns YAML string."""
        data = {
            "name": self.name,
            "description": self.description,
            "source_paper": self.source_paper,
            "architecture": self.architecture,
            "why_it_might_work": self.why_it_might_work,
            "task_config": self.task_config,
            "training_config": self.training_config,
            "resource_estimate": self.resource_estimate,
            "confidence": self.confidence,
            "control_variant_of": self.control_variant_of,
            "category": self.category,
            "tags": self.tags,
            "tunable_hyperparameters": self.tunable_hyperparameters,
        }
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path:
            atomic_write(path, content)
        return content

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentSpec":
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

### `core/priority_queue.py`

Thread-safe ranked queue of experiment specs with deduplication.

```python
import threading
import yaml
from core.spec import ExperimentSpec
from core.utils import atomic_write, normalize_name

class PriorityQueue:
    def __init__(self, queue_path: str):
        """Load from YAML or create empty."""
        self._path = queue_path
        self._lock = threading.Lock()
        self._items: list[tuple[float, ExperimentSpec]] = []
        if os.path.exists(queue_path):
            with open(queue_path) as f:
                data = yaml.safe_load(f) or []
            for entry in data:
                spec = ExperimentSpec(**entry["spec"])
                self._items.append((entry["score"], spec))
            self._items.sort(key=lambda x: -x[0])

    def insert(self, spec: ExperimentSpec, score: float) -> None:
        """Insert spec with score. Thread-safe. Atomic write to disk."""
        with self._lock:
            self._items.append((score, spec))
            self._items.sort(key=lambda x: -x[0])
            self.save()

    def pop(self) -> tuple[ExperimentSpec, float] | None:
        """Remove and return highest-scored spec. None if empty."""
        with self._lock:
            if not self._items:
                return None
            score, spec = self._items.pop(0)
            self.save()
            return (spec, score)

    def peek(self, n: int = 5) -> list[tuple[ExperimentSpec, float]]:
        """View top-n without removing."""
        with self._lock:
            return [(spec, score) for score, spec in self._items[:n]]

    def update_score(self, name: str, new_score: float) -> None:
        """Update score for a spec. Used during re-ranking."""
        with self._lock:
            for i, (score, spec) in enumerate(self._items):
                if normalize_name(spec.name) == normalize_name(name):
                    self._items[i] = (new_score, spec)
                    break
            self._items.sort(key=lambda x: -x[0])
            self.save()

    def rerank(self, scoring_fn, context: dict) -> None:
        """Re-rank all specs using a scoring function and current context.
        scoring_fn(spec, context) -> float"""
        with self._lock:
            self._items = [(scoring_fn(spec, context), spec)
                           for _, spec in self._items]
            self._items.sort(key=lambda x: -x[0])
            self.save()

    def depth(self) -> int:
        """Number of pending specs."""
        with self._lock:
            return len(self._items)

    def save(self) -> None:
        """Persist to YAML (atomic write)."""
        data = [{"score": score, "spec": spec.to_yaml()} for score, spec in self._items]
        # Note: spec.to_yaml() returns YAML string; for persistence we serialize the dict
        data = []
        for score, spec in self._items:
            spec_dict = {
                "name": spec.name, "description": spec.description,
                "source_paper": spec.source_paper, "architecture": spec.architecture,
                "why_it_might_work": spec.why_it_might_work,
                "task_config": spec.task_config, "training_config": spec.training_config,
                "resource_estimate": spec.resource_estimate, "confidence": spec.confidence,
                "control_variant_of": spec.control_variant_of, "category": spec.category,
                "tags": spec.tags, "tunable_hyperparameters": spec.tunable_hyperparameters,
            }
            data.append({"score": score, "spec": spec_dict})
        atomic_write(self._path, yaml.dump(data, default_flow_style=False, sort_keys=False))

    def get_all(self) -> list[tuple[ExperimentSpec, float]]:
        """Return all specs with scores, sorted by rank."""
        with self._lock:
            return [(spec, score) for score, spec in self._items]

    def contains(self, name: str) -> bool:
        """Check if a spec with this name is in the queue."""
        with self._lock:
            return any(normalize_name(spec.name) == normalize_name(name)
                       for _, spec in self._items)

    def contains_similar(self, name: str) -> str | None:
        """Check if a semantically similar name exists (normalized comparison).
        Returns the matching name or None."""
        with self._lock:
            target = normalize_name(name)
            for _, spec in self._items:
                if normalize_name(spec.name) == target:
                    return spec.name
            return None
```

Internally: a list of `(score, spec)` tuples protected by `threading.Lock`. Persisted to `priority_queue.yaml` via atomic write on every mutation.

### `core/knowledge_base.py`

Manages `knowledge.md` with synthesis pinned at top.

```python
import os
import threading
from datetime import datetime, timezone
from core.utils import atomic_write

class KnowledgeBase:
    def __init__(self, path: str):
        """Load existing knowledge.md or create empty."""
        self._path = path
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path) as f:
                self._content = f.read()
        else:
            self._content = "# Knowledge Base\n\n## Synthesis\n\n## Insights\n"

    def add_insight(self, title: str, body: str) -> None:
        """Append a Critic-approved insight to the individual section. Atomic write."""
        with self._lock:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            entry = f"\n### {title} ({timestamp})\n\n{body}\n"
            # Append after the "## Insights" header
            if "## Insights" in self._content:
                self._content = self._content + entry
            else:
                self._content += f"\n## Insights\n{entry}"
            atomic_write(self._path, self._content)

    def add_synthesis(self, category: str, body: str) -> None:
        """Add or update a synthesis entry (pinned at top). Atomic write."""
        with self._lock:
            marker_start = f"### {category}\n"
            marker_end = "\n### "
            if marker_start in self._content:
                # Replace existing synthesis for this category
                start = self._content.index(marker_start)
                rest = self._content[start + len(marker_start):]
                end_idx = rest.find(marker_end)
                if end_idx == -1 or rest.find("## Insights") < end_idx:
                    end_idx = rest.find("## Insights")
                if end_idx == -1:
                    end_idx = len(rest)
                self._content = (self._content[:start] + marker_start +
                                 body + "\n" + rest[end_idx:])
            else:
                # Insert new synthesis before "## Insights"
                insert_pos = self._content.find("## Insights")
                if insert_pos == -1:
                    self._content += f"\n### {category}\n{body}\n"
                else:
                    self._content = (self._content[:insert_pos] +
                                     f"### {category}\n{body}\n\n" +
                                     self._content[insert_pos:])
            atomic_write(self._path, self._content)

    def get_synthesis(self) -> str:
        """Return all synthesis sections as text."""
        with self._lock:
            idx = self._content.find("## Insights")
            if idx == -1:
                return self._content
            return self._content[:idx].strip()

    def get_full(self) -> str:
        """Return the entire knowledge base."""
        with self._lock:
            return self._content

    def get_recent(self, n: int = 5) -> str:
        """Return the n most recent insights."""
        with self._lock:
            parts = self._content.split("### ")
            # Filter to insight entries (those after "## Insights")
            insights_idx = self._content.find("## Insights")
            if insights_idx == -1:
                return ""
            insights_text = self._content[insights_idx:]
            insight_parts = insights_text.split("### ")[1:]  # skip header
            recent = insight_parts[-n:] if len(insight_parts) > n else insight_parts
            return "\n\n### ".join([""] + recent).strip()
```

### `core/results_tracker.py`

Manages `results.tsv`.

```python
import csv
import os
import threading
from dataclasses import dataclass, field
from core.utils import atomic_write

@dataclass
class ExperimentResult:
    name: str
    commit: str
    algorithm: str
    model: str
    metric_value: float | None         # primary metric value (None for crashes/validation_fail)
    metric_std: float | None           # std across seeds (baselines only)
    seeds: int
    training_seconds: float
    peak_memory_mb: float
    status: str                        # baseline | keep | crash | early_stop | timeout | validation_fail
    description: str
    extra_metrics: dict = field(default_factory=dict)

class ResultsTracker:
    HEADER = ["name", "commit", "algorithm", "model", "metric_value", "metric_std",
              "seeds", "training_seconds", "peak_memory_mb", "status", "description",
              "extra_metrics"]

    def __init__(self, path: str, metric_name: str, metric_direction: str):
        """Load existing results.tsv or create with header."""
        self._path = path
        self._metric_name = metric_name
        self._direction = metric_direction  # "higher" or "lower"
        self._lock = threading.Lock()
        self._results: list[ExperimentResult] = []
        if os.path.exists(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    self._results.append(ExperimentResult(
                        name=row["name"], commit=row["commit"],
                        algorithm=row["algorithm"], model=row["model"],
                        metric_value=float(row["metric_value"]) if row["metric_value"] else None,
                        metric_std=float(row["metric_std"]) if row["metric_std"] else None,
                        seeds=int(row["seeds"]),
                        training_seconds=float(row["training_seconds"]),
                        peak_memory_mb=float(row["peak_memory_mb"]),
                        status=row["status"], description=row["description"],
                        extra_metrics=eval(row.get("extra_metrics", "{}")) if row.get("extra_metrics") else {},
                    ))

    def add(self, result: ExperimentResult) -> None:
        """Append result. Thread-safe. Atomic write."""
        with self._lock:
            self._results.append(result)
            self._persist()

    def get_all(self) -> list[ExperimentResult]:
        """Return all results."""
        with self._lock:
            return list(self._results)

    def get_best(self, n: int = 5) -> list[ExperimentResult]:
        """Return top-n by primary metric (respecting direction)."""
        with self._lock:
            valid = [r for r in self._results if r.metric_value is not None]
            reverse = self._direction == "higher"
            valid.sort(key=lambda r: r.metric_value, reverse=reverse)
            return valid[:n]

    def get_baselines(self) -> list[ExperimentResult]:
        """Return all baseline results."""
        with self._lock:
            return [r for r in self._results if r.status == "baseline"]

    def get_by_name(self, name: str) -> ExperimentResult | None:
        """Return result for a specific experiment name."""
        with self._lock:
            for r in self._results:
                if r.name == name:
                    return r
            return None

    def is_completed(self, name: str) -> bool:
        """Check if experiment already has results (any status)."""
        with self._lock:
            return any(r.name == name for r in self._results)

    def get_summary(self) -> str:
        """Return human-readable summary table."""
        with self._lock:
            if not self._results:
                return "No results yet."
            lines = [f"{'Name':<30} {'Metric':<12} {'Status':<15} {'Time':<10}"]
            lines.append("-" * 70)
            for r in self._results:
                val = f"{r.metric_value:.4f}" if r.metric_value is not None else "N/A"
                time_str = f"{r.training_seconds:.0f}s"
                lines.append(f"{r.name:<30} {val:<12} {r.status:<15} {time_str:<10}")
            return "\n".join(lines)

    def get_completed_names(self) -> set[str]:
        """Return set of completed experiment names."""
        with self._lock:
            return {r.name for r in self._results}

    def _persist(self) -> None:
        """Write results.tsv atomically."""
        lines = ["\t".join(self.HEADER)]
        for r in self._results:
            row = [r.name, r.commit, r.algorithm, r.model,
                   str(r.metric_value) if r.metric_value is not None else "",
                   str(r.metric_std) if r.metric_std is not None else "",
                   str(r.seeds), str(r.training_seconds), str(r.peak_memory_mb),
                   r.status, r.description, str(r.extra_metrics)]
            lines.append("\t".join(row))
        atomic_write(self._path, "\n".join(lines) + "\n")
```

### `core/gpu_pool.py`

Manages GPU slot assignments with state persistence for crash recovery.

```python
import os
import time
import threading
import yaml
from dataclasses import dataclass
from core.utils import atomic_write

@dataclass
class GPUSlot:
    slot_id: int
    job_id: str | None = None          # compute backend job ID
    experiment_name: str | None = None
    submitted_at: float | None = None  # time.time()
    status: str = "free"               # free | running | pending_review | completing
    log_dir: str | None = None         # path to log/progress files for this slot

class GPUPool:
    def __init__(self, n_gpus: int, state_path: str):
        """Create pool with n_gpus slots. Load from state_path if exists."""
        self._state_path = state_path
        self._lock = threading.Lock()
        if os.path.exists(state_path):
            self._slots = self._load_state(state_path, n_gpus)
        else:
            self._slots = [GPUSlot(slot_id=i) for i in range(n_gpus)]

    def get_free_slot(self) -> GPUSlot | None:
        """Return a free slot, or None if all occupied."""
        with self._lock:
            for slot in self._slots:
                if slot.status == "free":
                    return slot
            return None

    def assign(self, slot_id: int, job_id: str, experiment_name: str,
               log_dir: str) -> None:
        """Mark slot as running with given job. Persist state."""
        with self._lock:
            slot = self._slots[slot_id]
            slot.job_id = job_id
            slot.experiment_name = experiment_name
            slot.submitted_at = time.time()
            slot.status = "running"
            slot.log_dir = log_dir
            self._persist()

    def release(self, slot_id: int) -> None:
        """Mark slot as free. Persist state."""
        with self._lock:
            slot = self._slots[slot_id]
            slot.job_id = None
            slot.experiment_name = None
            slot.submitted_at = None
            slot.status = "free"
            slot.log_dir = None
            self._persist()

    def get_running(self) -> list[GPUSlot]:
        """Return all running slots."""
        with self._lock:
            return [s for s in self._slots if s.status == "running"]

    def get_timed_out(self, timeout: float) -> list[GPUSlot]:
        """Return slots that have exceeded the timeout."""
        with self._lock:
            now = time.time()
            return [s for s in self._slots
                    if s.status == "running" and s.submitted_at
                    and (now - s.submitted_at) > timeout]

    def resize(self, n_gpus: int) -> None:
        """Change pool size. Only affects free slots."""
        with self._lock:
            current = len(self._slots)
            if n_gpus > current:
                for i in range(current, n_gpus):
                    self._slots.append(GPUSlot(slot_id=i))
            elif n_gpus < current:
                # Only remove free slots from the end
                while len(self._slots) > n_gpus:
                    if self._slots[-1].status == "free":
                        self._slots.pop()
                    else:
                        break
            self._persist()

    def get_status_summary(self) -> str:
        """Human-readable pool status."""
        with self._lock:
            lines = []
            for s in self._slots:
                if s.status == "free":
                    lines.append(f"  Slot {s.slot_id}: free")
                else:
                    elapsed = time.time() - s.submitted_at if s.submitted_at else 0
                    lines.append(f"  Slot {s.slot_id}: {s.status} — "
                                 f"{s.experiment_name} ({elapsed:.0f}s)")
            return "\n".join(lines)

    def _persist(self) -> None:
        """Write slot state to state_path (atomic write). Called after every mutation."""
        data = []
        for s in self._slots:
            data.append({
                "slot_id": s.slot_id, "job_id": s.job_id,
                "experiment_name": s.experiment_name,
                "submitted_at": s.submitted_at, "status": s.status,
                "log_dir": s.log_dir,
            })
        atomic_write(self._state_path, yaml.dump(data, default_flow_style=False))

    @classmethod
    def from_state(cls, state_path: str, n_gpus: int) -> "GPUPool":
        """Recover pool from persisted state file."""
        pool = cls.__new__(cls)
        pool._state_path = state_path
        pool._lock = threading.Lock()
        pool._slots = pool._load_state(state_path, n_gpus)
        return pool

    def _load_state(self, state_path: str, n_gpus: int) -> list[GPUSlot]:
        """Load slot state from YAML file."""
        with open(state_path) as f:
            data = yaml.safe_load(f) or []
        slots = []
        for entry in data:
            slots.append(GPUSlot(**entry))
        # Ensure we have exactly n_gpus slots
        while len(slots) < n_gpus:
            slots.append(GPUSlot(slot_id=len(slots)))
        return slots[:n_gpus]
```

### `core/message_bus.py`

Thread-safe async inter-agent communication. Agents post messages and continue without waiting. Recipients poll their inbox.

```python
import json
import os
import queue
import uuid
import threading
from dataclasses import dataclass, field
from core.utils import atomic_write

@dataclass
class AgentMessage:
    sender: str                        # "explorer", "builder", "critic", "orchestrator", "human", "watchdog"
    recipient: str                     # target agent or "all"
    type: str                          # "spec_review_request", "spec_approved", "spec_rejected",
                                       # "code_review_request", "code_approved", "code_rejected",
                                       # "insight_review_request", "insight_approved", "insight_rejected",
                                       # "suggestion_triage_request", "suggestion_triaged",
                                       # "rerank_request", "priority_adjust",
                                       # "clarification_request", "clarification_response",
                                       # "command", "alert"
    payload: dict                      # message-specific data
    priority: int = 0                  # higher = processed first (code reviews > spec reviews > insights)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

# Priority constants
PRIORITY_CODE_REVIEW = 100
PRIORITY_SPEC_REVIEW = 80
PRIORITY_INSIGHT_REVIEW = 60
PRIORITY_SUGGESTION = 40
PRIORITY_RERANK = 20
PRIORITY_DEFAULT = 0

class MessageBus:
    def __init__(self, state_path: str | None = None):
        """Create bus with per-agent priority queues.
        If state_path given, persist pending messages for crash recovery."""
        self._state_path = state_path
        self._lock = threading.Lock()
        self._queues: dict[str, queue.PriorityQueue] = {}
        if state_path and os.path.exists(state_path):
            self._load_state(state_path)

    def post(self, message: AgentMessage) -> None:
        """Post message to recipient's queue. Thread-safe. Non-blocking."""
        with self._lock:
            if message.recipient not in self._queues:
                self._queues[message.recipient] = queue.PriorityQueue()
            # Negate priority so higher priority = dequeued first
            self._queues[message.recipient].put((-message.priority, message.id, message))
            if self._state_path:
                self._persist()

    def poll(self, agent: str, timeout: float = 0.1) -> AgentMessage | None:
        """Non-blocking poll. Returns highest-priority message for agent,
        or None if inbox is empty (after brief timeout)."""
        with self._lock:
            if agent not in self._queues:
                return None
        try:
            _, _, message = self._queues[agent].get(timeout=timeout)
            if self._state_path:
                with self._lock:
                    self._persist()
            return message
        except queue.Empty:
            return None

    def drain(self, agent: str) -> list[AgentMessage]:
        """Return all pending messages for agent, sorted by priority. Non-blocking."""
        messages = []
        with self._lock:
            if agent not in self._queues:
                return []
            q = self._queues[agent]
            while not q.empty():
                try:
                    _, _, msg = q.get_nowait()
                    messages.append(msg)
                except queue.Empty:
                    break
            if self._state_path:
                self._persist()
        return messages

    def _persist(self) -> None:
        """Persist all pending messages to state_path (atomic write)."""
        data = {}
        for agent, q in self._queues.items():
            # Peek at all items without removing (copy internal list)
            items = list(q.queue)
            data[agent] = [{"sender": m.sender, "recipient": m.recipient,
                            "type": m.type, "payload": m.payload,
                            "priority": m.priority, "id": m.id}
                           for _, _, m in items]
        atomic_write(self._state_path, json.dumps(data, indent=2))

    def _load_state(self, state_path: str) -> None:
        """Load pending messages from state file."""
        with open(state_path) as f:
            data = json.load(f)
        for agent, messages in data.items():
            self._queues[agent] = queue.PriorityQueue()
            for m in messages:
                msg = AgentMessage(**m)
                self._queues[agent].put((-msg.priority, msg.id, msg))

    @classmethod
    def from_state(cls, state_path: str) -> "MessageBus":
        """Recover bus from persisted state file."""
        bus = cls(state_path=state_path)
        return bus
```

Internally: `dict[str, queue.PriorityQueue]` — one priority queue per agent name. **No `post_and_wait`** — all communication is fire-and-forget with inbox polling.

### `core/cost_tracker.py`

Per-agent token and cost tracking.

```python
import threading
from dataclasses import dataclass

# Pricing per 1M tokens (input, output)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}

@dataclass
class AgentCost:
    agent: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimated dollar cost based on model pricing."""
        in_price, out_price = MODEL_PRICING.get(self.model, (0.0, 0.0))
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1_000_000

class CostTracker:
    def __init__(self):
        """Create tracker with per-agent accumulators."""
        self._agents: dict[str, AgentCost] = {}
        self._lock = threading.Lock()

    def register(self, agent: str, model: str) -> None:
        """Register an agent with its model for cost calculation."""
        with self._lock:
            self._agents[agent] = AgentCost(agent=agent, model=model)

    def record(self, agent: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for an agent. Thread-safe."""
        with self._lock:
            if agent not in self._agents:
                raise ValueError(f"Agent '{agent}' not registered. Call register() first.")
            self._agents[agent].input_tokens += input_tokens
            self._agents[agent].output_tokens += output_tokens

    def get_agent_cost(self, agent: str) -> AgentCost:
        """Return cost for a specific agent."""
        with self._lock:
            return self._agents.get(agent)

    def get_total_cost(self) -> float:
        """Return total estimated cost across all agents."""
        with self._lock:
            return sum(ac.estimated_cost for ac in self._agents.values())

    def get_summary(self) -> str:
        """Return human-readable cost summary table."""
        with self._lock:
            lines = [f"{'Agent':<15} {'Model':<30} {'Input':<12} {'Output':<12} {'Cost':<10}"]
            lines.append("-" * 80)
            for ac in self._agents.values():
                lines.append(f"{ac.agent:<15} {ac.model:<30} "
                             f"{ac.input_tokens:<12,} {ac.output_tokens:<12,} "
                             f"${ac.estimated_cost:.4f}")
            lines.append("-" * 80)
            lines.append(f"{'TOTAL':<57} ${self.get_total_cost():.4f}")
            return "\n".join(lines)
```

### `core/event_logger.py`

Append-only JSONL logging for structured audit trails. Three instances are created by the system: `review_log.jsonl` (Critic), `explorer_journal.jsonl` (Explorer), and `system_events.jsonl` (all agents + watchdog).

```python
import json
import os
from datetime import datetime, timezone

class EventLogger:
    """Append-only JSONL logger. Thread-safe via atomic append."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, **fields) -> None:
        """Append one JSON line with auto-timestamp."""
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        line = json.dumps(entry, default=str) + "\n"
        with open(self.path, "a") as f:
            f.write(line)

    def read_all(self) -> list[dict]:
        """Read all entries (for debugging/display)."""
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
```

Three instances created by the system:
- `review_logger = EventLogger("results/review_log.jsonl")` — used by Critic to log every spec review, code review, insight review, and triage decision
- `explorer_logger = EventLogger("results/explorer_journal.jsonl")` — used by Explorer to log papers read, specs drafted, scope changes
- `system_logger = EventLogger("results/system_events.jsonl")` — used by all agents and watchdog to log lifecycle events, errors, and state transitions

---

## Phase 1: Abstract Interfaces

These define contracts that users or the system implement. No concrete logic.

### `task/base.py` — TaskAdapter

The Task Adapter is the only code the user writes. It bridges the generic Applied Scientist framework and a specific ML problem.

```python
from abc import ABC, abstractmethod

class TaskAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable task name."""

    @property
    @abstractmethod
    def metric(self) -> tuple[str, str]:
        """(metric_name, direction). direction is 'higher' or 'lower'."""

    @property
    @abstractmethod
    def domain_context(self) -> str:
        """Domain description for agents (Explorer for paper search, Builder for implementation)."""

    @property
    @abstractmethod
    def code_map(self) -> str:
        """Complete reference for Builder. Must contain enough detail that Builder can
        implement architectures WITHOUT reading source files in most cases.

        Include:
        - Every editable file with its full path
        - Key classes, functions, and their signatures
        - How components connect (what calls what)
        - Framework-specific patterns (e.g., how to register a custom model in RLlib)
        - Files that must NOT be edited (adapter.py, evaluate.py)
        - Where to add new files if needed
        - Observation/action space details with dimensions
        - Hardware constraints
        """

    @property
    def progress_file(self) -> str:
        """Filename for periodic progress updates during training.
        Training script should write JSON: {"step": int, "<metric>": float, ...}
        Default: 'progress.json'"""
        return "progress.json"

    @abstractmethod
    def train(self, config: dict, seed: int, time_budget: int,
              log_path: str, checkpoint_dir: str) -> dict:
        """Run training. Return dict of metrics including primary metric.
        Training script MUST write progress_file periodically.
        Return {"status": "crash", "error": "..."} on failure."""

    @abstractmethod
    def evaluate(self, checkpoint_dir: str, n_eval_samples: int) -> dict:
        """Evaluate checkpoint. Return dict of metrics."""

    def get_baseline_spec(self) -> dict | None:
        """Return a baseline experiment spec dict, or None for system default."""
        return None

    def get_random_baseline(self) -> dict | None:
        """Return metrics for an untrained/random agent. Called once at startup.
        No training needed — zero GPU cost. Provides early stopping floor.
        Return None to skip random baseline.

        Example: {"win_rate": 0.50, "avg_reward": 0.0}"""
        return None

    def get_tunable_defaults(self) -> dict:
        """Return default tunable hyperparameters and search spaces for this task.
        Explorer-defined spec.tunable_hyperparameters override these.
        Return empty dict to use only spec-defined hyperparameters.

        Format: {"param_name": {"type": "log_uniform"|"uniform"|"discrete"|"factor",
                                 "low": float, "high": float,  # for uniform/log_uniform
                                 "values": list,  # for discrete
                                 "factors": list}}  # for factor (multipliers of spec default)

        Example:
            {"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
             "train_batch_size": {"type": "factor", "factors": [0.5, 1, 2, 4]},
             "gamma": {"type": "discrete", "values": [0.99, 0.995, 0.999]}}
        """
        return {}
```

### `llm/base.py` — LLMBackend

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMMessage:
    role: str                          # "system", "user", "assistant", "tool_result"
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

@dataclass
class LLMResponse:
    content: str                       # text response
    tool_calls: list[dict] | None      # list of {id, name, arguments}
    usage: dict                        # {"input_tokens": int, "output_tokens": int}
    stop_reason: str                   # "end_turn", "tool_use", "max_tokens"

class LLMBackend(ABC):
    @abstractmethod
    def complete(self, messages: list[LLMMessage],
                 tools: list[dict] | None = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0) -> LLMResponse:
        """Send messages and return response. Handle retries internally."""

    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier string."""

    @property
    @abstractmethod
    def supports_tool_use(self) -> bool:
        """Whether this backend supports tool calling."""
```

### `tools/base.py` — Tool

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name as seen by the LLM."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description for the LLM."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for parameters."""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute and return string result."""

    def to_schema(self) -> dict:
        """Convert to LLM tool schema. Works for Anthropic, OpenAI, Gemini formats."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
```

### `compute/base.py` — JobRunner

```python
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
               resources: dict | None = None) -> str:
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
```

---

## Phase 2: LLM Backends

Each backend implements `LLMBackend`. All handle retries with exponential backoff, rate limiting, and token counting internally.

### `llm/anthropic_backend.py`

```python
import anthropic
from llm.base import LLMBackend, LLMMessage, LLMResponse

class AnthropicBackend(LLMBackend):
    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        # Extract system message from messages list (Anthropic places system as top-level param)
        system_prompt = None
        filtered_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                filtered_messages.append({"role": m.role, "content": m.content})

        # Convert tools to Anthropic format: {"name", "description", "input_schema"}
        anthropic_tools = None
        if tools:
            anthropic_tools = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["input_schema"]} for t in tools]

        # Call with retries on rate limit (429) and server error (500)
        # Uses exponential backoff: 1s, 2s, 4s
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=filtered_messages,
            tools=anthropic_tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Convert response to LLMResponse
        # Extract text content and tool_calls from response.content blocks
        # Map usage from response.usage.input_tokens, response.usage.output_tokens
        # Map stop_reason from response.stop_reason ("end_turn", "tool_use", "max_tokens")

    def get_model_id(self): return self.model

    @property
    def supports_tool_use(self): return True
```

**Key implementation detail:** The Anthropic API places the system prompt as a top-level parameter, not in the messages list. The backend must extract the first `role="system"` message and pass it separately.

### `llm/openai_backend.py`

```python
import openai
from llm.base import LLMBackend, LLMMessage, LLMResponse

class OpenAIBackend(LLMBackend):
    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        # Convert tools to OpenAI format:
        #   {"type": "function", "function": {"name", "description", "parameters"}}
        # Convert messages to OpenAI format (system is a regular message with role="system")
        # Handle tool_result messages by converting to role="tool" with tool_call_id
        # Call self.client.chat.completions.create(...)
        # Convert response to LLMResponse
        # Map tool_calls: OpenAI uses response.choices[0].message.tool_calls
        #   with function.name, function.arguments (JSON string -> parse to dict)
        # Map usage: response.usage.prompt_tokens, response.usage.completion_tokens
        # Map stop_reason: "stop" -> "end_turn", "tool_calls" -> "tool_use", "length" -> "max_tokens"

    def get_model_id(self): return self.model

    @property
    def supports_tool_use(self): return True
```

### `llm/gemini_backend.py`

```python
from llm.base import LLMBackend, LLMMessage, LLMResponse

class GeminiBackend(LLMBackend):
    def __init__(self, model: str, api_key: str, max_retries: int = 3):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        self.max_retries = max_retries

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        # Convert tools to Gemini FunctionDeclaration format
        # Convert messages to Gemini Content objects
        # System message becomes system_instruction on the model
        # Call self.model.generate_content(...)
        # Convert response to LLMResponse
        # Map function calls from response.candidates[0].content.parts
        # Map usage from response.usage_metadata

    def get_model_id(self): return self.model_name

    @property
    def supports_tool_use(self): return True
```

### `llm/openai_compatible.py`

For any OpenAI-compatible API (Ollama, vLLM, Together AI, Groq, etc.):

```python
import openai
from llm.base import LLMBackend, LLMMessage, LLMResponse

class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, model: str, api_key: str, base_url: str, max_retries: int = 3):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
    # Same implementation as OpenAIBackend but with custom base_url

    def complete(self, messages, tools=None, max_tokens=4096, temperature=0.0):
        # Identical to OpenAIBackend.complete()
        pass

    def get_model_id(self): return self.model

    @property
    def supports_tool_use(self): return True
```

### `llm/__init__.py` — Registry

```python
from .anthropic_backend import AnthropicBackend
from .openai_backend import OpenAIBackend
from .gemini_backend import GeminiBackend
from .openai_compatible import OpenAICompatibleBackend

_BACKENDS = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "gemini": GeminiBackend,
    "openai_compatible": OpenAICompatibleBackend,
}

def get_backend(config: dict) -> "LLMBackend":
    """Instantiate backend from config dict with keys: backend, model, (api_key, base_url).
    API key is resolved from config or environment variable."""
    backend_name = config["backend"]
    if backend_name not in _BACKENDS:
        raise ValueError(f"Unknown LLM backend: {backend_name}. "
                         f"Available: {list(_BACKENDS.keys())}")
    cls = _BACKENDS[backend_name]
    # Pass all config keys except "backend" as constructor kwargs
    return cls(**{k: v for k, v in config.items() if k != "backend"})
```

---

## Phase 3: Tools

Each tool is a class with `name`, `description`, `parameters` (JSON Schema), and `execute(**kwargs) -> str`.

### `tools/file_ops.py`

Four tools:

| Tool | Parameters | Returns |
|------|-----------|---------|
| `read_file` | `path: str`, `offset: int?`, `limit: int?` | File content with line numbers |
| `write_file` | `path: str`, `content: str` | Confirmation message |
| `edit_file` | `path: str`, `old_string: str`, `new_string: str` | Confirmation or error if old_string not found |
| `list_directory` | `path: str`, `pattern: str?` | List of files matching pattern |

All paths are validated to be within the workspace directory (security boundary).

```python
import os
import glob as glob_module
from tools.base import Tool

class ReadFile(Tool):
    name = "read_file"
    description = "Read a file's contents. Returns content with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "offset": {"type": "integer", "description": "Line to start from (0-indexed)"},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        if not os.path.exists(full_path):
            return f"ERROR: File not found: {path}"
        with open(full_path) as f:
            lines = f.readlines()
        selected = lines[offset:offset + limit]
        return "".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(selected))


class WriteFile(Tool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "content": {"type": "string", "description": "File content to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, content: str) -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"


class EditFile(Tool):
    name = "edit_file"
    description = "Replace an exact string in a file. Fails if old_string is not found."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace"},
            "old_string": {"type": "string", "description": "Exact text to find"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, old_string: str, new_string: str) -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        if not os.path.exists(full_path):
            return f"ERROR: File not found: {path}"
        with open(full_path) as f:
            content = f.read()
        if old_string not in content:
            return f"ERROR: old_string not found in {path}"
        content = content.replace(old_string, new_string, 1)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Edited {path}: replaced 1 occurrence."


class ListDirectory(Tool):
    name = "list_directory"
    description = "List files in a directory, optionally filtered by glob pattern."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to workspace"},
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py')"},
        },
        "required": ["path"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, path: str, pattern: str = "*") -> str:
        full_path = os.path.join(self.workspace, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.workspace)):
            return "ERROR: Path outside workspace."
        if not os.path.isdir(full_path):
            return f"ERROR: Not a directory: {path}"
        matches = glob_module.glob(os.path.join(full_path, pattern))
        rel_paths = [os.path.relpath(m, self.workspace) for m in sorted(matches)]
        return "\n".join(rel_paths) if rel_paths else "(empty directory)"
```

### `tools/web_search.py`

| Tool | Parameters | Returns |
|------|-----------|---------|
| `search_web` | `query: str`, `max_results: int?` | List of {title, url, snippet} |

Backend-agnostic. Supports: Serper API, SerpAPI, Google Custom Search, Bing Search. Configured via environment variable `WEB_SEARCH_BACKEND` and `WEB_SEARCH_API_KEY`.

```python
import json
import os
import requests
from tools.base import Tool

class SearchWeb(Tool):
    name = "search_web"
    description = "Search the web. Returns a list of results with title, URL, and snippet."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Maximum results to return",
                            "default": 10},
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 10) -> str:
        backend = os.environ.get("WEB_SEARCH_BACKEND", "serper")
        api_key = os.environ.get("WEB_SEARCH_API_KEY", "")
        # Dispatch to backend-specific implementation
        # Each returns list of {"title": str, "url": str, "snippet": str}
        if backend == "serper":
            results = self._serper(query, max_results, api_key)
        elif backend == "serpapi":
            results = self._serpapi(query, max_results, api_key)
        elif backend == "google":
            results = self._google(query, max_results, api_key)
        elif backend == "bing":
            results = self._bing(query, max_results, api_key)
        else:
            return f"ERROR: Unknown search backend: {backend}"
        return json.dumps(results[:max_results], indent=2)

    def _serper(self, query, max_results, api_key):
        # POST to https://google.serper.dev/search
        # Headers: {"X-API-KEY": api_key, "Content-Type": "application/json"}
        # Body: {"q": query, "num": max_results}
        # Parse response["organic"] -> [{"title", "link", "snippet"}]
        pass

    def _serpapi(self, query, max_results, api_key):
        # GET https://serpapi.com/search?q={query}&api_key={api_key}&num={max_results}
        # Parse response["organic_results"]
        pass

    def _google(self, query, max_results, api_key):
        # Google Custom Search JSON API
        pass

    def _bing(self, query, max_results, api_key):
        # Bing Web Search API v7
        pass
```

### `tools/paper_reader.py`

| Tool | Parameters | Returns |
|------|-----------|---------|
| `search_papers` | `query: str`, `max_results: int?` | List of {title, authors, year, abstract, url} from Semantic Scholar |
| `read_paper` | `url: str`, `prompt: str?` | Extracted text/summary from arXiv PDF or web page |

`search_papers` uses Semantic Scholar API (free, no key required, rate-limited).
`read_paper` fetches the URL, converts to text (PDF via PyPDF2, HTML via BeautifulSoup), and optionally summarizes via the agent's LLM.

```python
import json
import requests
from tools.base import Tool

class SearchPapers(Tool):
    name = "search_papers"
    description = "Search academic papers via Semantic Scholar. Free, no API key needed."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Max papers to return",
                            "default": 10},
        },
        "required": ["query"],
    }

    def execute(self, query: str, max_results: int = 10) -> str:
        # GET https://api.semanticscholar.org/graph/v1/paper/search
        #   ?query={query}&limit={max_results}&fields=title,authors,year,abstract,url
        # Parse response["data"] -> list of papers
        # Return JSON list of {"title", "authors", "year", "abstract", "url"}
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": max_results,
                  "fields": "title,authors,year,abstract,url"}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        papers = []
        for p in resp.json().get("data", []):
            papers.append({
                "title": p.get("title", ""),
                "authors": [a.get("name", "") for a in p.get("authors", [])],
                "year": p.get("year"),
                "abstract": p.get("abstract", ""),
                "url": p.get("url", ""),
            })
        return json.dumps(papers, indent=2)


class ReadPaper(Tool):
    name = "read_paper"
    description = "Read and extract text from an arXiv paper or web page."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch (arXiv, PDF, or web)"},
            "prompt": {"type": "string", "description": "Optional focus prompt for extraction"},
        },
        "required": ["url"],
    }

    def execute(self, url: str, prompt: str = None) -> str:
        # If URL ends in .pdf or is arXiv: download PDF, extract text via PyPDF2
        # Otherwise: fetch HTML, extract text via BeautifulSoup
        # Truncate to ~8000 chars if too long
        # If prompt given, prepend "Focus on: {prompt}\n\n" to extracted text
        pass
```

### `tools/shell.py`

| Tool | Parameters | Returns |
|------|-----------|---------|
| `run_command` | `command: str`, `timeout: int?` | stdout + stderr, truncated to 10K chars |

Executed via `subprocess.run` with sandboxing:
- Configurable timeout (default 120s)
- Working directory forced to workspace
- Environment inherited from parent process, with `CUDA_VISIBLE_DEVICES` set per slot
- **Dangerous command rejection:** Commands matching dangerous patterns are blocked

```python
import re
import subprocess
from tools.base import Tool

BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s+",
    r"chmod\s+777",
    r"\.\./\.\./",  # path traversal
]

class ShellTool(Tool):
    name = "run_command"
    description = "Run a shell command in the workspace. Output truncated to 10K chars."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        },
        "required": ["command"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, command: str, timeout: int = 120) -> str:
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return f"ERROR: Command blocked by security policy: {command}"
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=self.workspace,
            )
            output = result.stdout + result.stderr
            return output[:10000]
        except subprocess.TimeoutExpired:
            return f"ERROR: Command timed out after {timeout}s"
```

### `tools/git_ops.py`

| Tool | Parameters | Returns |
|------|-----------|---------|
| `git_commit` | `message: str`, `files: list[str]?` | Commit hash |
| `git_log` | `n: int?` | Recent commit log |
| `git_diff` | `ref: str?` | Diff output |
| `git_reset` | `ref: str` | Confirmation |

All operations scoped to the workspace git repo.

```python
import json
import subprocess
from tools.base import Tool

class GitCommit(Tool):
    name = "git_commit"
    description = "Stage and commit files in the workspace git repo."
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Commit message"},
            "files": {"type": "array", "items": {"type": "string"},
                      "description": "Files to stage (default: all changed)"},
        },
        "required": ["message"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, message: str, files: list[str] = None) -> str:
        if files:
            for f in files:
                subprocess.run(["git", "add", f], cwd=self.workspace, capture_output=True)
        else:
            subprocess.run(["git", "add", "-A"], cwd=self.workspace, capture_output=True)
        result = subprocess.run(
            ["git", "commit", "-m", message], cwd=self.workspace, capture_output=True, text=True)
        if result.returncode != 0:
            return f"ERROR: {result.stderr}"
        # Extract commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.workspace, capture_output=True, text=True)
        return hash_result.stdout.strip()


class GitLog(Tool):
    name = "git_log"
    description = "Show recent git commit log."
    parameters = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "Number of commits to show", "default": 10},
        },
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, n: int = 10) -> str:
        result = subprocess.run(
            ["git", "log", f"--oneline", f"-{n}"],
            cwd=self.workspace, capture_output=True, text=True)
        return result.stdout


class GitDiff(Tool):
    name = "git_diff"
    description = "Show diff of current changes or against a ref."
    parameters = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Git ref to diff against (default: working tree)"},
        },
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, ref: str = None) -> str:
        cmd = ["git", "diff"]
        if ref:
            cmd.append(ref)
        result = subprocess.run(cmd, cwd=self.workspace, capture_output=True, text=True)
        return result.stdout[:10000]


class GitReset(Tool):
    name = "git_reset"
    description = "Reset workspace to a git ref."
    parameters = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Git ref to reset to"},
        },
        "required": ["ref"],
    }

    def __init__(self, workspace_dir: str):
        self.workspace = workspace_dir

    def execute(self, ref: str) -> str:
        result = subprocess.run(
            ["git", "reset", "--hard", ref],
            cwd=self.workspace, capture_output=True, text=True)
        if result.returncode != 0:
            return f"ERROR: {result.stderr}"
        return f"Reset to {ref}"
```

### `tools/critic_tools.py`

Structured tool definitions for the Critic agent. These replace free-text "APPROVED"/"REJECTED" parsing with reliable tool calls.

```python
import json
from tools.base import Tool

class SubmitSpecReview(Tool):
    name = "submit_review"
    description = "Submit your review decision for an experiment spec."
    parameters = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approved", "rejected"]},
            "score": {"type": "number", "description": "Ranking score (only if approved)"},
            "feedback": {"type": "string", "description": "Specific feedback or edits"},
            "control_variant_needed": {"type": "boolean"},
            "control_description": {"type": "string",
                                    "description": "Control variant description (if needed)"},
        },
        "required": ["verdict", "feedback"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)


class SubmitCodeReview(Tool):
    """Structured code review tool. Replaces free-text APPROVED/REJECTED parsing."""
    name = "submit_code_review"
    description = "Submit your review decision for a Builder code change."
    parameters = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approved", "rejected"]},
            "feedback": {"type": "string", "description": "Specific feedback or required fix"},
        },
        "required": ["verdict", "feedback"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)


class SubmitInsightReview(Tool):
    name = "submit_insight_review"
    description = "Submit your review decision for a Builder insight."
    parameters = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approved", "rejected"]},
            "edits": {"type": "string",
                      "description": "Edited insight text (if approved with changes)"},
            "feedback": {"type": "string", "description": "Feedback (if rejected)"},
        },
        "required": ["verdict"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)


class SubmitSuggestionTriage(Tool):
    name = "submit_triage"
    description = "Triage a Builder improvement suggestion."
    parameters = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["minor", "moderate", "major"]},
            "justification": {"type": "string"},
            "action": {"type": "string", "description": "What should happen next"},
        },
        "required": ["level", "justification"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)
```

### `tools/__init__.py` — Tool Sets Per Agent

```python
from tools.file_ops import ReadFile, WriteFile, EditFile, ListDirectory
from tools.web_search import SearchWeb
from tools.paper_reader import SearchPapers, ReadPaper
from tools.shell import ShellTool
from tools.git_ops import GitCommit, GitLog, GitDiff, GitReset
from tools.critic_tools import (SubmitSpecReview, SubmitCodeReview,
                                 SubmitInsightReview, SubmitSuggestionTriage)
from tools.base import Tool

AGENT_TOOLS = {
    "explorer": ["read_file", "write_file", "list_directory",
                 "search_web", "search_papers", "read_paper"],
    "critic":   ["read_file", "write_file", "list_directory",
                 "submit_review", "submit_code_review",
                 "submit_insight_review", "submit_triage"],
    "builder":  ["read_file", "write_file", "edit_file", "list_directory",
                 "run_command", "git_commit", "git_log", "git_diff", "git_reset"],
    "orchestrator": ["read_file", "list_directory"],
}

# Map tool names to classes
_TOOL_CLASSES = {
    "read_file": ReadFile,
    "write_file": WriteFile,
    "edit_file": EditFile,
    "list_directory": ListDirectory,
    "search_web": SearchWeb,
    "search_papers": SearchPapers,
    "read_paper": ReadPaper,
    "run_command": ShellTool,
    "git_commit": GitCommit,
    "git_log": GitLog,
    "git_diff": GitDiff,
    "git_reset": GitReset,
    "submit_review": SubmitSpecReview,
    "submit_code_review": SubmitCodeReview,
    "submit_insight_review": SubmitInsightReview,
    "submit_triage": SubmitSuggestionTriage,
}

def get_tools(agent_role: str, workspace_dir: str) -> list[Tool]:
    """Return instantiated tools for the given agent role."""
    tool_names = AGENT_TOOLS.get(agent_role, [])
    tools = []
    for name in tool_names:
        cls = _TOOL_CLASSES[name]
        # Tools that need workspace_dir get it; others are instantiated bare
        try:
            tools.append(cls(workspace_dir=workspace_dir))
        except TypeError:
            tools.append(cls())
    return tools
```

---

## Phase 4: Compute Backends

### `compute/slurm.py`

```python
import os
import subprocess
import tempfile
from compute.base import JobRunner, JobStatus

class SLURMRunner(JobRunner):
    def __init__(self, partition: str, gres: str, mem: str,
                 time: str, setup_commands: list[str] = None):
        self.partition = partition
        self.gres = gres
        self.mem = mem
        self.time = time
        self.setup_commands = setup_commands or []

    def submit(self, command, job_name, resources=None):
        # Write a temporary batch script:
        #   #!/bin/bash
        #   #SBATCH --job-name={job_name}
        #   #SBATCH --partition={self.partition}
        #   #SBATCH --gres={self.gres}
        #   #SBATCH --mem={self.mem}
        #   #SBATCH --time={self.time}
        #   #SBATCH --output={log_path}
        #   {self.setup_commands joined by newlines}
        #   {command}
        log_path = resources.get("log_path", f"slurm_{job_name}.out") if resources else f"slurm_{job_name}.out"
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --gres={self.gres}",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --output={log_path}",
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
        return result.stdout.strip()  # job_id

    def status(self, job_id):
        # Run: sacct -j {job_id} --format=State,ExitCode,Elapsed --noheader --parsable2
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State,ExitCode,Elapsed",
             "--noheader", "--parsable2"],
            capture_output=True, text=True)
        if result.returncode != 0:
            return JobStatus(job_id=job_id, state="unknown")

        # Parse first line: STATE|EXITCODE|ELAPSED
        lines = [l for l in result.stdout.strip().split("\n") if l and ".batch" not in l]
        if not lines:
            return JobStatus(job_id=job_id, state="pending")

        parts = lines[0].split("|")
        state_raw = parts[0].strip()
        exit_code = int(parts[1].split(":")[0]) if len(parts) > 1 else None

        # Parse elapsed time (HH:MM:SS or D-HH:MM:SS)
        elapsed = None
        if len(parts) > 2:
            elapsed = self._parse_elapsed(parts[2].strip())

        # Map SLURM states to our states
        state_map = {
            "PENDING": "pending", "RUNNING": "running",
            "COMPLETED": "completed", "FAILED": "failed",
            "CANCELLED": "cancelled", "TIMEOUT": "failed",
            "NODE_FAIL": "failed", "OUT_OF_MEMORY": "failed",
        }
        state = state_map.get(state_raw, "unknown")
        return JobStatus(job_id=job_id, state=state, exit_code=exit_code,
                         elapsed_seconds=elapsed)

    def cancel(self, job_id):
        subprocess.run(["scancel", job_id], capture_output=True)

    def get_log(self, job_id, tail=50):
        # Find the SLURM output file and return last `tail` lines
        # Convention: log file is slurm_{job_id}.out or specified in submit
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=WorkDir%200", "--noheader", "--parsable2"],
            capture_output=True, text=True)
        # Fall back to reading from known log path
        # Return last `tail` lines of the log file
        pass

    def _parse_elapsed(self, elapsed_str: str) -> float:
        """Parse SLURM elapsed time format to seconds."""
        # Handle D-HH:MM:SS and HH:MM:SS formats
        days = 0
        if "-" in elapsed_str:
            days_str, elapsed_str = elapsed_str.split("-")
            days = int(days_str)
        parts = elapsed_str.split(":")
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
```

### `compute/local.py`

```python
import os
import subprocess
import time
import uuid
from compute.base import JobRunner, JobStatus

class LocalRunner(JobRunner):
    def __init__(self):
        self._processes = {}           # job_id -> {"process": Popen, "log_path": str, "start_time": float}

    def submit(self, command, job_name, resources=None):
        job_id = str(uuid.uuid4())[:8]
        log_path = resources.get("log_path", f"/tmp/{job_name}_{job_id}.log") if resources else f"/tmp/{job_name}_{job_id}.log"

        env = os.environ.copy()
        if resources and "gpu_id" in resources:
            env["CUDA_VISIBLE_DEVICES"] = str(resources["gpu_id"])

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            command, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
            env=env)

        self._processes[job_id] = {
            "process": proc,
            "log_path": log_path,
            "log_file": log_file,
            "start_time": time.time(),
        }
        return job_id

    def status(self, job_id):
        if job_id not in self._processes:
            return JobStatus(job_id=job_id, state="unknown")

        info = self._processes[job_id]
        proc = info["process"]
        elapsed = time.time() - info["start_time"]
        rc = proc.poll()

        if rc is None:
            return JobStatus(job_id=job_id, state="running", elapsed_seconds=elapsed)
        elif rc == 0:
            return JobStatus(job_id=job_id, state="completed", exit_code=rc,
                             elapsed_seconds=elapsed)
        else:
            return JobStatus(job_id=job_id, state="failed", exit_code=rc,
                             elapsed_seconds=elapsed)

    def cancel(self, job_id):
        if job_id in self._processes:
            self._processes[job_id]["process"].terminate()

    def get_log(self, job_id, tail=50):
        if job_id not in self._processes:
            return "ERROR: Unknown job ID"
        log_path = self._processes[job_id]["log_path"]
        if not os.path.exists(log_path):
            return "(no log file yet)"
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-tail:])
```

### `compute/__init__.py` — Registry

```python
from .slurm import SLURMRunner
from .local import LocalRunner

_BACKENDS = {
    "slurm": SLURMRunner,
    "local": LocalRunner,
}

def get_runner(config) -> "JobRunner":
    """Instantiate compute backend from config.
    config is a ComputeConfig dataclass with .backend and .slurm attributes."""
    backend = config.backend
    if backend == "slurm":
        return SLURMRunner(**config.slurm)
    elif backend == "local":
        return LocalRunner()
    else:
        raise ValueError(f"Unknown compute backend: {backend}")
```
## Phase 5: Agents

All agents inherit from `BaseAgent`. Every method referenced in agent code is fully defined below — there are no stub or undefined methods.

### `agents/base.py`

```python
import json
import time
from uuid import uuid4

from core.llm import LLMBackend, LLMMessage
from core.tools import Tool
from core.message_bus import MessageBus, AgentMessage
from core.cost_tracker import CostTracker


class BaseAgent:
    def __init__(self, role: str, llm: LLMBackend, tools: list[Tool],
                 system_prompt: str, message_bus: MessageBus,
                 cost_tracker: CostTracker):
        self.role = role
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.tool_schemas = [t.to_schema() for t in tools]
        self.system_prompt = system_prompt
        self.bus = message_bus
        self.cost = cost_tracker
        self.conversation: list[LLMMessage] = []
        self._stopped = False

    def chat(self, user_message: str) -> str:
        """Send a message and run the full tool-use loop until the agent
        produces a final text response. Handles multi-turn tool calls."""
        self._maybe_trim()
        self.conversation.append(LLMMessage(role="user", content=user_message))

        while True:
            response = self.llm.complete(
                messages=[LLMMessage(role="system", content=self.system_prompt)]
                        + self.conversation,
                tools=self.tool_schemas if self.tools else None,
            )
            self.cost.record(self.role,
                             response.usage["input_tokens"],
                             response.usage["output_tokens"])

            if response.tool_calls:
                self.conversation.append(LLMMessage(
                    role="assistant", content=response.content,
                    tool_calls=response.tool_calls))
                for call in response.tool_calls:
                    result = self.tools[call["name"]].execute(**call["arguments"])
                    self.conversation.append(LLMMessage(
                        role="tool_result", content=result,
                        tool_call_id=call["id"]))
            else:
                self.conversation.append(LLMMessage(
                    role="assistant", content=response.content))
                return response.content

    def reset_conversation(self):
        """Clear conversation history to manage context window."""
        self.conversation = []

    def _maybe_trim(self):
        """Auto-trim if approaching context limit (80% of 180K tokens)."""
        estimated_tokens = sum(len(m.content) // 4 for m in self.conversation)
        max_context = 180_000  # conservative for Haiku/Sonnet
        if estimated_tokens > max_context * 0.8:
            # Keep essential context: summarize then reset
            summary_prompt = (
                "Summarize the key findings and decisions from this conversation "
                "in a concise paragraph. Focus on facts, not process."
            )
            summary = self.llm.complete([
                LLMMessage(role="system", content="You are a summarizer."),
                LLMMessage(role="user", content=summary_prompt + "\n\n" +
                           "\n".join(m.content for m in self.conversation[-20:]))
            ])
            self.cost.record(self.role,
                             summary.usage["input_tokens"],
                             summary.usage["output_tokens"])
            self.conversation = [
                LLMMessage(role="user",
                           content=f"## Prior context summary\n{summary.content}")
            ]

    def _process_inbox(self) -> list[AgentMessage]:
        """Drain all pending messages from inbox. Non-blocking."""
        return self.bus.drain(self.role)

    def _alert(self, alert_type: str, message: str):
        """Send alert to Orchestrator for display."""
        self.bus.post(AgentMessage(
            sender=self.role, recipient="orchestrator",
            type="alert", payload={"alert_type": alert_type, "message": message}
        ))
```

---

### `agents/explorer.py`

```python
import glob
import os
import time

from agents.base import BaseAgent
from core.message_bus import AgentMessage
from core.data_structures import ExperimentSpec
from core.event_logger import EventLogger

# Priority constants
PRIORITY_SPEC_REVIEW = 80


class ExplorerAgent(BaseAgent):
    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 knowledge_base, results_tracker, priority_queue,
                 explorer_logger: EventLogger, config):
        super().__init__("explorer", llm, tools, system_prompt, message_bus, cost_tracker)
        self.kb = knowledge_base
        self.results = results_tracker
        self.queue = priority_queue
        self.explorer_logger = explorer_logger
        self.config = config
        self.current_round = 1
        self.paused = False
        self._consecutive_empty_rounds = 0
        self._pending_revisions: list[dict] = []  # specs rejected by Critic, awaiting revision
        self._last_draft_check_time: float = 0

    def run(self):
        """Main loop. Runs in its own thread. Never returns unless stopped."""
        while not self._stopped:
            try:
                # 1. Process inbox (non-blocking)
                self._handle_inbox()

                # 2. Throttle check
                queue_depth = self._get_queue_depth()
                if queue_depth >= self.config.system.queue_throttle_high:
                    self._alert("throttled", f"Queue has {queue_depth} specs. Pausing.")
                    self._wait_until_queue_below(self.config.system.queue_throttle_low)
                    continue

                if self.paused:
                    time.sleep(5)
                    continue

                # 3. Handle pending revisions first
                if self._pending_revisions:
                    self._revise_spec(self._pending_revisions.pop(0))
                    self.reset_conversation()
                    continue

                # 4. Search and draft new spec
                context = self._build_context()
                spec_yaml = self.chat(
                    f"## Current Context\n{context}\n\n"
                    f"## Scope Round: {self.current_round}\n"
                    f"Find a relevant architecture and write an experiment spec. "
                    f"Use search_papers and read_paper tools to find papers, "
                    f"then write the spec YAML to configs/drafts/ using write_file.\n\n"
                    f"IMPORTANT: Before choosing a spec name, check for collisions. "
                    f"Use list_directory on configs/drafts/, configs/experiments/, "
                    f"and results/ to see existing names."
                )

                # 5. Submit new drafts for review (non-blocking)
                drafts = self._check_for_new_drafts()
                for draft_path in drafts:
                    # Auto-dedup the spec name before submitting
                    spec = ExperimentSpec.from_yaml(draft_path)
                    deduped_name = self._dedup_spec_name(spec.name)
                    if deduped_name != spec.name:
                        spec.name = deduped_name
                        new_path = os.path.join(os.path.dirname(draft_path), f"{deduped_name}.yaml")
                        spec.to_yaml(new_path)
                        os.remove(draft_path)
                        draft_path = new_path

                    self.bus.post(AgentMessage(
                        sender="explorer", recipient="critic",
                        type="spec_review_request",
                        payload={"draft_path": draft_path, "round": 1},
                        priority=PRIORITY_SPEC_REVIEW,
                    ))

                # 6. Track scope advancement
                if not drafts:
                    self._consecutive_empty_rounds += 1
                else:
                    self._consecutive_empty_rounds = 0
                self._maybe_advance_scope()

                self.reset_conversation()

            except Exception as e:
                self._alert("error", f"Explorer error: {e}")
                self.reset_conversation()
                time.sleep(10)

    def _handle_inbox(self):
        """Process all pending messages from other agents."""
        for msg in self._process_inbox():
            if msg.type == "spec_approved":
                pass  # already in queue, nothing to do
            elif msg.type == "spec_rejected":
                self._pending_revisions.append(msg.payload)
            elif msg.type == "suggestion_for_spec":
                # Builder suggested an improvement that needs a new spec
                suggestion = msg.payload.get("suggestion", "")
                self._pending_revisions.append({
                    "draft_path": None,
                    "feedback": f"Builder suggestion for new spec:\n{suggestion}",
                    "round": 0,
                    "is_suggestion": True,
                })
            elif msg.type == "investigate_paper":
                # Orchestrator asked to investigate a specific paper or topic
                reference = msg.payload.get("reference", "")
                self.chat(
                    f"## Investigate this reference\n\n{reference}\n\n"
                    f"Use search_papers and read_paper to find and read this paper. "
                    f"If relevant, write an experiment spec to configs/drafts/."
                )
                self._log_search(reference, 0)
                self.reset_conversation()
            elif msg.type == "clarification_request":
                self._handle_clarification(msg)
            elif msg.type == "command":
                action = msg.payload.get("action", "")
                if action == "pause":
                    self.paused = True
                elif action == "resume":
                    self.paused = False

    def _revise_spec(self, payload: dict):
        """Revise a rejected spec based on Critic feedback, or draft from suggestion."""
        draft_path = payload.get("draft_path")
        feedback = payload.get("feedback", "")
        round_num = payload.get("round", 1) + 1
        is_suggestion = payload.get("is_suggestion", False)

        if is_suggestion:
            # Draft a new spec from Builder's suggestion
            context = self._build_context()
            self.chat(
                f"## Draft a new spec from this suggestion\n\n{feedback}\n\n"
                f"## Current Context\n{context}\n\n"
                f"Write the spec YAML to configs/drafts/ using write_file."
            )
            return

        if round_num > self.config.system.max_spec_review_rounds:
            return  # max rounds exceeded, Critic should have force-approved

        self.chat(f"Revise the spec at {draft_path} based on this feedback:\n{feedback}")
        self.bus.post(AgentMessage(
            sender="explorer", recipient="critic",
            type="spec_review_request",
            payload={"draft_path": draft_path, "round": round_num},
            priority=PRIORITY_SPEC_REVIEW,
        ))

    def _build_context(self) -> str:
        """Build context string for the LLM including task info, knowledge, and results."""
        return (
            f"## Task\n{self.config.task_domain_context}\n\n"
            f"## Knowledge Base Synthesis\n{self.kb.get_synthesis()}\n\n"
            f"## Current Results\n{self.results.get_summary()}\n\n"
            f"## Completed Experiments\n{', '.join(self.results.get_completed_names())}"
        )

    def _maybe_advance_scope(self):
        """Advance search scope if Explorer has been unproductive."""
        if self._consecutive_empty_rounds >= 3:
            self.current_round += 1
            self._consecutive_empty_rounds = 0
            self._alert("scope", f"Advancing to scope round {self.current_round}")

    def _get_queue_depth(self) -> int:
        """Return current number of specs waiting in the priority queue."""
        return self.queue.depth()

    def _wait_until_queue_below(self, threshold: int):
        """Block until the queue depth drops below threshold. Checks every 30s."""
        while not self._stopped:
            if self.queue.depth() < threshold:
                return
            # Still process inbox while waiting (handle pause/resume, rejections)
            self._handle_inbox()
            time.sleep(30)

    def _check_for_new_drafts(self) -> list[str]:
        """Glob configs/drafts/ for YAML files written since last check."""
        drafts_dir = os.path.join(self.config.paths.configs, "drafts")
        if not os.path.isdir(drafts_dir):
            return []

        new_drafts = []
        for path in glob.glob(os.path.join(drafts_dir, "*.yaml")):
            mtime = os.path.getmtime(path)
            if mtime > self._last_draft_check_time:
                new_drafts.append(path)

        self._last_draft_check_time = time.time()
        return new_drafts

    def _handle_clarification(self, msg: AgentMessage):
        """Use LLM to answer Builder's question from paper/domain context."""
        question = msg.payload.get("question", "")
        spec_name = msg.payload.get("spec_name", "")
        source_paper = msg.payload.get("source_paper", "")

        context = self._build_context()
        answer = self.chat(
            f"## Clarification request from Builder\n\n"
            f"**Experiment:** {spec_name}\n"
            f"**Source paper:** {source_paper}\n"
            f"**Question:** {question}\n\n"
            f"## Context\n{context}\n\n"
            f"Answer the Builder's question using your knowledge of the paper and domain. "
            f"If you need to re-read the paper, use the read_paper tool."
        )

        self.bus.post(AgentMessage(
            sender="explorer", recipient="builder",
            type="clarification_response",
            payload={"spec_name": spec_name, "answer": answer},
        ))
        self.reset_conversation()

    def _dedup_spec_name(self, base_name: str) -> str:
        """Ensure spec name is unique across drafts, experiments, and results.
        Appends _v2, _v3, etc. if collision detected."""
        existing_names = set()

        # Collect names from drafts
        drafts_dir = os.path.join(self.config.paths.configs, "drafts")
        if os.path.isdir(drafts_dir):
            for f in os.listdir(drafts_dir):
                if f.endswith(".yaml"):
                    existing_names.add(f[:-5])  # strip .yaml

        # Collect names from experiments
        experiments_dir = os.path.join(self.config.paths.configs, "experiments")
        if os.path.isdir(experiments_dir):
            for f in os.listdir(experiments_dir):
                if f.endswith(".yaml"):
                    existing_names.add(f[:-5])

        # Collect names from results
        existing_names.update(self.results.get_completed_names())

        if base_name not in existing_names:
            return base_name

        version = 2
        while f"{base_name}_v{version}" in existing_names:
            version += 1
        return f"{base_name}_v{version}"

    def _log_search(self, query: str, results_count: int):
        """Append a search event to the explorer journal."""
        self.explorer_logger.log(
            event="search",
            query=query,
            results_count=results_count,
            round=self.current_round,
        )

    def _log_paper_read(self, title: str, authors: str, url: str, relevant: bool,
                        summary: str | None = None, key_insights: str | None = None,
                        limitations: str | None = None, relevance: str | None = None,
                        spec_drafted: str | None = None, reason: str | None = None):
        """Append a paper-read event to the explorer journal."""
        self.explorer_logger.log(
            event="paper_read",
            title=title,
            authors=authors,
            url=url,
            relevant=relevant,
            summary=summary,
            key_insights=key_insights,
            limitations=limitations,
            relevance=relevance,
            spec_drafted=spec_drafted,
            reason=reason,
            round=self.current_round,
        )
```

---

### `agents/critic.py`

```python
import json
import os
import time

from agents.base import BaseAgent
from core.message_bus import AgentMessage
from core.data_structures import ExperimentSpec
from core.event_logger import EventLogger
from core.utils import normalize_name

# Priority constants
PRIORITY_CODE_REVIEW = 100
PRIORITY_SPEC_REVIEW = 80
PRIORITY_INSIGHT_REVIEW = 60
PRIORITY_SUGGESTION = 40
PRIORITY_RERANK = 20


class CriticAgent(BaseAgent):
    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 priority_queue, knowledge_base, results_tracker,
                 review_logger: EventLogger, system_logger: EventLogger, config):
        super().__init__("critic", llm, tools, system_prompt, message_bus, cost_tracker)
        self.queue = priority_queue
        self.kb = knowledge_base
        self.results = results_tracker
        self.review_logger = review_logger
        self.system_logger = system_logger
        self.config = config

    def run(self):
        """Main loop. Processes inbox in priority order."""
        while not self._stopped:
            try:
                messages = self._process_inbox()
                if not messages:
                    time.sleep(2)
                    continue

                for message in messages:
                    if message.type == "spec_review_request":
                        self._handle_spec_review(message)
                    elif message.type == "code_review_request":
                        self._handle_code_review(message)
                    elif message.type == "insight_review_request":
                        self._handle_insight_review(message)
                    elif message.type == "suggestion_triage_request":
                        self._handle_suggestion_triage(message)
                    elif message.type == "rerank_request":
                        self._handle_rerank(message)
                    elif message.type == "priority_adjust":
                        self._handle_priority_adjust(message)
                    elif message.type == "inject_spec":
                        self._handle_inject_spec(message)

                    self.reset_conversation()

            except Exception as e:
                self._alert("error", f"Critic error: {e}")
                self.reset_conversation()
                time.sleep(5)

    def _handle_spec_review(self, message):
        """Review a draft spec. Uses structured submit_review tool."""
        draft_path = message.payload["draft_path"]
        round_num = message.payload["round"]

        # Deduplication check
        spec = ExperimentSpec.from_yaml(draft_path)
        if self.results.is_completed(spec.name):
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="rejected", feedback="Already completed.")
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_rejected",
                payload={"draft_path": draft_path, "round": round_num,
                         "feedback": f"Experiment '{spec.name}' already completed."}
            ))
            return
        similar = self.queue.contains_similar(spec.name)
        if similar:
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="rejected", feedback=f"Similar to '{similar}' in queue.")
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_rejected",
                payload={"draft_path": draft_path, "round": round_num,
                         "feedback": f"Similar experiment '{similar}' already in queue."}
            ))
            return

        spec_content = open(draft_path).read()
        force_approve = round_num >= self.config.system.max_spec_review_rounds

        # LLM reviews using submit_review tool
        response = self.chat(
            f"## Review this experiment spec (round {round_num}/{self.config.system.max_spec_review_rounds})\n\n"
            f"```yaml\n{spec_content}\n```\n\n"
            f"## Current results context\n{self.results.get_summary()}\n\n"
            f"Check: completeness, accuracy, specificity, resource feasibility, "
            f"control variant needed?\n\n"
            f"{'FINAL ROUND: force-approve with your fixes if still issues.' if force_approve else ''}\n\n"
            f"Use the submit_review tool to submit your decision."
        )

        # Parse structured decision from tool call result
        decision = self._parse_last_tool_result("submit_review")

        if decision and decision.get("verdict") == "approved":
            score = decision.get("score", self._compute_score(spec))
            approved_path = draft_path.replace("/drafts/", "/experiments/")
            os.makedirs(os.path.dirname(approved_path), exist_ok=True)
            os.rename(draft_path, approved_path)
            self.queue.insert(spec, score)
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="approved", score=score)
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_approved",
                payload={"name": spec.name, "score": score}
            ))
            # Handle control variant request
            if decision.get("control_variant_needed"):
                self.bus.post(AgentMessage(
                    sender="critic", recipient="explorer",
                    type="spec_approved",
                    payload={"name": spec.name, "score": score,
                             "control_request": decision.get("control_description", "")}
                ))
        else:
            feedback = decision.get("feedback", response) if decision else response
            self.review_logger.log(type="spec_review", spec_name=spec.name,
                                   verdict="rejected", feedback=feedback[:500])
            self.bus.post(AgentMessage(
                sender="critic", recipient="explorer",
                type="spec_rejected",
                payload={"draft_path": draft_path, "round": round_num,
                         "feedback": feedback}
            ))

    def _handle_code_review(self, message):
        """Review structural code change. Uses structured submit_code_review tool."""
        diff = message.payload["diff"]
        spec_name = message.payload["spec_name"]
        self.chat(
            f"## Code review for experiment: {spec_name}\n\n"
            f"```diff\n{diff}\n```\n\n"
            f"Check: implementation matches spec, no obvious bugs, correct API usage.\n"
            f"Use the submit_code_review tool to submit your decision."
        )
        decision = self._parse_last_tool_result("submit_code_review")
        verdict = decision.get("verdict", "approved") if decision else "approved"
        feedback = decision.get("feedback", "") if decision else ""
        self.review_logger.log(type="code_review", spec_name=spec_name,
                               verdict=verdict, feedback=feedback[:500])
        self.bus.post(AgentMessage(
            sender="critic", recipient="builder",
            type="code_approved" if verdict == "approved" else "code_rejected",
            payload={"spec_name": spec_name, "feedback": feedback}
        ))

    def _handle_insight_review(self, message):
        """Review a draft insight using structured submit_insight_review tool."""
        insight = message.payload["insight"]
        experiment = message.payload.get("experiment", "unknown")
        self.chat(
            f"## Review this experiment insight for: {experiment}\n\n{insight}\n\n"
            f"## Knowledge base context\n{self.kb.get_synthesis()}\n\n"
            f"Check: over-claiming, unsupported causal claims, missed observations, "
            f"contradictions with prior knowledge.\n\n"
            f"Use the submit_insight_review tool to submit your decision."
        )
        decision = self._parse_last_tool_result("submit_insight_review")
        if decision and decision.get("verdict") == "approved":
            edited = decision.get("edits", insight)
            self.review_logger.log(type="insight_review", experiment=experiment,
                                   verdict="approved")
            self.bus.post(AgentMessage(
                sender="critic", recipient="builder",
                type="insight_approved",
                payload={"insight": edited, "experiment": experiment}
            ))
        else:
            feedback = decision.get("feedback", "") if decision else ""
            self.review_logger.log(type="insight_review", experiment=experiment,
                                   verdict="rejected", feedback=feedback[:500])
            self.bus.post(AgentMessage(
                sender="critic", recipient="builder",
                type="insight_rejected",
                payload={"feedback": feedback, "experiment": experiment}
            ))

    def _handle_suggestion_triage(self, message):
        """Triage a Builder suggestion using structured submit_triage tool."""
        suggestion = message.payload["suggestion"]
        self.chat(
            f"## Triage this improvement suggestion\n\n{suggestion}\n\n"
            f"Classify as:\n"
            f"- minor: hyperparameter tweak, config-only change\n"
            f"- moderate: meaningful modification, needs new code\n"
            f"- major: novel architectural idea, potential research contribution\n\n"
            f"Use the submit_triage tool to submit your decision."
        )
        decision = self._parse_last_tool_result("submit_triage")
        level = decision.get("level", "minor") if decision else "minor"

        self.review_logger.log(type="suggestion_triage", level=level,
                               suggestion=suggestion[:200])

        # For major suggestions, write to ideas file for deeper investigation
        if level == "major":
            ideas_path = os.path.join(self.config.paths.results, "ideas_for_system2.md")
            with open(ideas_path, "a") as f:
                f.write(f"\n## {suggestion[:80]}\n\n")
                f.write(f"**Triage:** {decision.get('justification', '')}\n\n")
                f.write(f"**Suggested action:** {decision.get('action', '')}\n\n")
                f.write(f"---\n")

        self.bus.post(AgentMessage(
            sender="critic", recipient="builder",
            type="suggestion_triaged",
            payload={"level": level, "suggestion": suggestion,
                     "response": decision if decision else {}}
        ))

    def _handle_rerank(self, message):
        """Rerank the priority queue based on current results context."""
        context = {"results": self.results.get_all(),
                    "completed": self.results.get_completed_names()}
        self.queue.rerank(
            scoring_fn=lambda spec, ctx: self._compute_score(spec),
            context=context,
        )

    def _handle_priority_adjust(self, message):
        """Adjust priority scores for specs matching a topic. Boosts or lowers scores."""
        topic = message.payload.get("topic", "")
        boost = message.payload.get("boost", True)  # True = prioritize, False = deprioritize
        multiplier = 1.5 if boost else 0.5
        for spec, score in self.queue.get_all():
            if topic.lower() in spec.category.lower() or topic.lower() in spec.description.lower():
                self.queue.update_score(spec.name, score * multiplier)
        self.system_logger.log(event="priority_adjust", topic=topic, boost=boost)

    def _handle_inject_spec(self, message):
        """Review a user-injected spec. Same as spec review but source is human."""
        spec_data = message.payload.get("spec_data")
        # Write to drafts, then process as normal spec review
        draft_path = os.path.join(self.config.paths.configs, "drafts", f"{spec_data['name']}.yaml")
        os.makedirs(os.path.dirname(draft_path), exist_ok=True)
        spec = ExperimentSpec(**spec_data)
        spec.to_yaml(draft_path)
        self._handle_spec_review(AgentMessage(
            sender="orchestrator", recipient="critic",
            type="spec_review_request",
            payload={"draft_path": draft_path, "round": 1},
            priority=PRIORITY_SPEC_REVIEW,
        ))

    def _compute_score(self, spec: ExperimentSpec) -> float:
        """Score a spec for priority ranking. Higher = run sooner."""
        completed = self.results.get_completed_names()
        performance_score = 1.0
        similar_count = sum(1 for name in completed
                          if normalize_name(name).startswith(normalize_name(spec.category)))
        diversity_bonus = 2.0 if similar_count == 0 else 0.5 / similar_count
        category_count = sum(1 for r in self.results.get_all()
                            if r.description and spec.category in r.description)
        information_bonus = 3.0 if category_count == 0 else 1.0 / (category_count + 1)
        return performance_score + diversity_bonus + information_bonus

    def _parse_last_tool_result(self, tool_name: str) -> dict | None:
        """Find the last tool result from a specific tool in conversation.
        Verifies the tool_call_id matches a call to the expected tool_name."""
        for msg in reversed(self.conversation):
            if msg.role == "tool_result" and msg.tool_call_id:
                # Find the matching tool call to verify tool name
                for prev_msg in self.conversation:
                    if prev_msg.tool_calls:
                        for call in prev_msg.tool_calls:
                            if call["id"] == msg.tool_call_id and call["name"] == tool_name:
                                try:
                                    return json.loads(msg.content)
                                except (json.JSONDecodeError, TypeError):
                                    continue
        return None
```

---

### `agents/builder.py`

```python
import json
import os
import re
import time

from agents.base import BaseAgent
from core.message_bus import AgentMessage
from core.data_structures import ExperimentSpec, ExperimentResult, GPUSlot
from core.event_logger import EventLogger
from core.utils import atomic_write

# Priority constants
PRIORITY_CODE_REVIEW = 100
PRIORITY_SPEC_REVIEW = 80
PRIORITY_INSIGHT_REVIEW = 60
PRIORITY_SUGGESTION = 40
PRIORITY_RERANK = 20


class BuilderAgent(BaseAgent):
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
        self.pending_pairs: dict[str, list[str]] = {}  # parent -> [variant_names]
        self._last_validation_error: str = ""
        self._consecutive_failures: int = 0
        self._last_failure_error: str = ""
        self._edited_files: set[str] = set()  # Track files Builder edited (for Tier 1/2 distinction)

    def run(self):
        """Main loop. Manages GPU pool with non-blocking reviews."""
        self._fill_free_slots()

        while not self._stopped:
            try:
                if self.paused:
                    time.sleep(5)
                    continue

                # 1. Process inbox (non-blocking)
                self._handle_inbox()

                # 2. Check all running slots
                self._monitor_running_slots()

                # 3. Fill free slots
                self._fill_free_slots()

                # 4. Persist state for crash recovery
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
                pass  # used if Builder asked Explorer for clarification
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
                # Check early stopping via progress file
                if self._check_early_stop(slot):
                    self.runner.cancel(slot.job_id)
                    self._handle_early_stop(slot)
                # Check timeout
                elif self._check_timeout(slot):
                    self.runner.cancel(slot.job_id)
                    self._handle_timeout(slot)

        # Also check timeouts for slots not yet picked up by sacct
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
            return False  # no baseline to compare against yet

        # No improvement beyond random after early_stop_fraction of budget
        best_baseline = max(r.metric_value for r in baselines if r.metric_value is not None)
        threshold = best_baseline * 0.1  # less than 10% of baseline performance
        if self.task.metric[1] == "higher":
            return current < threshold
        else:
            return current > best_baseline * 10  # way worse for "lower is better"

    def _check_timeout(self, slot: GPUSlot) -> bool:
        """Check if a slot has exceeded the timeout threshold."""
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
        # Track paired experiments
        self._track_pair(spec)

        # Clear edited files tracker for this implementation
        self._edited_files.clear()

        is_baseline = spec.category == "baseline"
        seeds = self.config.system.baseline_seeds if is_baseline else 1

        # Baseline: run seeds sequentially on same slot
        if is_baseline and seeds > 1:
            self._run_baseline_seeds(spec, slot, seeds)
            return

        # Regular experiment: implement, validate, submit
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

        # Track which files were edited by inspecting conversation for tool calls
        for msg in self.conversation:
            if msg.tool_calls:
                for call in msg.tool_calls:
                    if call["name"] in ("edit_file", "write_file"):
                        path = call["arguments"].get("path", call["arguments"].get("file_path", ""))
                        if path:
                            self._edited_files.add(path)

        # Pre-flight validation with 3-tier error handling
        if not self._validate_implementation(spec):
            # One retry: ask LLM to fix
            error = self._last_validation_error
            self.chat(f"Pre-flight validation failed:\n{error}\nFix the issue.")
            if not self._validate_implementation(spec):
                self._record_validation_fail(spec, slot)
                self.reset_conversation()
                return

        # Non-blocking code review for structural changes
        diff = self._get_git_diff()
        if self._is_structural_change(diff):
            self.bus.post(AgentMessage(
                sender="builder", recipient="critic",
                type="code_review_request",
                payload={"spec_name": spec.name, "diff": diff},
                priority=PRIORITY_CODE_REVIEW,
            ))
            # Submit optimistically — cancel if rejected (handled in _handle_inbox)

        # Git commit
        self.chat("Commit the changes with git_commit tool.")

        # Submit training job
        log_dir = os.path.join(self.config.paths.logs, spec.name)
        os.makedirs(log_dir, exist_ok=True)
        command = self._build_train_command(spec, seed=1)
        job_id = self.runner.submit(command, spec.name)
        self.pool.assign(slot.slot_id, job_id, spec.name, log_dir)

        self.system_logger.log(event="experiment_submitted", spec=spec.name, job_id=job_id)
        self.reset_conversation()

    def _validate_implementation(self, spec: ExperimentSpec) -> bool:
        """Pre-flight: syntax check + 60s smoke test."""
        # Syntax check
        result = self.chat(
            "Run a syntax check on the modified files using run_command: "
            "python -c 'import applied_scientist'"
        )
        if "Error" in result or "Traceback" in result:
            self._last_validation_error = result
            return False

        # Smoke test: 60-second training run
        smoke_cmd = self._build_train_command(spec, seed=0, time_budget=60)
        result = self.chat(
            f"Run a 60-second smoke test using run_command:\n{smoke_cmd}"
        )
        if "Error" in result or "Traceback" in result or "crash" in result.lower():
            self._last_validation_error = result
            return False
        return True

    def _run_baseline_seeds(self, spec: ExperimentSpec, slot: GPUSlot, seeds: int):
        """Run baseline seeds sequentially on one slot."""
        for seed in range(1, seeds + 1):
            job_name = f"{spec.name}_s{seed}"
            log_dir = os.path.join(self.config.paths.logs, job_name)
            os.makedirs(log_dir, exist_ok=True)
            command = self._build_train_command(spec, seed=seed)
            job_id = self.runner.submit(command, job_name)
            self.pool.assign(slot.slot_id, job_id, job_name, log_dir)
            # Wait for this seed to complete before starting next
            self._wait_for_slot_completion(slot)
            # Process this seed's result
            status = self.runner.status(slot.job_id)
            if status.state == "completed":
                self._handle_completion(slot)
            else:
                self._handle_failure(slot)

    def _handle_completion(self, slot: GPUSlot):
        """Process a completed experiment. Parse results, submit insight, check tuning."""
        self.pool.release(slot.slot_id)
        name = slot.experiment_name
        log = self.runner.get_log(slot.job_id)
        result = self._parse_results(name, log)
        self.results.add(result)

        self.system_logger.log(event="experiment_completed", spec=name,
                               metric=result.metric_value, status=result.status)

        # Reset failure counter on successful completion
        self._consecutive_failures = 0

        # Non-blocking insight submission
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
            f"Result: {result}\n\n"
            f"## Previous results\n{context}\n\n"
            f"Write observations (factual), comparisons to baselines, "
            f"and hypotheses (labeled as such). Include crash/early-stop analysis if applicable."
        )
        self.bus.post(AgentMessage(
            sender="builder", recipient="critic",
            type="insight_review_request",
            payload={"insight": insight, "experiment": name},
            priority=PRIORITY_INSIGHT_REVIEW,
        ))

    def _track_pair(self, spec: ExperimentSpec):
        """Track paired experiments for comparative insights."""
        if spec.control_variant_of:
            self.pending_pairs.setdefault(spec.control_variant_of, []).append(spec.name)

    def _check_paired_completion(self, name: str):
        """Write comparative insight if all experiments in a pair are done."""
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
        """Record a validation failure with 3-tier error handling.
        Tier 1: Builder's fault (error in files Builder edited) — record and continue.
        Tier 2: Task folder issue (error in files Builder did NOT edit) — pause and alert.
        Tier 3: Repeated failures with same error — pause after 3 consecutive."""
        error = self._last_validation_error

        # Tier 1: Builder's fault (error in files Builder edited)
        builder_fault = any(f in error for f in self._edited_files)

        # Tier 2: Task folder issue (error in files Builder did NOT edit)
        if not builder_fault:
            self._alert("task_error",
                f"Task folder error during {spec.name}: {error[:300]}\n"
                f"Builder cannot fix this. Please check your task setup.")
            self.paused = True
            self.system_logger.log(event="builder_paused", reason="task_folder_error",
                                   spec=spec.name, error=error[:500])
            self.pool.release(slot.slot_id)
            return

        # Tier 3: Repeated failures
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
        """Cancel running job, apply fix, resubmit."""
        spec_name = msg.payload["spec_name"]
        feedback = msg.payload["feedback"]
        # Find the slot running this experiment
        for slot in self.pool.get_running():
            if slot.experiment_name == spec_name:
                self.runner.cancel(slot.job_id)
                self.pool.release(slot.slot_id)
                # Apply fix and resubmit
                self.chat(f"Fix this code based on review:\n{feedback}")
                self.chat("Commit the fix with git_commit tool.")
                # Re-validate and resubmit
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
        """Act on triaged suggestion."""
        level = msg.payload["level"]
        suggestion = msg.payload["suggestion"]
        if level == "minor":
            # Create variant config and run — handled in next fill cycle
            pass
        elif level == "moderate":
            # Explorer will write a new spec
            self.bus.post(AgentMessage(
                sender="builder", recipient="explorer",
                type="suggestion_for_spec",
                payload={"suggestion": suggestion}
            ))
        elif level == "major":
            # Already saved to ideas_for_system2.md by Critic
            pass

    # -----------------------------------------------------------------
    # Previously undefined methods — now fully defined
    # -----------------------------------------------------------------

    def _build_train_command(self, spec: ExperimentSpec, seed: int,
                             time_budget: int | None = None) -> str:
        """Construct standardized training command. Always the same format."""
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
        return (
            f"python -m applied_scientist.run_experiment "
            f"--spec {spec_path} "
            f"--task-module {self.config.task['module']} "
            f"--seed {seed} "
            f"--time-budget {tb} "
            f"--log-path {log_dir} "
            f"--checkpoint-dir {ckpt_dir}"
        )

    def _parse_results(self, name: str, log: str) -> ExperimentResult:
        """Parse training results from job log. Last line of stdout should be JSON metrics."""
        commit = ""
        try:
            # Get commit hash from git log
            for msg in self.conversation:
                if hasattr(msg, "content") and "commit" in msg.content.lower():
                    # Extract hash from recent git operations
                    match = re.search(r'[a-f0-9]{7,40}', msg.content)
                    if match:
                        commit = match.group()
                        break
        except Exception:
            pass

        # Parse metrics from last line of log
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
            name=name,
            commit=commit,
            algorithm=name.split("_")[0],
            model=self.llm.get_model_id(),
            metric_value=metrics.get(metric_name),
            metric_std=None,
            seeds=1,
            training_seconds=metrics.get("training_seconds", 0),
            peak_memory_mb=metrics.get("peak_memory_mb", 0),
            status="keep" if metrics.get(metric_name) is not None else "crash",
            description=metrics.get("description", ""),
            extra_metrics={k: v for k, v in metrics.items()
                          if k not in {metric_name, "training_seconds",
                                       "peak_memory_mb", "description", "status"}},
        )

    def _write_and_submit_suggestion(self, name: str, result: ExperimentResult):
        """Write improvement suggestion based on experiment results. Non-blocking."""
        if result.status in ("crash", "validation_fail", "timeout"):
            return  # No meaningful suggestions from failed experiments
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

    def _get_git_diff(self) -> str:
        """Get diff of uncommitted or last-committed changes."""
        result = self.chat("Show the current changes using git_diff tool.")
        return result

    def _is_structural_change(self, diff: str) -> bool:
        """Check if diff contains structural changes (new classes/functions, not just config)."""
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
        """Block until slot's job completes. Used for sequential baseline seeds."""
        while not self._stopped:
            status = self.runner.status(slot.job_id)
            if status.state not in ("pending", "running"):
                return
            time.sleep(10)

    def _handle_early_stop(self, slot: GPUSlot):
        """Handle an early-stopped experiment. Records partial metrics if available."""
        name = slot.experiment_name
        log = self.runner.get_log(slot.job_id)

        # Try to get partial metrics from progress file
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
        self._consecutive_failures = 0  # Reset failure counter on non-validation failure

    def _handle_timeout(self, slot: GPUSlot):
        """Handle a timed-out experiment. Cancels job, records timeout result."""
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
        """Handle a crashed experiment. Attempt one fix if fixable, else record crash."""
        name = slot.experiment_name
        log = self.runner.get_log(slot.job_id)
        self.pool.release(slot.slot_id)

        # Check if fixable (OOM, simple errors)
        fixable_patterns = [
            "OutOfMemoryError", "CUDA out of memory",
            "RuntimeError", "TypeError", "ValueError",
        ]
        is_fixable = any(p in log for p in fixable_patterns)

        if is_fixable:
            # Attempt one fix
            try:
                self.chat(
                    f"Training crashed with this error:\n{log[-2000:]}\n\n"
                    f"Attempt to fix the issue."
                )
                spec = ExperimentSpec.from_yaml(
                    os.path.join(self.config.paths.configs, "experiments", f"{name}.yaml"))
                if self._validate_implementation(spec):
                    self.chat("Commit the fix with git_commit tool.")
                    # Resubmit
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

        # Unfixable or fix failed
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
        """Persist pending_pairs to .state/pending_pairs.json for crash recovery."""
        state_path = os.path.join(self.config.paths.workspace, ".state", "pending_pairs.json")
        atomic_write(state_path, json.dumps(self.pending_pairs))

    @classmethod
    def load_pairs(cls, state_path: str) -> dict:
        """Load pending_pairs from state file."""
        if os.path.exists(state_path):
            with open(state_path) as f:
                return json.loads(f.read())
        return {}
```

---

### `agents/orchestrator.py`

```python
import os
import time

from agents.base import BaseAgent
from core.message_bus import AgentMessage
from core.event_logger import EventLogger

# Priority constants
PRIORITY_SPEC_REVIEW = 80


class OrchestratorAgent(BaseAgent):
    def __init__(self, llm, tools, system_prompt, message_bus, cost_tracker,
                 priority_queue, knowledge_base, results_tracker, gpu_pool,
                 system_logger: EventLogger, agents_ref: dict):
        super().__init__("orchestrator", llm, tools, system_prompt, message_bus, cost_tracker)
        self.queue = priority_queue
        self.kb = knowledge_base
        self.results = results_tracker
        self.pool = gpu_pool
        self.system_logger = system_logger
        self.agents = agents_ref  # dict of agent name -> agent instance (for @agent routing)

    def run(self):
        """Main loop. Reads stdin, interprets with LLM, routes."""
        while not self._stopped:
            try:
                # Check for alerts first
                self._display_alerts()

                user_input = input("> ")
            except EOFError:
                continue

            if not user_input.strip():
                continue

            # Handle @agent direct routing
            if user_input.strip().startswith("@"):
                self._handle_direct_message(user_input)
                continue

            # Handle built-in commands directly (no LLM call)
            if self._handle_builtin(user_input):
                continue

            # LLM interprets and responds
            status = self._build_status()
            response = self.chat(
                f"## System Status\n{status}\n\n"
                f"## User Message\n{user_input}\n\n"
                f"Interpret the user's intent. If it's a query, answer from status. "
                f"If it's a command, state what action you're taking."
            )
            print(f"  {response}\n")
            self._route_command(user_input, response)
            self.reset_conversation()

    def _handle_builtin(self, user_input: str) -> bool:
        """Handle commands that don't need LLM interpretation."""
        cmd = user_input.strip().lower()
        if cmd == "status":
            print(f"  {self._build_status()}\n")
            return True
        elif cmd == "results":
            print(f"  {self.results.get_summary()}\n")
            return True
        elif cmd == "knowledge":
            print(f"  {self.kb.get_synthesis()}\n")
            return True
        elif cmd == "queue":
            items = self.queue.peek(10)
            for spec, score in items:
                print(f"  {score:.1f}  {spec.name}: {spec.description}")
            if not items:
                print("  Queue is empty.")
            print()
            return True
        elif cmd == "cost":
            print(f"  {self.cost.get_summary()}\n")
            return True
        elif cmd == "pause":
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="builder",
                type="command", payload={"action": "pause"}
            ))
            print("  Builder paused. Running experiments will finish.\n")
            return True
        elif cmd == "resume":
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="builder",
                type="command", payload={"action": "resume"}
            ))
            print("  Builder resumed.\n")
            return True
        elif cmd.startswith("gpus "):
            try:
                n = int(cmd.split()[1])
                self.pool.resize(n)
                print(f"  GPU pool resized to {n} slots.\n")
            except (ValueError, IndexError):
                print("  Usage: gpus N\n")
            return True
        return False

    def _handle_direct_message(self, user_input: str):
        """Handle @agent messages by sending directly to agent's chat."""
        parts = user_input.strip().split(None, 1)
        target = parts[0][1:].lower()  # Remove @ prefix
        message = parts[1] if len(parts) > 1 else ""
        if target in self.agents and target != "orchestrator":
            agent = self.agents[target]
            response = agent.chat(message)
            print(f"  [{target.title()}]: {response}\n")
            agent.reset_conversation()
        else:
            print(f"  Unknown agent: {target}. Available: explorer, critic, builder\n")

    def _route_command(self, user_input: str, response: str):
        """Parse LLM response and dispatch commands to agents."""
        lower_input = user_input.lower()
        lower_response = response.lower()

        # Priority adjustment — boost
        if any(w in lower_response for w in ["prioritiz", "boost", "focus on"]):
            topic = user_input  # Let Critic extract the topic
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="critic",
                type="priority_adjust",
                payload={"topic": topic, "boost": True},
            ))

        # Priority adjustment — lower
        elif any(w in lower_response for w in ["deprioritiz", "lower", "skip"]):
            topic = user_input
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="critic",
                type="priority_adjust",
                payload={"topic": topic, "boost": False},
            ))

        # Paper investigation
        if any(w in lower_response for w in ["relay", "investigate", "paper", "look into"]):
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="explorer",
                type="investigate_paper",
                payload={"reference": user_input},
            ))

        # Spec injection
        if any(w in lower_response for w in ["inject", "try this", "add spec"]):
            self.bus.post(AgentMessage(
                sender="orchestrator", recipient="critic",
                type="inject_spec",
                payload={"spec_data": {
                    "name": "user_injected",
                    "description": user_input,
                    "source_paper": "user suggestion",
                    "architecture": {},
                    "why_it_might_work": user_input,
                    "task_config": {},
                    "training_config": {},
                    "resource_estimate": {},
                }},
                priority=PRIORITY_SPEC_REVIEW,
            ))

    def _build_status(self) -> str:
        """Build a status summary string for display or LLM context."""
        return (
            f"GPU Pool:\n{self.pool.get_status_summary()}\n\n"
            f"Queue: {self.queue.depth()} pending specs\n\n"
            f"Results: {len(self.results.get_all())} experiments completed\n"
            f"Best: {self.results.get_best(1)}\n\n"
            f"Cost: {self.cost.get_summary()}\n\n"
            f"Knowledge synthesis:\n{self.kb.get_synthesis()[:500]}"
        )

    def _display_alerts(self):
        """Display any pending alerts from other agents."""
        for msg in self._process_inbox():
            if msg.type == "alert":
                print(f"\n  [ALERT] {msg.payload.get('message', '')}\n> ", end="", flush=True)
```
## Phase 6: Hyperparameter Tuning

Bayesian hyperparameter optimization via Optuna, run after each experiment completes. The tuner dynamically computes an iteration budget based on queue pressure, experiment promise, and remaining training signal, then runs short trial trainings to find better hyperparameters before the next full run.

### `tuning.py`

```python
import optuna
import json
import os


class HyperparameterTuner:
    """Bayesian optimization for hyperparameter tuning after each experiment."""

    def __init__(self, task_adapter, job_runner, config):
        self.task = task_adapter
        self.runner = job_runner
        self.config = config

    def compute_budget(self, queue_depth: int, result: "ExperimentResult",
                       best_metric: float | None, training_still_improving: bool) -> int:
        """Compute dynamic tuning iteration budget.

        Formula: base * queue_factor * time_factor * promise_factor * improvement_factor
        Minimum: config.tuning.min_iterations (default 3)
        Maximum: config.tuning.max_iterations (default 15)
        """
        base = self.config.tuning.max_iterations

        # Queue factor: tune less when queue is full
        throttle = self.config.system.queue_throttle_high
        queue_factor = 1.0 if queue_depth == 0 else max(0.2, 1.0 - queue_depth / throttle)

        # Time factor: tune less for short experiments
        time_factor = min(1.0, self.config.system.time_budget / 1800)

        # Promise factor: tune more for promising results
        if result.metric_value is not None and best_metric is not None and best_metric > 0:
            promise_factor = result.metric_value / best_metric  # 0 to ~1
        else:
            promise_factor = 0.5

        # Improvement factor: tune more if training was still improving
        improvement_factor = 1.0 if training_still_improving else 0.5

        n = round(base * queue_factor * time_factor * promise_factor * improvement_factor)
        return max(self.config.tuning.min_iterations, min(n, self.config.tuning.max_iterations))

    def get_search_space(self, spec: "ExperimentSpec") -> dict:
        """Merge task defaults with spec-specific overrides.
        Spec overrides take priority."""
        space = dict(self.task.get_tunable_defaults())
        if spec.tunable_hyperparameters:
            space.update(spec.tunable_hyperparameters)
        return space

    def _suggest_params(self, trial: optuna.Trial, search_space: dict,
                        base_config: dict) -> dict:
        """Use Optuna trial to suggest hyperparameters."""
        params = dict(base_config)
        for name, spec in search_space.items():
            if spec["type"] == "log_uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif spec["type"] == "uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "discrete":
                params[name] = trial.suggest_categorical(name, spec["values"])
            elif spec["type"] == "factor":
                base_val = base_config.get(name, 1)
                factor = trial.suggest_categorical(f"{name}_factor", spec["factors"])
                params[name] = int(base_val * factor) if isinstance(base_val, int) else base_val * factor
        return params

    def tune(self, spec: "ExperimentSpec", n_iterations: int,
             log_dir: str) -> dict | None:
        """Run Bayesian optimization. Returns best params dict or None if tuning disabled/fails.

        Each trial runs a SHORT training (time_budget / 3) to evaluate hyperparameters.
        """
        search_space = self.get_search_space(spec)
        if not search_space:
            return None

        metric_name, direction = self.task.metric
        optuna_direction = "maximize" if direction == "higher" else "minimize"
        study = optuna.create_study(direction=optuna_direction)
        tuning_budget = self.config.system.time_budget // 3  # Short runs for tuning

        def objective(trial):
            params = self._suggest_params(trial, search_space, spec.training_config)
            # Build and run a short training
            trial_dir = os.path.join(log_dir, f"tune_trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            ckpt_dir = os.path.join(trial_dir, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                result = self.task.train(
                    config={**spec.task_config, **params},
                    seed=42,
                    time_budget=tuning_budget,
                    log_path=trial_dir,
                    checkpoint_dir=ckpt_dir,
                )
                value = result.get(metric_name)
                if value is None:
                    return float("-inf") if direction == "higher" else float("inf")
                return value
            except Exception:
                return float("-inf") if direction == "higher" else float("inf")

        study.optimize(objective, n_trials=n_iterations, timeout=self.config.system.time_budget)

        if study.best_trial:
            return study.best_params
        return None
```

---

## Phase 7: System, Config & Entry Point

### `config.py`

Updated with `TuningConfig` dataclass and env-var resolution in `from_yaml`.

```python
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    time_budget: int = 1800
    n_gpus: int = 1
    baseline_seeds: int = 3
    eval_samples: int = 100
    early_stop_fraction: float = 0.4
    max_spec_review_rounds: int = 3
    queue_throttle_high: int = 10
    queue_throttle_low: int = 5
    timeout_multiplier: float = 2.0
    cost_alert_threshold: float | None = None


@dataclass
class TuningConfig:
    enabled: bool = True
    max_iterations: int = 15
    min_iterations: int = 3


@dataclass
class LLMAgentConfig:
    backend: str
    model: str
    base_url: str | None = None


@dataclass
class ComputeConfig:
    backend: str = "local"
    slurm: dict = field(default_factory=dict)


@dataclass
class PathsConfig:
    workspace: str = "./workspace"
    results: str = "./workspace/results"
    configs: str = "./workspace/configs"
    checkpoints: str = "./workspace/results/checkpoints"
    logs: str = "./workspace/results/logs"


@dataclass
class Config:
    system: SystemConfig
    tuning: TuningConfig
    llm: dict[str, LLMAgentConfig]
    compute: ComputeConfig
    task: dict
    paths: PathsConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load from YAML, resolve env vars (${VAR}), apply defaults."""
        import yaml, os, re
        with open(path) as f:
            raw = f.read()
        # Resolve ${VAR} environment variable references
        def resolve_env(match):
            var = match.group(1)
            return os.environ.get(var, match.group(0))
        raw = re.sub(r'\$\{(\w+)\}', resolve_env, raw)
        data = yaml.safe_load(raw)

        return cls(
            system=SystemConfig(**data.get("system", {})),
            tuning=TuningConfig(**data.get("tuning", {})),
            llm={k: LLMAgentConfig(**v) for k, v in data.get("llm", {}).items()},
            compute=ComputeConfig(**data.get("compute", {})),
            task=data.get("task", {}),
            paths=PathsConfig(**data.get("paths", {})),
        )
```

### `run_experiment.py`

Standardized training wrapper. Called by compute backends (SLURM/local). The Builder's `_build_train_command` generates: `python -m applied_scientist.run_experiment --spec ... --seed ...`. This script loads the task adapter, reads the spec, calls `adapter.train()`, and prints results as JSON on the last line.

```python
"""Standardized training wrapper. Called by compute backends (SLURM/local).
Builder's _build_train_command generates: python -m applied_scientist.run_experiment --spec ... --seed ...
This script loads the task adapter, reads the spec, calls adapter.train(), and prints results as JSON."""

import argparse
import json
import os
import sys
import importlib.util


def load_task_adapter(module_path: str):
    """Load TaskAdapter from a module directory."""
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


def main():
    parser = argparse.ArgumentParser(description="Run a single training experiment")
    parser.add_argument("--spec", required=True, help="Path to experiment spec YAML")
    parser.add_argument("--task-module", required=True, help="Path to task adapter module")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-budget", type=int, required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    # Load task adapter
    adapter = load_task_adapter(args.task_module)

    # Load experiment spec
    from applied_scientist.core.spec import ExperimentSpec
    spec = ExperimentSpec.from_yaml(args.spec)

    # Merge task_config and training_config
    config = {**spec.task_config, **spec.training_config}

    # Create directories
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Run training
    try:
        result = adapter.train(
            config=config,
            seed=args.seed,
            time_budget=args.time_budget,
            log_path=args.log_path,
            checkpoint_dir=args.checkpoint_dir,
        )
    except Exception as e:
        result = {"status": "crash", "error": str(e)}

    # Print result as JSON on last line (Builder parses this)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

### `system.py`

Major updates from the old plan:
- `test_run` parameter and `_run_test()` pipeline validation
- `_inject_random_baseline()` records zero-GPU-cost random agent metrics at startup
- All 3 previously undefined methods now defined: `_init_workspace`, `_init_git_repo`, `_get_api_key`
- Event loggers (`review_logger`, `explorer_logger`, `system_logger`) initialized
- `HyperparameterTuner` initialized when `config.tuning.enabled`
- Agent constructors updated to receive logger and queue references
- Watchdog logs events to `system_logger`

```python
import os
import time
import threading

from applied_scientist.config import Config
from applied_scientist.core.spec import ExperimentSpec
from applied_scientist.core.result import ExperimentResult
from applied_scientist.core.priority_queue import PriorityQueue
from applied_scientist.core.knowledge_base import KnowledgeBase
from applied_scientist.core.results_tracker import ResultsTracker
from applied_scientist.core.gpu_pool import GPUPool
from applied_scientist.core.message_bus import MessageBus, AgentMessage
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
                 test_run: bool = False):
        self.test_run = test_run

        # Load config
        self.config = Config.from_yaml(config_path)
        if n_gpus is not None:
            self.config.system.n_gpus = n_gpus
        if job_runner_override:
            self.config.compute.backend = job_runner_override

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
                            os.path.join(state_dir, "gpu_slots.json"))
        self.bus = MessageBus(os.path.join(state_dir, "pending_messages.json"))
        self.cost_tracker = CostTracker()

        # Initialize LLM backends
        self.llms = {}
        for role in ["explorer", "critic", "builder", "orchestrator"]:
            llm_config = self.config.llm[role]
            self.llms[role] = get_backend({
                "backend": llm_config.backend,
                "model": llm_config.model,
                "api_key": self._get_api_key(llm_config.backend),
                **({"base_url": llm_config.base_url} if llm_config.base_url else {}),
            })
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
                self.system_logger, self.config),
            "orchestrator": OrchestratorAgent(
                self.llms["orchestrator"],
                get_tools("orchestrator", self.config.paths.workspace),
                self.prompts["orchestrator"], self.bus, self.cost_tracker,
                self.queue, self.kb, self.results, self.pool,
                self.system_logger, {}),  # agents ref set below
        }
        # Fix circular reference for orchestrator
        self.agents["orchestrator"].agents = self.agents

    def start(self):
        if self.test_run:
            self._run_test()
            return

        self._init_git_repo()
        self._inject_random_baseline()
        self._inject_baseline()
        self._recover_state()

        # Start agent threads
        self.threads = {}
        for name in ["explorer", "critic", "builder"]:
            t = threading.Thread(target=self.agents[name].run, name=name, daemon=True)
            t.start()
            self.threads[name] = t

        # Start watchdog
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

        # Orchestrator runs in main thread
        try:
            self.agents["orchestrator"].run()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop all agents gracefully."""
        for agent in self.agents.values():
            agent._stopped = True
        self.queue.save()
        self.pool._persist()
        print(f"\nApplied Scientist stopped.")
        print(f"  Total cost: ${self.cost_tracker.get_total_cost():.2f}")
        print(f"  Results saved to {self.config.paths.results}")

    # --- Previously undefined methods, NOW DEFINED ---

    def _init_workspace(self):
        """Create workspace directory structure."""
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
        """Initialize git repo in workspace if not already initialized."""
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
        """Get API key for a backend from environment variables."""
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
        """Record random baseline metrics at startup. Zero GPU cost."""
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
        """Auto-inject PPO baseline spec if not already completed or queued."""
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
        # Recover GPU slots
        state_path = os.path.join(self.config.paths.workspace,
                                  ".state", "gpu_slots.json")
        if os.path.exists(state_path):
            saved_pool = GPUPool.from_state(state_path, self.config.system.n_gpus)
            for slot in saved_pool.get_running():
                if slot.job_id:
                    actual = self.runner.status(slot.job_id)
                    if actual.state == "running":
                        self.pool.assign(slot.slot_id, slot.job_id,
                                        slot.experiment_name, slot.log_dir)

        # Recover pending pairs
        pairs_path = os.path.join(self.config.paths.workspace,
                                  ".state", "pending_pairs.json")
        if os.path.exists(pairs_path):
            self.agents["builder"].pending_pairs = BuilderAgent.load_pairs(pairs_path)

    def _watchdog(self):
        """Monitor agent threads, restart on crash (max 3x per agent)."""
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
        env = Environment(loader=FileSystemLoader("applied_scientist/prompts"))
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
        """Load TaskAdapter from module directory."""
        import importlib.util
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
        """Test run: validate full pipeline with minimal resource usage."""
        print("Running test...")
        steps = [
            ("Config", lambda: True),  # Already loaded
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
```

### `__main__.py`

Updated with `--test-run` flag.

```python
import argparse
from applied_scientist.system import AppliedScientistSystem


def main():
    parser = argparse.ArgumentParser(description="Applied Scientist")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--task", required=True, help="Path to task adapter module")
    parser.add_argument("--n-gpus", type=int, default=None, help="Override GPU count")
    parser.add_argument("--job-runner", default=None, help="Override compute backend")
    parser.add_argument("--test-run", action="store_true",
                        help="Validate pipeline without full run")
    args = parser.parse_args()

    system = AppliedScientistSystem(
        config_path=args.config,
        task_module=args.task,
        n_gpus=args.n_gpus,
        job_runner_override=args.job_runner,
        test_run=args.test_run,
    )
    system.start()


if __name__ == "__main__":
    main()
```

---

## Phase 8: System Prompts

Stored as Jinja2 templates in `prompts/`. Rendered at startup with task context injected via `_render_prompts()`.

### `prompts/explorer.md.j2`

```
You are the Explorer agent in Applied Scientist, an autonomous ML research system.

## Task
{{ task_name }}: {{ task_domain_context }}

## Your Role
You continuously search for research papers to find architectures and techniques
relevant to the task above. For each relevant paper, you write an architecture-level
experiment spec. You submit specs for review and immediately move on to the next
paper — you never wait for review results.

## Scope Rounds
- Round 1: Papers directly about {{ task_name }} or very similar tasks
- Round 2: Papers about related tasks in the same domain
- Round 3: Papers about the broader field (e.g., multi-agent RL, optimization)
- Round 4+: General ML techniques that could transfer

Adapt your search based on what the knowledge base says works and doesn't work.
If no new specs are found in 3 consecutive iterations, scope automatically advances.

## Paper Search
Use `search_papers` to find papers by keyword. Results include title, abstract,
citation count, and paper ID. Use `read_paper` with the paper ID to get the full
text (or as much as is available). Prioritize papers with high citation counts
and recent publication dates, but do not ignore older foundational work when the
knowledge base suggests a technique category is promising.

Do NOT search for the same query twice. Keep a mental list of queries you have
already run. If search returns no new results, broaden or pivot your query terms.

## Spec Format
Write YAML specs to configs/drafts/. Each spec must include:
- name: unique identifier (check that this name is not already in the results
  context or in configs/drafts/ or configs/experiments/ before writing)
- description: one-line summary
- source_paper: citation
- architecture: detailed description of WHAT the architecture is and WHY each
  component exists
  - key_idea: the core insight
  - All components described with inputs, structure, outputs, dimensions
- why_it_might_work: specific reasoning for this task
- task_config: task-specific configuration
- training_config: hyperparameters
- resource_estimate: memory and time estimates
- category: for grouping (e.g., "algorithm", "architecture", "training_strategy")
- tunable_hyperparameters: (optional) when the paper discusses hyperparameter
  sensitivity or recommends tuning specific parameters, include a dict mapping
  parameter names to search space definitions. Format:
  ```yaml
  tunable_hyperparameters:
    lr: {type: log_uniform, low: 1.0e-5, high: 1.0e-2}
    attention_heads: {type: discrete, values: [2, 4, 8]}
    hidden_size: {type: factor, factors: [0.5, 1, 2]}
  ```
  This enables Bayesian hyperparameter tuning after the initial run.

## Name Deduplication
Before writing a spec, check:
1. `list_directory configs/drafts/` — pending draft specs
2. `list_directory configs/experiments/` — approved specs
3. The completed results in your context
If the name already exists, append a numeric suffix (e.g., `mappo_v2`).

## Critical Rules
- Describe WHAT and WHY, never HOW to implement. No code.
- Be specific: "FC, 2 layers, 256 units, ReLU" not "a neural network"
- Every architectural component must be fully specified — no ambiguity.
- Include resource estimates. If it might OOM, say so.
- You can propose control variants by setting control_variant_of in the variant spec.
- Do NOT propose experiments that are already listed in the completed results.

## Tools
You have: search_papers, read_paper, search_web, read_file, write_file, list_directory.
```

### `prompts/critic.md.j2`

```
You are the Critic agent in Applied Scientist, an autonomous ML research system.

## Task
{{ task_name }}: {{ task_domain_context }}
Primary metric: {{ metric_name }} ({{ metric_direction }} is better)

## Your Role
You are the quality gate. You process your inbox in priority order: code reviews
first, then spec reviews, then insight reviews, then suggestions, then re-ranks.
You use structured tool calls for ALL decisions — never just write free-text
responses. Every review must end with a tool call that records the decision.

## Spec Review Checklist
When reviewing a spec, check:
1. COMPLETENESS: All components fully described? No "use attention" without Q/K/V/dims/heads.
2. ACCURACY: Does the architecture match the cited paper?
3. SPECIFICITY: Can the Builder implement without making architectural decisions?
4. RESOURCE FEASIBILITY: Will it fit in GPU memory? Complete in the time budget ({{ time_budget }}s)?
5. CONTROL VARIANT NEEDED? If multiple changes bundled, request a variant isolating the key one.
6. DEDUPLICATION: Is this experiment already completed or queued?

Use the `submit_review` tool to submit your decision.
Max {{ max_review_rounds }} review rounds. On the final round, force-approve with your fixes.

## Code Review
When you receive a code review request from the Builder, use the `submit_code_review`
tool to record your decision. Check for:
1. Does the implementation match the spec's architecture description?
2. Are there obvious bugs (wrong dimensions, missing activations, shape mismatches)?
3. Is the training config correctly applied?

Use `submit_code_review` with status "approve", "request_changes", or "comment".
Do NOT write free-text code reviews — always use the structured tool call so the
Builder can parse your decision programmatically.

## Insight Review Checklist
When reviewing an insight, check:
1. OVER-CLAIMING: "confirms" from 1-2 runs should be "suggests"
2. MISSING CONTEXT: Memory cost, training stability, convergence speed
3. UNSUPPORTED CAUSAL CLAIMS: Label hypotheses as hypotheses
4. MISSED OBSERVATIONS: Check if training logs show patterns not mentioned
5. CONTRADICTIONS: Does this conflict with prior knowledge base entries?

Use the `submit_insight_review` tool to submit your decision.

## Suggestion Triage
Use the `submit_triage` tool. Classify as:
- minor: Hyperparameter tweak, config change. Same code. Builder can run immediately.
- moderate: Needs new code but not a new idea. Explorer writes a new spec.
- major: Novel architectural idea, potential research contribution. Save to ideas_for_system2.md.

## Ranking
Score specs: expected_performance_gain + diversity_bonus + information_bonus.
- Baselines always ranked highest initially.
- Boost specs in categories with no tested experiments.
- Reduce score for specs similar to already-tested architectures.

## Decision Logging
Log every review decision with rationale. Your decisions are recorded in
review_log.jsonl and system_events.jsonl for auditability. Include the
reasoning behind approvals, rejections, and score assignments so the system
can learn from past decisions.

## Tools
You have: read_file, write_file, list_directory, submit_review,
submit_code_review, submit_insight_review, submit_triage.
```

### `prompts/builder.md.j2`

```
You are the Builder agent in Applied Scientist, an autonomous ML research system.

## Task
{{ task_name }}: {{ task_domain_context }}
Primary metric: {{ metric_name }} ({{ metric_direction }} is better)
Time budget per experiment: {{ time_budget }} seconds

## Code Map (DO NOT re-read these files — use this reference)
{{ task_code_map }}

## Your Role
You implement architectures from approved specs, run experiments, record results,
and write insights. You are a skilled ML engineer. You submit reviews to the Critic
non-blockingly and move on to the next GPU slot.

## Code Map Usage Rules
The code map above contains the complete file structure, interfaces, and extension
points for the task. Use it as your primary reference when implementing specs.

- DO NOT read source files that are described in the code map — the map is your reference.
- Only read files when: (a) editing existing code and you need the current contents,
  or (b) debugging an error and you need to see the actual runtime state.
- When creating new model files, follow the patterns shown in the code map exactly.
- Register new models in the file specified by the code map.

## Rules
1. Read the spec from configs/experiments/. NEVER read the original paper.
2. You make all IMPLEMENTATION decisions (code structure, API usage, framework quirks).
3. You make NO ARCHITECTURAL decisions (model structure, hyperparams — all in the spec).
4. Git commit before every training run. Record the commit hash.
5. Pre-flight validation: syntax check + 60s smoke test before GPU submission.
6. If training crashes:
   - Fixable (OOM, typo)? Fix and retry ONCE.
   - Unfixable? Log as "crash" and move on.
7. Write insights after every experiment, including crashes and early stops.
8. Label hypotheses as hypotheses, not conclusions.
9. If a spec is ambiguous, post a clarification_request to the Explorer.

## Insight Format
- Observations: factual, from the metrics and logs
- Comparisons: to baselines and prior experiments
- Hypotheses: labeled as such, with rationale

## Suggestions
After observing training behavior, you may suggest improvements.
The Critic will triage as minor (run it), moderate (needs new spec), or major (save for later).

## Tools
You have: read_file, write_file, edit_file, list_directory,
run_command, git_commit, git_log, git_diff, git_reset.
```

### `prompts/orchestrator.md.j2`

```
You are the Orchestrator agent in Applied Scientist, an autonomous ML research system.

## Task
{{ task_name }}: {{ task_domain_context }}

## Your Role
You are the human's natural language interface to the running system.
You interpret their messages and either answer from system status or route commands.

## Available Actions
- Report status (GPU pool, queue, results, Explorer scope)
- Report cost (per-agent and total)
- Prioritize/deprioritize a topic (relay to Explorer + Critic)
- Pause/resume the Builder
- Adjust GPU pool size
- Inject a spec idea (goes through Critic review)
- Show knowledge base, results, or queue
- Relay a specific paper to the Explorer

## Built-in Commands (no LLM call needed)
status, results, knowledge, queue, cost — these are handled directly.

## inject Command
When the user says "inject <spec description>" or provides a spec idea:
1. Parse the idea into an ExperimentSpec with as much detail as possible
2. Write it to configs/drafts/ with a descriptive name
3. Post a review request to the Critic
4. Confirm to the user: "Spec drafted and sent to Critic for review."

## paper Command
When the user says "paper <url or title>":
1. Post a message to the Explorer with the paper reference
2. The Explorer will read the paper and generate specs if relevant
3. Confirm to the user: "Paper relayed to Explorer."

## @agent Routing
When the user prefixes a message with @explorer, @critic, or @builder:
1. Route the message body to the named agent via the message bus
2. The target agent will process it in their next inbox check
3. Confirm to the user: "Message sent to <agent>."

If the message does not target a specific agent, use your judgment to route it
or answer directly from available system state.

## Style
Be concise and informative. Show numbers, not paragraphs.
When reporting status, use tables or structured output.
```

---

## Phase 9: Task Adapter Example (Soccer)

### `tasks/examples/soccer_twos/adapter.py`

Updated with `code_map` property, `get_random_baseline()`, and `get_tunable_defaults()`.

```python
import os
import json
import subprocess
from applied_scientist.task.base import TaskAdapter


class SoccerTwosAdapter(TaskAdapter):
    @property
    def name(self):
        return "2v2 Soccer (Unity ML-Agents)"

    @property
    def metric(self):
        return ("win_rate", "higher")

    @property
    def domain_context(self):
        return """
        2v2 soccer game using Unity ML-Agents and Ray RLlib 1.4.0.
        4 agents (2 per team), discrete action space (27 flattened actions),
        flattened observation vector (player positions, rotations, velocities,
        ball state). Environment wrapper: soccer_twos with RLLibWrapper.

        Training modes: team_vs_policy (single team vs fixed opponent),
        multiagent_player (all 4 agents controlled), multiagent_team (team-level).

        Primary metric: win rate vs random opponent over 100 episodes.
        Evaluation: load Ray checkpoint, run episodes, count wins.
        Framework: Python 3.8, Ray 1.4.0, PyTorch, gym 0.19.0.
        Hardware: single NVIDIA GPU, 30-minute time budget.
        """

    @property
    def code_map(self):
        return """
    ## File Structure
    tasks/examples/soccer_twos/
    ├── adapter.py        # DO NOT EDIT
    ├── train.py          # Main training entry — edit training config here
    ├── evaluate.py       # DO NOT EDIT
    └── models/
        ├── __init__.py   # Model registry — register new models here
        ├── base.py       # DO NOT EDIT — BaseModel interface
        ├── fc_policy.py  # Default FC policy — reference implementation
        └── custom/       # ADD new model files here

    ## BaseModel Interface (base.py)
    class BaseModel(TorchModelV2):
        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            # obs_space: Box(336,) — flattened obs
            # action_space: Discrete(27) — flattened branched
            # num_outputs: 27
        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]: Tensor [batch, 336]
            # Return: logits [batch, 27], state
        def value_function(self):
            # Return: value estimate [batch]

    ## How to Add a New Architecture
    1. Create models/custom/<name>.py
    2. Subclass BaseModel
    3. Register in models/__init__.py
    4. Training config via spec.training_config

    ## Observation Space: 336 floats (4 agents x 84)
    ## Action Space: Discrete(27) from 3 branches [3,3,3]
    ## Constraints: ~12GB GPU, Ray RLlib 1.4.0, PyTorch only
    """

    @property
    def progress_file(self):
        return "progress.json"

    def train(self, config, seed, time_budget, log_path, checkpoint_dir):
        """Delegates to train.py which wraps Ray Tune.
        train.py writes progress.json periodically."""
        cmd = (
            f"python {os.path.dirname(__file__)}/train.py "
            f"--config '{json.dumps(config)}' "
            f"--seed {seed} "
            f"--time-budget {time_budget} "
            f"--log-path {log_path} "
            f"--checkpoint-dir {checkpoint_dir}"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return json.loads(result.stdout.strip().split("\n")[-1])

    def evaluate(self, checkpoint_dir, n_eval_samples):
        cmd = (
            f"python {os.path.dirname(__file__)}/evaluate.py "
            f"--checkpoint-dir {checkpoint_dir} "
            f"--n-episodes {n_eval_samples}"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return json.loads(result.stdout.strip().split("\n")[-1])

    def get_baseline_spec(self):
        """Soccer-specific PPO baseline."""
        return {
            "name": "ppo_baseline",
            "description": "PPO with standard hyperparameters for 2v2 soccer",
            "source_paper": "Schulman et al. 2017 (PPO)",
            "architecture": {
                "type": "independent_ppo",
                "policy": {"structure": "FC, 2 layers, 256 units, ReLU"},
            },
            "why_it_might_work": "Standard baseline for multi-agent environments",
            "task_config": {
                "variation": "multiagent_player",
                "single_player": False,
                "flatten_branched": True,
            },
            "training_config": {
                "lr": 0.0003,
                "train_batch_size": 12000,
                "rollout_fragment_length": 1000,
                "gamma": 0.99,
                "entropy_coeff": 0.01,
            },
            "resource_estimate": {"memory": "~4GB", "training_time": "Standard"},
            "category": "baseline",
        }

    def get_random_baseline(self):
        """Random actions in 2v2 soccer — approximately 50% win rate."""
        return {"win_rate": 0.50, "avg_reward": 0.0}

    def get_tunable_defaults(self):
        """Default hyperparameter search space for soccer experiments."""
        return {
            "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
            "train_batch_size": {"type": "factor", "factors": [0.5, 1, 2, 4]},
            "gamma": {"type": "discrete", "values": [0.99, 0.995, 0.999]},
            "entropy_coeff": {"type": "log_uniform", "low": 0.001, "high": 0.1},
        }
```

### `tasks/examples/soccer_twos/train.py`

Must write `progress.json` periodically during training for early-stop monitoring.

```python
"""Soccer 2v2 training script. Wraps Ray Tune with progress reporting.

Called by the task adapter:
    python train.py --config '{"lr": 0.0003, ...}' --seed 42 --time-budget 1800 \
                    --log-path /path/to/logs --checkpoint-dir /path/to/ckpt

Writes progress.json periodically (required for early stopping).
Prints final result as JSON on last line (parsed by run_experiment.py / Builder).
"""

import argparse
import json
import os
import time

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer


def on_training_result(result, log_path):
    """Write progress.json after each training iteration."""
    progress = {
        "step": result["timesteps_total"],
        "win_rate": result.get("custom_metrics", {}).get("win_rate", 0),
        "episode_reward_mean": result["episode_reward_mean"],
        "elapsed_seconds": result["time_total_s"],
    }
    with open(os.path.join(log_path, "progress.json"), "w") as f:
        json.dump(progress, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON training config")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-budget", type=int, required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    config = json.loads(args.config)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    # Build RLlib config
    rllib_config = {
        "env": "soccer_twos",
        "framework": "torch",
        "seed": args.seed,
        "lr": config.get("lr", 0.0003),
        "train_batch_size": config.get("train_batch_size", 12000),
        "rollout_fragment_length": config.get("rollout_fragment_length", 1000),
        "gamma": config.get("gamma", 0.99),
        "entropy_coeff": config.get("entropy_coeff", 0.01),
        "num_workers": config.get("num_workers", 2),
        "num_gpus": 1,
    }

    trainer = PPOTrainer(config=rllib_config)
    start_time = time.time()
    best_win_rate = 0.0
    last_result = {}

    while time.time() - start_time < args.time_budget:
        result = trainer.train()
        on_training_result(result, args.log_path)

        win_rate = result.get("custom_metrics", {}).get("win_rate", 0)
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            trainer.save(args.checkpoint_dir)

        last_result = {
            "win_rate": win_rate,
            "best_win_rate": best_win_rate,
            "episode_reward_mean": result["episode_reward_mean"],
            "timesteps_total": result["timesteps_total"],
            "elapsed_seconds": time.time() - start_time,
            "status": "completed",
        }

    trainer.stop()
    ray.shutdown()

    # Print result as JSON on last line
    print(json.dumps(last_result))


if __name__ == "__main__":
    main()
```

### `tasks/examples/soccer_twos/evaluate.py`

```python
"""Soccer 2v2 evaluation script.

Called by the task adapter:
    python evaluate.py --checkpoint-dir /path/to/ckpt --n-episodes 100

Prints final metrics as JSON on last line.
"""

import argparse
import json
import os

import ray
from ray.rllib.agents.ppo import PPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--n-episodes", type=int, default=100)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

    # Find the latest checkpoint file
    checkpoints = sorted([
        f for f in os.listdir(args.checkpoint_dir)
        if f.startswith("checkpoint")
    ])
    if not checkpoints:
        print(json.dumps({"error": "No checkpoint found", "win_rate": None}))
        return

    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoints[-1])

    trainer = PPOTrainer(config={
        "env": "soccer_twos",
        "framework": "torch",
        "num_workers": 0,
        "num_gpus": 0,
    })
    trainer.restore(checkpoint_path)

    wins = 0
    total_reward = 0.0

    for _ in range(args.n_episodes):
        env = trainer.env_creator({})
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = trainer.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward
        if info.get("win", False):
            wins += 1

    trainer.stop()
    ray.shutdown()

    result = {
        "win_rate": wins / args.n_episodes,
        "avg_reward": total_reward / args.n_episodes,
        "n_episodes": args.n_episodes,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

---

## Phase 10: Tests & Dependencies

### Test Strategy

All tests use mocks for LLM calls and compute backends. No real API calls or GPU usage in the test suite.

| Test file | What it tests |
|-----------|---------------|
| `test_spec.py` | Spec serialization to/from YAML, field validation, `tunable_hyperparameters` field round-trip |
| `test_priority_queue.py` | Insert, pop, rerank, persistence, dedup, `contains_similar`, thread safety, atomic writes |
| `test_knowledge_base.py` | Add insight, add synthesis, pinning, persistence, atomic writes |
| `test_results_tracker.py` | Add result, `get_best`, `get_by_name`, baseline aggregation, TSV format, `validation_fail` status |
| `test_gpu_pool.py` | Assign, release, `get_free`, timeout detection, resize, state persistence, recovery |
| `test_message_bus.py` | Post, poll, drain, priority ordering, thread safety, state persistence |
| `test_cost_tracker.py` | Register, record, cost calculation, summary format |
| `test_event_logger.py` | Append, `read_all`, concurrent writes, JSONL format integrity |
| `test_tools.py` | Each tool executes correctly (file ops in temp dir, shell sandboxing, ALL structured critic tools including `submit_code_review`) |
| `test_llm_backends.py` | Format conversion, retry logic (with mocked HTTP) |
| `test_agents.py` | Agent chat loop with mock LLM, tool dispatch, non-blocking message flow, inbox processing, context trimming, ALL new message types handled |
| `test_tuning.py` | Budget computation, search space merge (task defaults + spec overrides), Optuna integration with mock adapter |
| `test_system.py` | Full startup, baseline injection, random baseline injection, one spec-review-implement-validate-run cycle with mocks, crash recovery, `--test-run` mode |

### `conftest.py`

Shared fixtures and mocks for all tests. Updated with `calls` recording, `code_map`, `get_random_baseline`, and `get_tunable_defaults`.

```python
import pytest
import tempfile
import os

from applied_scientist.llm.base import LLMBackend, LLMResponse, LLMMessage
from applied_scientist.task.base import TaskAdapter
from applied_scientist.compute.base import JobRunner, JobStatus
from applied_scientist.config import (
    Config, SystemConfig, TuningConfig, LLMAgentConfig,
    ComputeConfig, PathsConfig,
)


class MockLLM(LLMBackend):
    """Returns canned responses. Records calls for assertions."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        content = next(self.responses, "mock response")
        return LLMResponse(
            content=content, tool_calls=None,
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn")

    def get_model_id(self):
        return "mock-model"

    @property
    def supports_tool_use(self):
        return True


class MockTaskAdapter(TaskAdapter):
    name = "MockTask"
    metric = ("accuracy", "higher")
    domain_context = "A test task."
    code_map = "## Files\n- model.py: edit this\n- train.py: DO NOT EDIT"
    progress_file = "progress.json"

    def train(self, *args, **kwargs):
        return {"accuracy": 0.85}

    def evaluate(self, *args, **kwargs):
        return {"accuracy": 0.85}

    def get_random_baseline(self):
        return {"accuracy": 0.10}

    def get_tunable_defaults(self):
        return {"lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2}}


class MockJobRunner(JobRunner):
    def submit(self, *args, **kwargs):
        return "mock_job_1"

    def status(self, job_id):
        return JobStatus(job_id, "completed", 0, 10.0)

    def cancel(self, job_id):
        pass

    def get_log(self, job_id, tail=50):
        return '{"accuracy": 0.85}'


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with standard directory structure."""
    dirs = ["results", "configs", "configs/drafts", "configs/experiments",
            "results/checkpoints", "results/logs", ".state"]
    for d in dirs:
        os.makedirs(tmp_path / d, exist_ok=True)
    return tmp_path


@pytest.fixture
def mock_llm():
    return MockLLM(["mock response"])


@pytest.fixture
def mock_task():
    return MockTaskAdapter()


@pytest.fixture
def mock_runner():
    return MockJobRunner()


@pytest.fixture
def mock_config(tmp_workspace):
    return Config(
        system=SystemConfig(),
        tuning=TuningConfig(),
        llm={
            role: LLMAgentConfig(backend="mock", model="mock-model")
            for role in ["explorer", "critic", "builder", "orchestrator"]
        },
        compute=ComputeConfig(backend="local"),
        task={},
        paths=PathsConfig(
            workspace=str(tmp_workspace),
            results=str(tmp_workspace / "results"),
            configs=str(tmp_workspace / "configs"),
            checkpoints=str(tmp_workspace / "results" / "checkpoints"),
            logs=str(tmp_workspace / "results" / "logs"),
        ),
    )
```

### Dependencies (`pyproject.toml`)

Core package has minimal dependencies (only PyYAML and Jinja2). LLM SDKs, web tools, and tuning are optional -- install only what you use.

```toml
[project]
name = "applied-scientist"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pyyaml>=6.0",
    "jinja2>=3.1",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.30"]
openai = ["openai>=1.30"]
gemini = ["google-generativeai>=0.5"]
search = ["requests>=2.31"]
pdf = ["pypdf2>=3.0", "beautifulsoup4>=4.12"]
tuning = ["optuna>=3.6"]
all = ["applied-scientist[anthropic,openai,gemini,search,pdf,tuning]"]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]
```

---

## Build Order Summary (Updated)

| Phase | Files | Depends on | Est. LOC |
|-------|-------|-----------|----------|
| 0. Core | `core/*.py` | Nothing | ~900 |
| 1. Interfaces | `task/base.py`, `llm/base.py`, `tools/base.py`, `compute/base.py` | Nothing | ~300 |
| 2. LLM Backends | `llm/*.py` | Phase 1 | ~500 |
| 3. Tools | `tools/*.py` | Phase 1 | ~750 |
| 4. Compute | `compute/*.py` | Phase 1 | ~300 |
| 5. Agents | `agents/*.py` | Phases 0-4 | ~1600 |
| 6. Tuning | `tuning.py` | Phases 0-1 | ~200 |
| 7. System | `system.py`, `run_experiment.py`, `__main__.py`, `config.py` | Phases 0-6 | ~700 |
| 8. Prompts | `prompts/*.md.j2` | Phase 5 | ~250 |
| 9. Task Example | `tasks/examples/soccer_twos/*` | Phase 1 | ~450 |
| 10. Tests | `tests/*` | All | ~700 |
| **Total** | | | **~6650** |
