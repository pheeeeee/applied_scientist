# Applied Scientist: Autonomous Multi-Agent System for ML Experiment Automation

**Applied Scientist** is a task-agnostic, LLM-agnostic multi-agent framework that automates the benchmark sweep phase of machine learning research. It autonomously surveys literature, designs experiments, implements architectures, trains models, and builds a curated knowledge base of what works and why.

You define the task. The system supplies the research team.

```
                       You
                        |
                   Orchestrator       natural language interface
                   /    |    \
             Explorer  Critic  Builder
                |        |       |
                |        |    GPU Pool
                v        v
             Shared Knowledge Base
```

Four AI agents — **Explorer** (literature), **Critic** (quality gate), **Builder** (implementation), **Orchestrator** (your interface) — run concurrently on CPU. Training jobs are dispatched to a pool of GPUs. The system is designed so that:

- **Any ML task** can be plugged in via a Task Adapter (RL, NLP, CV, etc.)
- **Any LLM** can power the agents (Claude, GPT, Gemini, Llama, or any OpenAI-compatible endpoint)
- **Any compute backend** can run training (SLURM, local GPU, cloud)
- **Minimal blocking** — inter-agent communication is async, except plan reviews (Builder waits for Critic approval before implementing)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Bring Your Own](#bring-your-own)
  - [Task](#bring-your-own-task)
  - [LLM](#bring-your-own-llm)
  - [Compute](#bring-your-own-compute)
- [Interacting with the System](#interacting-with-the-system)
- [Experiment Lifecycle](#experiment-lifecycle)
- [Output](#output)
- [Architecture](#architecture)
- [Deployment](#deployment)
- [Cost Estimates](#cost-estimates)
- [Related Work](#related-work)
- [Citation](#citation)

---

## How It Works

The system runs three concurrent stages in a continuous loop:

**Stage 1 — Spec Generation (Explorer + Critic)**

The Explorer agent continuously reads research papers in expanding scope: first task-specific literature, then adjacent domains, then general ML. If no new specs are found in 3 consecutive iterations, scope automatically advances. For each relevant architecture it finds, it writes an *architecture-level spec* — a structured description of *what* the architecture is and *why* each component exists, without prescribing implementation details. The Explorer submits each spec to the Critic for review and **immediately moves on** to the next paper — it never blocks waiting for a review result.

The Critic picks up review requests from its inbox in priority order. It checks completeness, accuracy, resource feasibility, and deduplication (rejecting specs for experiments already completed or queued). It also checks whether a control variant is needed (max 3 review rounds). The Critic uses structured tool calls for all decisions, ensuring reliable parsing. Approved specs are scored and inserted into a ranked priority queue.

**Stage 2 — Plan, Build & Run (Builder + Critic)**

The Builder manages a pool of GPU slots. Whenever a slot frees up, the Builder pulls the top-ranked spec from the queue and follows a structured implementation pipeline:

1. **Plan** — The Builder writes a `PLAN.md` describing every file it will create, key classes/functions, and how components connect. Each experiment gets its own **self-contained directory** (`workspace/experiments/<name>/`) with all code needed to train — this allows fundamentally different algorithms (PPO, MADDPG, SAC, hierarchical RL) rather than just model swaps.

2. **Plan Review (blocking)** — The Critic reviews the implementation plan for completeness, architecture match, environment compatibility, and the `train.py` contract (CLI args, progress reporting, result output). If rejected, the Builder revises (max 3 rounds). This is the only blocking inter-agent call — it's cheap (one LLM round) but prevents wasted GPU time from flawed implementations.

3. **Implement** — Once the plan is approved, the Builder reads the baseline `train.py` as a reference before implementing, then writes all files into the experiment directory using `write_file` tool calls.

4. **Validate & Submit** — Pre-flight validation (syntax check + 10-second smoke test), with up to 3 attempts to fix validation failures before abandoning the experiment. On success: git commit and training job submission. Code reviews for structural changes are submitted to the Critic non-blockingly — the job is submitted optimistically while the review is pending.

Baseline experiments run with multiple seeds **sequentially on one slot** (not consuming the entire GPU pool). An **early-stopping** mechanism polls a progress file written by the training script, killing experiments showing less than 10% of baseline performance after 40% of the time budget. No GPU ever waits for another — the pool is always maximally utilized.

**Stage 3 — Knowledge Capture (Builder + Critic)**

After each experiment, the Builder writes a draft insight (observations, comparisons to prior results, hypotheses) and submits it to the Critic non-blockingly. The Critic reviews for over-claiming, missed context, and accuracy using structured tool calls. Approved insights enter a curated knowledge base with synthesis summaries pinned at the top. The Builder can also suggest improvements; the Critic triages these as minor tweaks (run immediately), moderate changes (new spec needed), or major novel ideas (saved for deeper investigation later).

The Explorer reads the knowledge base continuously, adapting its literature search based on what the system has learned.

---

## Prerequisites

**Required:**

- Python 3.10+ for the orchestrator (training environments may use Python 3.8+)
- git
- At least one LLM API key (Anthropic, OpenAI, or Google)
- At least one GPU for training (local or cluster)

**For SLURM clusters:**

- Access to `sbatch`, `squeue`, `sacct`, `scancel` commands
- A conda/venv environment with your task's training dependencies
- Outbound HTTPS access from compute nodes (for LLM API calls)

  Test with: `curl -s https://api.anthropic.com/v1/messages -o /dev/null -w "%{http_code}"`
  — `401` means reachable, timeout means firewalled.

**For local machines:**

- NVIDIA GPU with CUDA installed
- `nvidia-smi` available

**Optional (for Explorer's paper search):**

- Web search API key (Serper, SerpAPI, or Google Custom Search)
- No key needed for Semantic Scholar (free, rate-limited)

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/pheeeeee/applied-scientist.git
cd applied-scientist
pip install -e ".[all]"       # all LLM backends
# or: pip install -e ".[anthropic]"   # Claude only
# or: pip install -e ".[openai]"      # GPT only
```

### 2. Set API keys

**On your own machine (single user):**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."    # if using Claude
export OPENAI_API_KEY="sk-..."           # if using GPT
export GOOGLE_API_KEY="AI..."            # if using Gemini
```

Add these to your `.bashrc` or `.zshrc` to persist across sessions.

**On a shared server / HPC cluster (e.g., PACE, SLURM):**

On multi-user systems, store keys in a dedicated file with strict permissions so other users cannot read them:

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets

cat > ~/.secrets/autoresearch.env << 'EOF'
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AI..."
EOF

chmod 600 ~/.secrets/autoresearch.env
```

Then source it before running:

```bash
source ~/.secrets/autoresearch.env
```

For SLURM job scripts, source at the top:

```bash
#!/bin/bash
#SBATCH --partition=gpu
source ~/.secrets/autoresearch.env
python -m applied_scientist --config configs/default.yaml --task tasks/examples/soccer_twos
```

Why this approach:
- `chmod 600` — only your user can read the file, other users on the cluster cannot
- Outside the repo — cannot be accidentally git committed
- Not in `.bashrc` — keys are only loaded when you need them, not in every shell session

**Do not** put API keys in `.env` files inside the repo, config YAML files, Python source files, or command-line arguments (visible in `ps aux`).

### 3. Configure

```bash
cp configs/default.yaml configs/my_config.yaml
# Edit: set LLM backend, compute backend, task path
```

### 4. Define your task

Use a provided example task, or write your own Task Adapter (see [Bring Your Own Task](#bring-your-own-task)).

**You must write a Task Adapter before starting the system.** This is the only code you need to write — it tells the system what your task is, how to train, how to evaluate, and what hyperparameters are tunable. The system is task-agnostic; without an adapter, it doesn't know anything about your problem. No agent creates or modifies the adapter during a run — you write it once, the agents call it.

### 5. Pre-flight checklist

Before running, verify everything is in place:

```
[ ] Task Adapter
    [ ] adapter.py with TaskAdapter subclass — defines name, metric, domain_context,
        train(), evaluate(), get_tunable_defaults(), and optionally get_baseline_spec()
    [ ] train.py — training script that writes progress.json periodically
        (required for early stopping)
    [ ] evaluate.py — evaluation script that returns metrics as JSON
    [ ] All three files in one directory (e.g., tasks/examples/soccer_twos/)

[ ] Training pipeline verified manually
    [ ] Ran at least one full training experiment by hand (not through the system)
    [ ] Confirmed it completes within the time_budget you plan to set
    [ ] Noted peak GPU memory usage (for resource_estimate in specs)
    [ ] Confirmed progress.json is written during training
    [ ] Confirmed evaluate.py produces correct metrics from a checkpoint

[ ] Configuration
    [ ] Config YAML with correct LLM backends, compute backend, paths
    [ ] time_budget set based on your manual training run
    [ ] n_gpus matches your available GPU slots

[ ] Environment
    [ ] Python 3.10+ (orchestrator) with applied-scientist installed
    [ ] All task dependencies installed (Ray, PyTorch, etc.)
    [ ] LLM API key(s) set as environment variables
    [ ] Outbound HTTPS access for LLM API calls
        Test: curl -s https://api.anthropic.com/v1/messages -o /dev/null -w "%{http_code}"
        (401 = reachable, timeout = firewalled)

[ ] Compute backend
    [ ] SLURM: sbatch, squeue, sacct, scancel accessible
    [ ] SLURM: conda/venv environment with task dependencies on compute nodes
    [ ] Local: nvidia-smi works, CUDA installed

[ ] Git
    [ ] git installed and configured (Builder commits before every training run)
```

### 6. Test run

Run a dry run to verify the full pipeline without burning GPU hours or LLM budget:

```bash
# Validates: config loading, task adapter import, LLM connectivity,
# compute backend access, workspace initialization, baseline spec injection,
# one full cycle (spec → review → implement → validate → submit → complete)
# using a 10-second time budget and a single GPU slot.
python -m applied_scientist \
  --config configs/my_config.yaml \
  --task tasks/examples/soccer_twos \
  --test-run
```

The test run will:

1. Load and validate config
2. Import your task adapter — check that all methods exist and return correct types
3. Test LLM connectivity — send a trivial prompt to each configured backend
4. Test compute backend — submit a 10-second no-op job, verify status polling and log retrieval
5. Initialize workspace — create directory structure, git init
6. Inject baseline spec and run one full cycle:
   - Explorer drafts one spec (using LLM)
   - Critic reviews it (using LLM)
   - Builder implements, validates (10-second smoke test), commits, submits
   - Training runs with 10-second time budget on 1 GPU
   - Builder records results and writes insight
7. Print summary of what worked and what failed

```
Test run complete.
  Config:         OK
  Task adapter:   OK (2v2 Soccer, metric=win_rate)
  LLM backends:   OK (anthropic/claude-haiku-4-5-20251001, anthropic/claude-sonnet-4-6)
  Compute:        OK (slurm, partition=gpu)
  Workspace:      OK (workspace/ created, git initialized)
  Full cycle:     OK
    Spec drafted:     ppo_baseline
    Review:           approved (score=100.0)
    Implementation:   OK (commit a3f72bc)
    Smoke test:       OK (60s, no crash)
    Training:         OK (60s, win_rate=0.12)
    Results recorded: OK (results.tsv has 1 row)
    Insight written:  OK (knowledge.md updated)
  Estimated cost:   $0.03

System is ready. Run without --test-run to start the full sweep.
```

If any step fails, the test run stops and prints the error with a suggested fix.

### 7. Run

```bash
# Local, single GPU
python -m applied_scientist \
  --config configs/my_config.yaml \
  --task tasks/examples/soccer_twos

# SLURM cluster, 4 GPUs
python -m applied_scientist \
  --config configs/my_config.yaml \
  --task tasks/examples/soccer_twos \
  --job-runner slurm \
  --n-gpus 4

# Interactive mode with human approval checkpoints
python -m applied_scientist \
  --config configs/my_config.yaml \
  --task tasks/examples/soccer_twos \
  --approve-plans \
  --approve-submit
```

**Human Approval Flags:**

| Flag | Description |
|------|-------------|
| `--approve-plans` | Pause after PLAN.md is written, wait for human approval before implementing |
| `--approve-submit` | Pause after code is validated, wait for human approval before using GPU |
| `--approve-all` | Enable all approval checkpoints |

When a checkpoint is reached, the terminal prompts:
```
============================================================
APPROVAL REQUIRED: PLAN
Experiment: mappo_attention
Review: experiments/mappo_attention/PLAN.md
============================================================
[a]pprove / [r]eject / [v]iew file? _
```

Use these flags to:
- Debug failing experiments step-by-step
- Learn how the system works before going fully autonomous
- Ensure human oversight on expensive GPU runs

**Where does output go?** All output is saved under the `paths.workspace` directory from your config. By default this is `./workspace/` relative to where you ran the command:

```
your-project/                          # where you ran the command
├── applied_scientist/                 # source code (installed package)
├── configs/                           # your config files
├── tasks/                             # your task adapter
└── workspace/                         # ALL OUTPUT GOES HERE
    ├── configs/
    │   ├── drafts/                    # specs under review
    │   ├── experiments/               # approved specs (baseline only)
    │   └── priority_queue.yaml        # ranked queue
    ├── experiments/                    # self-contained experiment directories
    │   └── <experiment>/              # EVERYTHING for one experiment in one place
    │       ├── spec.yaml              # copy of experiment spec
    │       ├── queue_snapshot.yaml    # queue state when this was pulled
    │       ├── PLAN.md                # Critic-reviewed implementation plan
    │       ├── train.py               # self-contained training script
    │       ├── model.py               # model/policy code
    │       ├── [other files]          # as needed per algorithm
    │       ├── run_<job_id>.sh        # saved sbatch script
    │       ├── logs/                  # training output (consolidated)
    │       │   ├── slurm-<jobid>.out  # stdout/stderr
    │       │   ├── progress.json      # live training progress
    │       │   └── validation.log     # pre-flight validation output
    │       ├── checkpoints/           # saved model weights
    │       └── errors/                # failure artifacts (if crashed)
    │           ├── validation_error.txt   # smoke test failure
    │           ├── runtime_error.txt      # training crash log
    │           └── traceback.txt          # extracted stack trace
    ├── results/
    │   ├── results.tsv                # experiment leaderboard
    │   ├── knowledge.md               # curated insights
    │   ├── review_log.jsonl           # Critic audit trail
    │   ├── explorer_journal.jsonl     # paper reading log
    │   ├── system_events.jsonl        # system timeline
    │   └── ideas_for_system2.md       # novel ideas
    ├── .state/                        # crash recovery (auto-managed)
    └── .git/                          # version history of all code changes
```

You can override the workspace path in your config to put output anywhere:

```yaml
paths:
  workspace: /scratch/my_project/sweep_01    # absolute path
  # results, configs, checkpoints, logs are derived from workspace by default
```

See [Output](#output) for a detailed description of every file, who writes it, and what it contains.

### 8. Interact

```
> status
> cost
> @explorer focus on attention mechanisms
> results
> pause
```

---

## Configuration

A single YAML file configures the entire system. Copy `configs/default.yaml` and edit for your setup.

```yaml
system:
  time_budget: 1800              # seconds per experiment
  n_gpus: 4                      # GPU slots in the pool
  baseline_seeds: 3              # seeds for baseline experiments (run sequentially on 1 slot)
  eval_samples: 100              # evaluation samples (episodes, test batches, etc.)
  early_stop_fraction: 0.4       # kill if no improvement after this fraction
  max_spec_review_rounds: 3      # max Explorer-Critic review cycles
  queue_throttle_high: 10        # Explorer pauses above this queue depth
  queue_throttle_low: 5          # Explorer resumes below this queue depth
  timeout_multiplier: 2.0        # kill experiment after time_budget * this
  cost_alert_threshold: null     # alert when cumulative LLM cost exceeds this (dollars) # (planned, not yet enforced)

tuning:
  enabled: true                  # enable Bayesian hyperparameter tuning via Optuna
  max_iterations: 15             # max tuning trials per experiment
  min_iterations: 3              # min tuning trials (even under queue pressure)

llm:
  explorer:
    backend: anthropic
    model: claude-haiku-4-5-20251001
  critic:
    backend: anthropic
    model: claude-sonnet-4-6
  builder:
    backend: anthropic
    model: claude-opus-4-6
  orchestrator:
    backend: anthropic
    model: claude-haiku-4-5-20251001

compute:
  backend: slurm                 # slurm | local
  slurm:
    partition: gpu
    account: my-account          # required on most HPC clusters
    gres: "gpu:1"
    mem: "48G"
    time: "01:00:00"
    setup_commands:
      - "source ~/miniconda3/etc/profile.d/conda.sh"
      - "conda activate my_env"

task:
  module: tasks/examples/soccer_twos

paths:
  workspace: ./workspace
  results: ./workspace/results
  configs: ./workspace/configs
  checkpoints: ./workspace/results/checkpoints  # used for baseline only
  logs: ./workspace/results/logs                # used for baseline only
```

**Note:** Non-baseline experiments use a consolidated structure where all artifacts (logs, checkpoints, errors) are stored inside `experiments/<name>/` for easier debugging. The `checkpoints` and `logs` paths above are only used for baseline experiments.

---

## Bring Your Own

Applied Scientist has three extension points. Each is a small Python class or a YAML config change.

### Bring Your Own Task

**The Task Adapter is the only required code you write.** It is the bridge between Applied Scientist (generic) and your specific ML problem. You write it once before launching the system. During a run, agents call the adapter's methods but never modify the adapter itself — Builder edits your task's training code (model definitions, configs), not the adapter file.

Implement a `TaskAdapter` with 5 methods:

```python
from applied_scientist.task import TaskAdapter

class MyTask(TaskAdapter):

    @property
    def name(self) -> str:
        return "2v2 Soccer"

    @property
    def metric(self) -> tuple:
        """(metric_name, direction). Direction: 'higher' or 'lower'."""
        return ("win_rate", "higher")

    @property
    def domain_context(self) -> str:
        """Description given to the Explorer (for literature search)
        and Builder (for implementation). Be specific about frameworks,
        observation/action spaces, and constraints."""
        return """
        2v2 soccer game using Unity ML-Agents and Ray RLlib.
        4 agents (2 per team), discrete action space (27 actions),
        flattened observation vector. Trained on a single GPU.
        """

    @property
    def progress_file(self) -> str:
        """Filename for periodic progress updates during training.
        Your train() must write this file with JSON: {"step": int, "<metric>": float}
        Used by the Builder for early stopping decisions."""
        return "progress.json"

    def train(self, config: dict, seed: int, time_budget: int,
              log_path: str, checkpoint_dir: str) -> dict:
        """Run one training experiment. Return dict of metrics.
        Must include the primary metric. Must write progress_file periodically.
        Return {"status": "crash", "error": "..."} on failure."""
        ...
        return {"win_rate": 0.78, "training_seconds": 1800, "peak_memory_mb": 6200}

    def evaluate(self, checkpoint_dir: str, n_eval_samples: int) -> dict:
        """Evaluate a trained checkpoint. Return dict of metrics."""
        ...
        return {"win_rate": 0.78, "avg_reward": 1.65}

    def get_baseline_spec(self) -> dict | None:
        """Optional: return a custom baseline spec for your task.
        If None, system uses a default PPO baseline."""
        return None
```

Place it in `tasks/my_task/adapter.py`. Point the config at it:

```yaml
task:
  module: tasks/my_task
```

See `tasks/template/` for a skeleton with docstrings. See `tasks/examples/soccer_twos/` for a complete example.

**The domain_context string is important.** It directly controls how well the Explorer finds relevant papers and how well the Builder implements architectures. Be specific:
- Name the framework (PyTorch, TensorFlow, JAX)
- Describe observation and action spaces with dimensions
- Mention hardware constraints (single GPU, memory limits)
- Name the training library (Ray RLlib, Stable-Baselines3, custom)
- State the evaluation protocol (win rate over N episodes, validation loss, etc.)

**The progress_file is required for early stopping.** Your training script must write a JSON file periodically during training:

```python
# Inside your training loop or callback:
progress = {"step": current_step, "win_rate": current_metric, "elapsed_seconds": elapsed}
with open(os.path.join(log_path, "progress.json"), "w") as f:
    json.dump(progress, f)
```

---

### Bring Your Own LLM

**Option A: Built-in backends — just change the config.** No code needed.

```yaml
# Claude (Anthropic)
llm:
  explorer:
    backend: anthropic
    model: claude-haiku-4-5-20251001
  critic:
    backend: anthropic
    model: claude-sonnet-4-6
  builder:
    backend: anthropic
    model: claude-opus-4-6
  orchestrator:
    backend: anthropic
    model: claude-haiku-4-5-20251001

# GPT (OpenAI)
llm:
  explorer:
    backend: openai
    model: gpt-4o-mini
  critic:
    backend: openai
    model: gpt-4o
  builder:
    backend: openai
    model: gpt-4o
  orchestrator:
    backend: openai
    model: gpt-4o-mini

# Gemini (Google)
llm:
  explorer:
    backend: gemini
    model: gemini-2.5-flash
  critic:
    backend: gemini
    model: gemini-2.5-pro
  builder:
    backend: gemini
    model: gemini-2.5-pro
  orchestrator:
    backend: gemini
    model: gemini-2.5-flash

# Mix and match across providers
llm:
  explorer:
    backend: anthropic
    model: claude-haiku-4-5-20251001
  critic:
    backend: openai
    model: gpt-4o
  builder:
    backend: anthropic
    model: claude-opus-4-6
  orchestrator:
    backend: gemini
    model: gemini-2.5-flash

# Local model via Ollama, vLLM, or any OpenAI-compatible API
llm:
  explorer:
    backend: openai_compatible
    model: llama3
    base_url: http://localhost:11434/v1
```

**Option B: Custom backend — implement 2 methods.**

If your LLM provider isn't covered above, write a backend:

```python
from applied_scientist.llm import LLMBackend, LLMMessage, LLMResponse

class MyLLMBackend(LLMBackend):

    def complete(self, messages: list[LLMMessage],
                 tools: list[dict] | None = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0) -> LLMResponse:
        # Call your LLM API
        # Return LLMResponse(content=..., tool_calls=..., usage=..., stop_reason=...)
        ...

    def get_model_id(self) -> str:
        return "my-custom-model"

    @property
    def supports_tool_use(self) -> bool:
        return True   # False if your model doesn't support function calling
```

Register it in the config:

```yaml
llm:
  explorer:
    backend: custom
    module: my_llm_backend.py
    class: MyLLMBackend
    model: my-model-name
```

---

### Bring Your Own Compute

**Option A: Built-in backends — just change the config.**

```yaml
# SLURM cluster
compute:
  backend: slurm
  slurm:
    partition: gpu
    gres: "gpu:1"
    mem: "48G"
    time: "01:00:00"
    setup_commands:
      - "source ~/miniconda3/etc/profile.d/conda.sh"
      - "conda activate my_env"

# Local machine (no cluster)
compute:
  backend: local
```

The SLURM backend automatically:
- Writes a batch script per experiment
- Submits via `sbatch --parsable`
- Monitors via `sacct`
- Cancels via `scancel` on timeout or early stop
- Reads logs from the SLURM output file

The local backend:
- Runs training as a subprocess
- Assigns `CUDA_VISIBLE_DEVICES` per GPU slot
- Monitors via process polling

**Option B: Custom backend — implement 4 methods.**

For cloud providers (AWS Batch, GCP Vertex, Azure ML) or other schedulers:

```python
from applied_scientist.compute import JobRunner, JobStatus

class AWSBatchRunner(JobRunner):

    def submit(self, command: str, job_name: str,
               resources: dict | None = None) -> str:
        """Submit a job. Return a job ID string."""
        ...
        return "aws-batch-job-12345"

    def status(self, job_id: str) -> JobStatus:
        """Check job status. Return JobStatus with state:
        'pending', 'running', 'completed', 'failed', or 'cancelled'."""
        ...
        return JobStatus(job_id=job_id, state="running")

    def cancel(self, job_id: str) -> None:
        """Cancel a running job."""
        ...

    def get_log(self, job_id: str, tail: int = 50) -> str:
        """Return last N lines of job output."""
        ...
        return "..."
```

Register it in the config:

```yaml
compute:
  backend: custom
  custom:
    module: my_cloud_runner.py
    class: AWSBatchRunner
```

---

## Interacting with the System

While the system runs, you have two ways to communicate:

**Talk to the Orchestrator (default):** Type without a prefix. The Orchestrator interprets your natural language and routes commands.

```
> what's the status?
  Explorer: round 2, reading multi-agent competitive game papers
  Builder: 3/4 GPU slots active [mappo: 72%, qmix: 45%, maddpg: 18%]
  Queue: 8 specs pending (top: commnet, tarmac, ppo_lstm)
  Completed: 5 experiments. Best: mappo (win_rate: 0.78)

> cost
  Today: $3.34 (Explorer $0.07, Critic $0.85, Builder $2.41, Orchestrator $0.01)
  Cumulative: $3.34 / $157 budget (day 1)

> prioritize hierarchical RL
  Relaying to Explorer. Critic will boost hierarchical specs in ranking.

> I read a paper about TarMAC, look into it
  Relaying to Explorer: investigate TarMAC (attention-based communication).

> pause
  Builder paused. Running experiments will finish. No new jobs submitted.

> gpus 2
  GPU pool reduced to 2 slots.

> resume
  Builder resumed.
```

**Talk to agents directly:** Prefix with `@agent` to bypass the Orchestrator and send your message directly to a specific agent's LLM.

```
> @explorer focus specifically on graph neural network approaches for
  multi-agent coordination. Check the DGN and G2ANet papers.
  [Explorer]: Searching for DGN (Deep Graph Network) and G2ANet...

> @builder the MAPPO implementation has a bug — the critic should use
  mean aggregation over agent observations, not concatenation. Fix it.
  [Builder]: Reading current implementation... You're right, line 142
  concatenates. Changing to mean aggregation and re-running.

> @critic why did you rank commnet above qmix?
  [Critic]: CommNet scored higher on diversity_bonus — no communication-based
  architecture has been tested, while value decomposition (QMIX's category)
  already has one result.
```

**Built-in commands** (handled directly, no LLM call):

| Command | Action |
|---------|--------|
| `status` | GPU pool, queue depth, recent results |
| `results` | Print results.tsv sorted by primary metric |
| `knowledge` | Print knowledge.md synthesis section |
| `queue` | Print priority queue with scores |
| `cost` | Per-agent token usage and estimated dollar cost |
| `pause` | Stop submitting new experiments |
| `resume` | Resume submitting |
| `gpus N` | Resize GPU pool to N slots |

**Human directive commands** (bypass Critic review, insert directly to queue):

| Command | Action |
|---------|--------|
| `/try <description>` | Quick experiment injection. Parses `key=value` config from description. |
| `/paper <reference>` | Send paper to Explorer for investigation with high priority. |
| `/config <spec> <changes>` | Clone existing spec with modified config (e.g., `/config mappo lr=0.0001`). |
| `/priority <spec_name>` | Boost existing spec to run next (score 100.0). |
| `/load <file.yaml>` | Bulk inject experiments from a YAML file. |

Human directives skip the Critic entirely and insert directly into the priority queue with a high score (100.0), ensuring they run next. All directives are logged as `human_directive` events in `system_events.jsonl`.

**Examples:**

```
> /try PPO with entropy bonus lr=0.001 ent_coef=0.05
Queued: human_ppo_with_entropy_1234 (score: 100.0)

> /paper "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
Sent to Explorer: investigate 'Multi-Agent Actor-Critic...'

> /config mappo lr=0.0001 batch_size=512
Created variant: mappo_variant_5678 with {'lr': 0.0001, 'batch_size': 512}

> /priority qmix
Boosted 'qmix' to score 100.0

> /load experiments_batch.yaml
Loaded 5 specs from experiments_batch.yaml
```

**YAML format for `/load`:**

```yaml
- score: 100.0
  spec:
    name: ppo_baseline
    description: "PPO baseline experiment"
    source_paper: "human directive"
    architecture: {}
    why_it_might_work: "Testing PPO"
    task_config: {}
    training_config: {lr: 0.001, gamma: 0.99}
    resource_estimate: {}
- score: 95.0
  spec:
    name: mappo_variant
    description: "MAPPO with lower learning rate"
    source_paper: "human directive"
    architecture: {}
    why_it_might_work: "Testing lower LR"
    task_config: {}
    training_config: {lr: 0.0001}
    resource_estimate: {}
```

---

## Experiment Lifecycle

Here is exactly what happens when one experiment runs, from start to finish:

```
1. SPEC CREATION
   Explorer reads a paper about MAPPO.
   Explorer writes configs/drafts/mappo.yaml:
     - Architecture: centralized critic seeing all 4 agents' observations
     - Policy: independent per agent, FC [256,256], ReLU
     - Critic: shared, FC [256,256], input = concatenated obs (4 x obs_dim)
     - Why: centralized training enables coordination

2. SPEC REVIEW (async — Explorer moves on immediately)
   Critic picks up review from inbox.
   Round 1: "You don't specify how observations are ordered in concatenation. Fix."
   Critic posts rejection to Explorer inbox.
   Explorer picks up rejection on next loop → revises → resubmits.
   Round 2: "Add a control variant without the centralized critic."
   Explorer writes mappo_no_critic.yaml.
   Round 3: Critic approves both via submit_review tool. Scores: mappo=4.2, mappo_no_critic=3.8.
   Both inserted into priority queue, tagged as a pair.
   Dedup check passes — neither name exists in results or queue.

3. QUEUE RANKING
   Queue state: [mappo: 4.2, mappo_no_critic: 3.8, commnet: 3.5, ...]
   mappo is top-ranked. It gets pulled first when a GPU slot frees up.

4. IMPLEMENTATION
   GPU slot 2 frees up. Builder pulls mappo spec.
   Builder LLM reads the spec (never the paper).
   Builder LLM uses edit_file tool to:
     - Add CentralizedCritic class to the training code
     - Modify the training config to use centralized value function

5. PRE-FLIGHT VALIDATION
   Syntax check: python -c "import ..." → passes
   Smoke test: 10-second training run → completes without crash
   Validation passes → proceed to submission.

6. CODE REVIEW (non-blocking)
   This is a structural change (new model class).
   Builder posts code review request to Critic inbox.
   Builder does NOT wait — proceeds to commit and submit.
   Critic reviews in background: "Implementation matches spec. Approved."
   (If rejected: Builder cancels job, applies fix, resubmits.)

7. GIT COMMIT
   Builder LLM uses git_commit: "experiment: mappo — centralized critic"
     → commit hash: a3f72bc

8. JOB SUBMISSION
   Builder Python code (not LLM) constructs the training command.
   Builder Python code calls job_runner.submit():
     SLURM → writes .sbatch file, runs sbatch → job ID 48291
     Local  → runs subprocess with CUDA_VISIBLE_DEVICES=2 → PID 12345
   GPU slot 2 is marked as running (experiment: mappo, job: 48291).
   Slot state persisted to .state/gpu_slots.json.

9. TRAINING (30 minutes on GPU)
   The training job runs task_adapter.train(config, seed=1, time_budget=1800).
   Training script writes progress.json every N steps:
     {"step": 150000, "win_rate": 0.65, "elapsed_seconds": 900}
   Builder polls progress.json every 10 seconds for early stopping.
   If no improvement after 40% of budget → early stop.
   If exceeds 2x budget → timeout, kill.

10. RESULTS
    Job completes. Builder Python code reads the log.
    Metrics: win_rate=0.78, episode_reward_mean=1.65, peak_memory_mb=6200.
    Appended to results.tsv with commit hash a3f72bc (atomic write).
    Checkpoint saved to experiments/mappo/checkpoints/.
    GPU slot 2 released → immediately filled with next spec from queue.

11. INSIGHT (non-blocking submit)
    Builder LLM writes:
      "MAPPO (0.78) outperformed PPO baseline (0.72 +/- 0.031).
       Used 50% more memory (6.2GB vs 4.1GB) due to centralized critic.
       Hypothesis: global state visibility improves credit assignment."
    Submitted to Critic inbox (Builder moves to next slot).
    Critic reviews via submit_insight_review tool: "Change 'confirms' to 'suggests'. Approved."
    Insight added to knowledge.md (atomic write).

12. PAIRED COMPLETION
    When mappo_no_critic also finishes (win_rate: 0.69):
    Builder writes comparative insight (tracked via pending_pairs):
      "MAPPO 0.78 vs no-critic 0.69 vs baseline 0.72.
       Removing centralized critic drops to baseline level.
       Suggests the critic is the key component."
    Critic reviews, approves. Added to knowledge.md.

13. RE-RANKING
    Critic re-ranks queue with new results.
    Centralized approaches boosted (MAPPO worked).
    Attention-based specs may be deprioritized if no GPU headroom.

14. NEXT EXPERIMENT
    Builder pulls the next top-ranked spec from queue.
    Cycle repeats.
```

---

## Output

After running, the system produces a complete research workspace. Every file below is described with who writes it, who reads it, and what it contains.

### Directory Structure

```
workspace/
├── .git/                                    Git repo (initialized by System on first start)
├── .state/                                  Crash recovery state (auto-managed)
│   ├── gpu_slots.json                       GPU slot assignments
│   └── pending_messages.json                Unprocessed inter-agent messages
│
├── configs/
│   ├── drafts/                              Specs under review or awaiting revision
│   │   ├── attention_ppo.yaml
│   │   └── tarmac_v2.yaml
│   ├── experiments/                         Approved specs (baseline only)
│   │   └── ppo_baseline.yaml
│   └── priority_queue.yaml                  Ranked queue of pending experiments
│
├── experiments/                             Self-contained experiment directories
│   └── mappo/                               EVERYTHING for one experiment
│       ├── spec.yaml                        Copy of experiment spec
│       ├── queue_snapshot.yaml              Queue state when pulled (for debugging)
│       ├── PLAN.md                          Critic-reviewed implementation plan
│       ├── train.py                         Self-contained training script
│       ├── model.py                         Model/policy code
│       ├── run_12346.sh                     Saved sbatch script
│       ├── logs/                            Training output
│       │   ├── slurm-12346.out              Raw stdout/stderr
│       │   ├── progress.json                Live training progress
│       │   └── validation.log               Pre-flight validation output
│       ├── checkpoints/                     Saved model weights
│       └── errors/                          Failure artifacts (if crashed)
│           ├── validation_error.txt         Smoke test failure details
│           ├── runtime_error.txt            Full crash log
│           └── traceback.txt                Extracted stack trace
│
├── results/
│   ├── results.tsv                          Leaderboard: one row per experiment
│   ├── knowledge.md                         Curated insights (synthesis pinned at top)
│   ├── review_log.jsonl                     Every Critic review decision (audit trail)
│   ├── explorer_journal.jsonl               Every paper Explorer read (with summaries)
│   ├── system_events.jsonl                  Timeline of notable system events
│   └── ideas_for_system2.md                 Novel ideas flagged for deeper investigation
│
└── tasks/
    └── examples/
        └── soccer_twos/                     Task source code (edited by Builder)
            ├── adapter.py
            ├── train.py
            └── evaluate.py
```

### File Details

#### Experiment Specs

| File | Writer | Reader | Description |
|------|--------|--------|-------------|
| `configs/drafts/<name>.yaml` | Explorer | Critic | Temporary. Explorer writes a draft spec after reading a paper. Names are auto-deduped (`mappo`, `mappo_v2`, `mappo_v3`). Overwritten on revision, deleted on approval (moved to `experiments/`). |
| `configs/experiments/<name>.yaml` | Critic (moves from `drafts/`) | Builder | Permanent. Only Critic-approved specs land here. Contains full architecture description, hyperparameters, resource estimates. |
| `configs/priority_queue.yaml` | Critic (insert, rerank), Builder (pop) | Builder, Critic, Orchestrator | Ranked queue of approved specs waiting for GPU slots. Atomic write on every mutation. |

#### Results and Knowledge

| File | Writer | Reader | Description |
|------|--------|--------|-------------|
| `results/results.tsv` | Builder | All agents | One row per experiment, including crashes, early stops, and validation failures. Every row links to a git commit hash for reproducibility. |
| `results/knowledge.md` | Builder (Critic-approved insights) | Explorer, Critic, Orchestrator | Curated research findings. Synthesis summaries pinned at top, individual per-experiment insights appended chronologically. Only Critic-approved insights appear here. |
| `results/ideas_for_system2.md` | Critic | You (offline) | Novel architectural ideas that Critic triaged as "major" — too significant for a quick experiment, saved for deeper investigation. |

**results.tsv columns:**
```
name  commit  algorithm  model  metric_value  metric_std  seeds  training_seconds  peak_memory_mb  status  description  extra_metrics
```

Status values: `baseline`, `keep`, `crash`, `early_stop`, `timeout`, `validation_fail`

#### Logs and Audit Trail

| File | Writer | Reader | Description |
|------|--------|--------|-------------|
| `results/review_log.jsonl` | Critic | You (offline) | Every review decision — spec reviews (with round, verdict, feedback, score), insight reviews, code reviews, suggestion triages. Full audit trail of the quality gate. |
| `results/explorer_journal.jsonl` | Explorer | You (offline), System 2 | Every paper Explorer read — title, authors, URL, summary, key insights, limitations, relevance to task, and whether a spec was drafted. Also logs search queries and scope advances. Papers found irrelevant are logged with the reason for dismissal. |
| `results/system_events.jsonl` | All agents, Watchdog | You (offline) | Timeline of notable events: clarification exchanges between agents, alerts, code rejections, experiment completions, early stops, cost threshold crossings, agent crashes and restarts. |

**review_log.jsonl example:**
```json
{"timestamp": "2026-04-03T14:22:01", "type": "spec_review", "spec_name": "mappo", "round": 1, "verdict": "rejected", "feedback": "Missing Q/K/V dimensions"}
{"timestamp": "2026-04-03T14:25:33", "type": "spec_review", "spec_name": "mappo", "round": 2, "verdict": "approved", "score": 4.2}
{"timestamp": "2026-04-03T15:10:00", "type": "insight_review", "experiment": "mappo", "verdict": "approved", "edits": "Changed 'confirms' to 'suggests'"}
{"timestamp": "2026-04-03T15:12:00", "type": "code_review", "spec_name": "qmix", "verdict": "approved"}
{"timestamp": "2026-04-03T16:00:00", "type": "suggestion_triage", "level": "major", "justification": "Novel attention mechanism for agent communication"}
```

**explorer_journal.jsonl example:**
```json
{"timestamp": "...", "action": "search", "query": "multi-agent reinforcement learning soccer", "results_count": 15}
{"timestamp": "...", "action": "read_paper", "title": "The Surprising Effectiveness of PPO in Cooperative MARL", "authors": "Yu et al. 2022", "url": "https://arxiv.org/abs/2103.01955", "relevant": true, "summary": "Shows PPO with parameter sharing matches specialized MARL algorithms", "key_insights": ["Parameter sharing reduces sample complexity", "Centralized value function matters more than policy algorithm"], "limitations": "Tested on cooperative tasks only", "relevance_to_task": "Directly applicable — 2v2 soccer has homogeneous teams", "spec_drafted": "mappo"}
{"timestamp": "...", "action": "read_paper", "title": "Some Single-Agent Paper", "url": "...", "relevant": false, "reason": "No multi-agent component"}
{"timestamp": "...", "action": "scope_advance", "from": 1, "to": 2}
```

**system_events.jsonl example:**
```json
{"timestamp": "...", "event": "experiment_completed", "spec": "mappo", "metric": 0.61}
{"timestamp": "...", "event": "experiment_early_stopped", "spec": "maddpg", "metric": 0.38, "reason": "Below 10% of baseline at 40% budget"}
{"timestamp": "...", "event": "clarification", "from": "builder", "to": "explorer", "spec": "mappo", "question": "Is the value function per-agent or shared?", "answer": "Shared, centralized with global state input"}
{"timestamp": "...", "event": "code_rejected", "spec": "qmix", "feedback": "Wrong API for value decomposition"}
{"timestamp": "...", "event": "alert", "from": "watchdog", "message": "critic crashed. Restarted (1/3)"}
{"timestamp": "...", "event": "cost_alert", "total": 50.23}
```

#### Training Output (per experiment)

All training artifacts are consolidated inside `experiments/<name>/` for easy debugging:

| File | Writer | Reader | Description |
|------|--------|--------|-------------|
| `experiments/<name>/spec.yaml` | Builder | Builder, You | Copy of the experiment spec that was executed. |
| `experiments/<name>/queue_snapshot.yaml` | Builder | You (debugging) | Queue state when this experiment was pulled — shows what else was pending. |
| `experiments/<name>/logs/slurm-<jobid>.out` | Compute backend | Builder, You | Raw stdout/stderr from the training process. |
| `experiments/<name>/logs/progress.json` | Training script | Builder (polls for early stopping) | Written periodically: `{"step": 50000, "win_rate": 0.43, "elapsed_seconds": 600}` |
| `experiments/<name>/logs/validation.log` | Builder | You (debugging) | Pre-flight validation output (syntax check + smoke test). |
| `experiments/<name>/checkpoints/` | Training script | Task adapter's `evaluate()` | Saved model weights. |
| `experiments/<name>/errors/validation_error.txt` | Builder | You (debugging) | Details of smoke test failure (if validation failed). |
| `experiments/<name>/errors/runtime_error.txt` | Builder | You (debugging) | Full crash log (if training crashed). |
| `experiments/<name>/errors/traceback.txt` | Builder | You (debugging) | Extracted Python traceback (if available). |

#### Implementation Code

| File | Writer | Reader | Description |
|------|--------|--------|-------------|
| Task source code (e.g., model definitions) | Builder (via `edit_file`, `write_file` tools) | Training script, Critic (reviews diff) | Builder edits code in-place in the task's source tree. Every edit is git committed before job submission. |
| Git history | Builder (`git commit` before each run) | Anyone (`git log`, `git checkout`) | Full history of every code change, linked to experiments via commit hash in `results.tsv`. Checkout any hash to reproduce an experiment. |

#### System State (crash recovery)

| File | Writer | Reader | Description |
|------|--------|--------|-------------|
| `.state/gpu_slots.json` | GPUPool (auto-persists on every assign/release) | System on restart | Current GPU slot assignments. On restart, System reconciles this with actual job status from the compute backend. |
| `.state/pending_messages.json` | MessageBus (auto-persists on every post) | System on restart | Unprocessed inter-agent messages. Reloaded into agent inboxes on restart so no messages are lost. |
| `configs/drafts/*.yaml` | Explorer | System on restart, Critic | Unreviewed draft specs. On restart, the system scans this directory and re-injects any drafts not yet in `configs/experiments/` into the Critic's inbox for review. |

### Who Writes What — Summary

| Agent | Writes | Reads |
|-------|--------|-------|
| **Explorer** | `configs/drafts/*.yaml`, `results/explorer_journal.jsonl` | `results/results.tsv`, `results/knowledge.md` (synthesis), `configs/drafts/` + `configs/experiments/` (name dedup) |
| **Critic** | Moves `drafts/` → `configs/experiments/`, inserts into `priority_queue.yaml`, appends to `results/review_log.jsonl`, appends to `results/ideas_for_system2.md` | `configs/drafts/*.yaml`, `results/results.tsv`, `results/knowledge.md`, `priority_queue.yaml` |
| **Builder** | Creates `experiments/<name>/` directories (spec.yaml, queue_snapshot.yaml, PLAN.md, train.py, model code, logs/, checkpoints/, errors/), `git commit`, appends to `results/results.tsv`, writes to `results/knowledge.md` (approved insights only) | `configs/experiments/*.yaml`, `experiments/<name>/spec.yaml`, `priority_queue.yaml` (pop), `results/results.tsv`, `experiments/<name>/logs/progress.json`, baseline `tasks/.../train.py` (as reference) |
| **Orchestrator** | Nothing persistent | Everything (read-only, for status display) |
| **Training script** | `experiments/<name>/logs/progress.json`, `experiments/<name>/checkpoints/`, stdout/stderr | Task source code |
| **System** | `.state/gpu_slots.json`, `.state/pending_messages.json`, `results/system_events.jsonl`, `git init` | Config YAML, all state files on recovery |

### Reproducibility

Every experiment is reproducible from its **git commit hash** (in `results.tsv`) + **experiment spec** (in `configs/experiments/`). Checkout the commit, load the spec, run the training command with the same seed.

---

## Architecture

### Agents and Tools

| Tool | Orchestrator | Explorer | Critic | Builder |
|------|:---:|:---:|:---:|:---:|
| Web search | | x | | |
| Paper/PDF reader | | x | | |
| Read files | x | x | x | x |
| Write files | | x | x | x |
| Edit code | | | | x |
| Shell execution (sandboxed) | | | | x |
| Git operations | | | | x |
| Job submission | | | | x |
| Structured review tools | | | x | |

### Non-Blocking Review Flow

No agent ever blocks waiting for another. All inter-agent communication uses fire-and-forget posting with inbox polling:

```
Explorer writes spec → posts to Critic inbox → searches next paper
                                 ↓
                           Critic inbox (priority-ordered)
                                 ↓
                        Critic reviews, scores
                                 ↓
                   Posts result to Explorer inbox
                                 ↓
          Explorer picks up on next loop iteration
```

Critic processes its inbox in priority order: code reviews > spec reviews > insight reviews > suggestions > re-ranks.

### Priority Queue Ranking

```
rank = expected_performance_gain
     + diversity_bonus        (no similar architecture tested yet)
     + information_bonus      (tests an untested category)
```

Deduplication: specs are rejected if the experiment name already exists in results.tsv or the priority queue.

Re-ranked after every experiment completes (not in batches).

### GPU Pool Model

Each GPU slot operates independently. When any slot frees up, the Builder immediately fills it with the next top-ranked spec. No GPU waits for any other. Throughput scales linearly with GPU count.

```
GPU0: [baseline_s1][baseline_s2][baseline_s3][exp4   ][exp8     ]
GPU1: [exp1              ][exp5         ][exp9              ]
GPU2: [exp2      ][exp3       ][exp6  ][exp10     ]
GPU3: [exp7       ][crash][exp11  ][exp12   ][exp13        ]
```

Baseline seeds run sequentially on one slot, leaving other GPUs free for experiments.

### Knowledge Base

`knowledge.md` is structured with **synthesis summaries pinned at the top** (updated as experiment categories complete) and **individual experiment insights** below. Every insight is reviewed by the Critic before entry. The Explorer reads the knowledge base to adapt its literature search.

### Robustness

- **Crash recovery:** GPU slot assignments and pending messages are persisted to `.state/`. On restart, the system reconciles with `sacct` or process state. Unreviewed draft specs in `configs/drafts/` are automatically re-injected into the Critic's inbox so no work is lost between runs.
- **Watchdog:** Monitors agent threads; restarts crashed threads up to 3 times.
- **Atomic writes:** All shared state files use atomic write-then-rename to prevent corruption.
- **Context management:** Agent conversations are automatically trimmed when approaching context limits.
- **Shell sandboxing:** Builder's shell commands are restricted to workspace, with dangerous patterns blocked.
- **Pre-flight validation:** Syntax check + smoke test before every GPU submission prevents wasting GPU time on broken code.

### Reproducibility

- Git commit before every training run; hash recorded in results
- Config YAML saved per experiment
- Baselines run with multiple seeds (mean +/- std)
- Early stopping at configurable fraction of time budget via progress file monitoring

### Pluggable Design

```
                    applied_scientist/
                    ├── task/base.py         TaskAdapter    (you implement)
                    ├── llm/base.py          LLMBackend     (built-in or custom)
                    ├── compute/base.py      JobRunner      (built-in or custom)
                    └── tools/base.py        Tool           (built-in, extensible)

Everything above the base classes is framework code that doesn't change.
Everything below is user-provided.
```

---

## Deployment

**Local (single machine):**
```bash
python -m applied_scientist \
  --config configs/my_config.yaml \
  --job-runner local \
  --n-gpus 1
```

**SLURM cluster:**
```bash
# 1. Request interactive CPU allocation for the agent loop
salloc --nodes=1 --ntasks=4 --mem=16G --time=14:00:00

# 2. Start in tmux for persistence across SSH disconnects
tmux new-session -s applied-scientist

# 3. Set API key and run
export ANTHROPIC_API_KEY=sk-ant-...
python -m applied_scientist \
  --config configs/my_config.yaml \
  --job-runner slurm \
  --n-gpus 4

# Detach: Ctrl-B d
# Reattach later: tmux attach -t applied-scientist
```

**Cloud:** Implement a `JobRunner` for your provider (see [Bring Your Own Compute](#bring-your-own-compute)). The agent loop runs on a small CPU instance; training jobs dispatch to GPU instances.

---

## Cost Estimates

LLM API costs depend on which models you choose. Example with Claude:

| Agent | Recommended Model | Daily Cost |
|-------|-------------------|------------|
| Explorer | Haiku | ~$0.26 |
| Critic | Sonnet | ~$2.25 |
| Builder | Sonnet | ~$4.95 |
| Orchestrator | Haiku | ~$0.03 |
| **Total** | | **~$7.50/day** |

**Budget options:**

| Configuration | Daily | 21 days |
|---|---|---|
| All Haiku (cheapest) | ~$2.60 | ~$55 |
| Haiku + Sonnet (recommended) | ~$7.50 | ~$157 |
| Opus for Critic (highest quality) | ~$17.00 | ~$357 |
| GPT-4o for all | ~$9.00 | ~$189 |

Mix and match across providers to optimize cost/quality per agent. Use the `cost` command while running to monitor spend in real-time.

Set `cost_alert_threshold` in config to get alerts when cumulative cost exceeds a threshold.

---

## Related Work

| System | Scope | Task-agnostic | LLM-agnostic | Multi-agent | Quality gate |
|--------|-------|:---:|:---:|:---:|:---:|
| [autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026) | LLM training | No | No | No (single agent) | No |
| [The AI Scientist](https://github.com/SakanaAI/AI-Scientist) (Lu et al., 2024) | Paper generation | Partial | Partial | No | No |
| [AIDE](https://github.com/WecoAI/aideml) (Weco AI) | ML engineering | Yes | Partial | No | No |
| **Applied Scientist** | Benchmark sweep | **Yes** | **Yes** | **Yes (4 agents)** | **Yes (Critic)** |

Key differentiators:
- **Dedicated Critic agent** that reviews specs, code, and insights before they enter the knowledge base, using structured tool calls for reliable decisions
- **Non-blocking concurrency** — no agent ever waits for another, maximizing throughput
- **GPU pool model** that maximizes hardware utilization with no idle waiting
- **Pre-flight validation** preventing wasted GPU time on broken code
- **Early stopping via progress file** — kills experiments that show no learning signal
- **Crash recovery** with persisted state and watchdog thread supervision
- **Continuous Explorer** with automatic scope advancement and priority-ranked experiment queue
- **Task adapter interface** that cleanly separates the framework from the domain
- **Knowledge accumulation** across experiments with synthesis and insight curation

---

## Citation

```bibtex
@software{applied_scientist_2026,
  title={Applied Scientist: Autonomous Multi-Agent System for ML Experiment Automation},
  author={},
  year={2026},
  url={https://github.com/pheeeeee/applied-scientist}
}
```

## License

MIT
