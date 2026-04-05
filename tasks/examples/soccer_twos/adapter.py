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
        """Delegates to train.py which wraps Ray Tune."""
        cmd = (
            f"python {os.path.dirname(__file__)}/train.py "
            f"--config '{json.dumps(config)}' "
            f"--seed {seed} "
            f"--time-budget {time_budget} "
            f"--log-path {log_path} "
            f"--checkpoint-dir {checkpoint_dir}"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        try:
            return json.loads(result.stdout.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            return {"status": "crash", "error": result.stderr[-500:]}

    def evaluate(self, checkpoint_dir, n_eval_samples):
        cmd = (
            f"python {os.path.dirname(__file__)}/evaluate.py "
            f"--checkpoint-dir {checkpoint_dir} "
            f"--n-episodes {n_eval_samples}"
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        try:
            return json.loads(result.stdout.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            return {"error": result.stderr[-500:]}

    def get_baseline_spec(self):
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
        return {"win_rate": 0.50, "avg_reward": 0.0}

    def get_tunable_defaults(self):
        return {
            "lr": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
            "train_batch_size": {"type": "factor", "factors": [0.5, 1, 2, 4]},
            "gamma": {"type": "discrete", "values": [0.99, 0.995, 0.999]},
            "entropy_coeff": {"type": "log_uniform", "low": 0.001, "high": 0.1},
        }
