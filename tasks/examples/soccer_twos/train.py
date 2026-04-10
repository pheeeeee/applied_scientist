"""Soccer 2v2 training script. Wraps Ray RLlib PPO with progress reporting.

Called by the task adapter:
    python train.py --config '{"lr": 0.0003, ...}' --seed 42 --time-budget 1800 \
                    --log-path /path/to/logs --checkpoint-dir /path/to/ckpt

Writes progress.json periodically (required for early stopping).
Prints final result as JSON on last line (parsed by adapter.train() /
Builder).

The script wraps the `soccer_twos` Unity ML-Agents environment as a
`ray.rllib.env.MultiAgentEnv` subclass and registers it with RLlib's
environment registry. A single shared policy is used across all four
agents (standard parameter-sharing baseline for cooperative MARL).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import gym
import numpy as np
import ray

# --- Monkey-patch for Ray 1.4 driver-side GPU detection bug -----------------
#
# In Ray 1.4.0's torch_policy.py (TorchPolicy.__init__), the GPU branch
# builds `self.devices` by iterating over `ray.get_gpu_ids()`:
#
#     gpu_ids = ray.get_gpu_ids()
#     self.devices = [torch.device(f"cuda:{i}") for i, id_ in enumerate(gpu_ids)
#                     if i < config["num_gpus"]]
#     self.device = self.devices[0]   # IndexError if gpu_ids is empty
#
# On the driver process (local worker), `ray.get_gpu_ids()` returns [] even
# when CUDA_VISIBLE_DEVICES is set and torch.cuda.is_available() is True,
# because the driver isn't a Ray actor with an explicit GPU allocation.
# The result is `self.devices = []` and an IndexError on `self.devices[0]`.
#
# Fix: replace `ray.get_gpu_ids` with a wrapper that falls back to the
# physical torch-visible GPUs when Ray's own accounting is empty.
# This is scoped to the training subprocess only (train.py runs in its
# own Python process), so there's no contamination of other Ray users.
_ray_original_get_gpu_ids = ray.get_gpu_ids


def _patched_get_gpu_ids():
    ids = _ray_original_get_gpu_ids()
    if not ids:
        import torch as _torch_for_gpu_check
        if _torch_for_gpu_check.cuda.is_available():
            return list(range(_torch_for_gpu_check.cuda.device_count()))
    return ids


ray.get_gpu_ids = _patched_get_gpu_ids
# ----------------------------------------------------------------------------

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env


# ----- Environment wrapper ---------------------------------------------------

class RLLibMultiAgentWrapper(MultiAgentEnv):
    """Wrap a soccer_twos multiagent_player env to inherit from RLlib's
    MultiAgentEnv.

    soccer_twos.make(multiagent_player) returns an env whose step() already
    yields (obs_dict, rew_dict, done_dict_with___all__, info_dict) —
    exactly the MultiAgentEnv contract — but doesn't inherit from
    MultiAgentEnv. RLlib's RolloutWorker does an isinstance check when
    multiple policies are configured, so we need a real subclass.
    """

    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action_dict):
        return self._env.step(action_dict)

    def close(self):
        self._env.close()

    def render(self, mode="human"):
        return self._env.render(mode)


def create_env(env_context):
    """Create a soccer_twos env for an RLlib rollout worker.

    env_context is an RLlib EnvContext — worker_index / vector_index are
    ATTRIBUTES on the object (not dict keys). We use them to assign each
    Unity subprocess a unique worker_id so ports don't collide.
    """
    import soccer_twos

    worker_index = getattr(env_context, "worker_index", 0)
    vector_index = getattr(env_context, "vector_index", 0)
    num_envs_per_worker = env_context.get("num_envs_per_worker", 1) if hasattr(
        env_context, "get"
    ) else 1
    unique_id = worker_index * num_envs_per_worker + vector_index

    env = soccer_twos.make(
        variation=soccer_twos.EnvType.multiagent_player,
        flatten_branched=True,
        worker_id=unique_id,
        render=False,
    )
    return RLLibMultiAgentWrapper(env)


# ----- Progress / result helpers --------------------------------------------

def write_progress(result, log_path):
    progress = {
        "step": result.get("timesteps_total", 0),
        "win_rate": result.get("custom_metrics", {}).get("win_rate", 0),
        "episode_reward_mean": result.get("episode_reward_mean", 0.0),
        "elapsed_seconds": result.get("time_total_s", 0.0),
    }
    with open(os.path.join(log_path, "progress.json"), "w") as f:
        json.dump(progress, f)


# ----- Main ------------------------------------------------------------------

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

    # Detect GPU and tell Ray about it — without this, ray.get_gpu_ids()
    # returns empty inside the torch policy's __init__ and self.devices
    # ends up as an empty list, causing an IndexError on self.devices[0].
    import torch as _torch
    n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    ray.init(
        num_gpus=n_gpus,
        num_cpus=config.get("num_cpus", 4),
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
    )

    # Register the env under a gym-style name. We use "_v0" suffix to
    # satisfy gym's `^.+-v\d+$` format in case RLlib falls back to gym.make.
    register_env("soccer_twos_v0", create_env)

    # Observation and action spaces for the multiagent policy spec.
    # soccer_twos flatten_branched=True yields a 336-dim flat obs and
    # 27-action discrete space (3 × 3 × 3 flattened).
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32
    )
    act_space = gym.spaces.Discrete(27)

    rllib_config = {
        "env": "soccer_twos_v0",
        "env_config": {
            "num_envs_per_worker": 1,
        },
        "framework": "torch",
        "seed": args.seed,

        # Single shared policy across all four agents (standard
        # parameter-sharing baseline for cooperative MARL).
        "multiagent": {
            "policies": {
                "shared_policy": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "shared_policy",
            "policies_to_train": ["shared_policy"],
        },

        # PPO hyperparameters (overridable via --config JSON).
        "lr": config.get("lr", 0.0003),
        "train_batch_size": config.get("train_batch_size", 4096),
        "sgd_minibatch_size": config.get("sgd_minibatch_size", 256),
        "num_sgd_iter": config.get("num_sgd_iter", 20),
        "rollout_fragment_length": config.get("rollout_fragment_length", 512),
        "gamma": config.get("gamma", 0.99),
        "lambda": config.get("lambda", 0.95),
        "clip_param": config.get("clip_param", 0.2),
        "vf_loss_coeff": config.get("vf_loss_coeff", 0.5),
        "entropy_coeff": config.get("entropy_coeff", 0.01),
        "grad_clip": config.get("grad_clip", 0.5),

        # Default FC policy — two hidden layers, 256 units each.
        "model": {
            "fcnet_hiddens": config.get("fcnet_hiddens", [256, 256]),
            "fcnet_activation": "relu",
        },

        # Workers — Unity subprocess can't be vectorized. Default to 0
        # so the driver runs rollouts directly.
        "num_workers": config.get("num_workers", 0),
        "num_envs_per_worker": 1,
        # num_gpus=1 is safe now because we monkey-patched ray.get_gpu_ids
        # at the top of this file. Without that patch, Ray 1.4's
        # TorchPolicy.__init__ builds `self.devices = []` on the driver
        # and crashes at `self.device = self.devices[0]`.
        "num_gpus": config.get("num_gpus", 1),

        # Misc
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "normalize_actions": False,
        "log_level": "INFO",
    }

    trainer = PPOTrainer(config=rllib_config)

    start_time = time.time()
    best_win_rate = 0.0
    last_result = {"status": "crash", "error": "no iterations completed"}

    try:
        while time.time() - start_time < args.time_budget:
            result = trainer.train()
            write_progress(result, args.log_path)

            win_rate = result.get("custom_metrics", {}).get("win_rate", 0)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                try:
                    trainer.save(args.checkpoint_dir)
                except Exception as save_err:  # checkpoint errors shouldn't kill training
                    print(f"[train.py] checkpoint save failed: {save_err}", file=sys.stderr)

            last_result = {
                "win_rate": win_rate,
                "best_win_rate": best_win_rate,
                "episode_reward_mean": result.get("episode_reward_mean", 0.0),
                "timesteps_total": result.get("timesteps_total", 0),
                "elapsed_seconds": time.time() - start_time,
                "status": "completed",
            }
    except Exception as e:
        last_result = {
            "status": "crash",
            "error": f"{type(e).__name__}: {e}",
            "elapsed_seconds": time.time() - start_time,
        }

    try:
        trainer.stop()
    except Exception:
        pass
    ray.shutdown()

    # Contract: last line on stdout MUST be JSON of the final result.
    print(json.dumps(last_result))


if __name__ == "__main__":
    main()
