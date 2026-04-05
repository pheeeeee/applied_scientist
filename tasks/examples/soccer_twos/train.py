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

    print(json.dumps(last_result))


if __name__ == "__main__":
    main()
