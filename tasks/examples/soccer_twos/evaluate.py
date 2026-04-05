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
