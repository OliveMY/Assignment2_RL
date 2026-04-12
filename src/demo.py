"""
Live demo: show trained PPO agents playing the Unity ball game.

Runs 10 games per environment (Simple → Medium → Hard) with graphics
at real-time speed. Displays a summary after each environment.

Usage:
    python src/demo.py
    python src/demo.py --rounds 5
    python src/demo.py --env medium
    python src/demo.py --env all --rounds 10
"""
import argparse
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from env_wrapper import make_env

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_demo(env_name: str, model_path: str, num_rounds: int):
    """Run a live demo for one environment."""
    print(f"\n{'='*60}")
    print(f"  DEMO: {env_name.upper()} Environment")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Rounds: {num_rounds}")
    print(f"{'='*60}\n")

    print("  Starting Unity environment (with graphics)...")
    env = make_env(
        env_name=env_name,
        time_scale=1.0,       # real-time speed
        no_graphics=False,    # show the game window
    )
    model = PPO.load(model_path)

    # Brief pause for Unity window to appear
    time.sleep(2)

    wins = 0
    losses = 0
    draws = 0
    rewards = []

    for i in range(num_rounds):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

        if episode_reward > 0:
            wins += 1
            outcome = "WIN"
        elif episode_reward < 0:
            losses += 1
            outcome = "LOSS"
        else:
            draws += 1
            outcome = "DRAW"

        print(f"  Round {i+1:>2}/{num_rounds}: {outcome:>4}  "
              f"(reward={episode_reward:+.3f})  "
              f"[W:{wins} L:{losses} D:{draws}]")

    env.close()

    win_rate = wins / num_rounds * 100
    avg_reward = np.mean(rewards)

    print(f"\n  --- {env_name.upper()} Results ---")
    print(f"  Win rate: {wins}/{num_rounds} ({win_rate:.0f}%)")
    print(f"  Avg reward: {avg_reward:+.3f}")
    print()

    return {
        "env": env_name,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Live demo of trained PPO agents")
    parser.add_argument("--env", type=str, default="all",
                        choices=["simple", "medium", "hard", "all"],
                        help="Which environment to demo (default: all)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Games per environment (default: 10)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing model .zip files "
                             "(default: models/{env}/final.zip)")
    args = parser.parse_args()

    envs = ["simple", "medium", "hard"] if args.env == "all" else [args.env]

    print("\n" + "=" * 60)
    print("  PPO Agent Demo — Unity Ball Game")
    print("  Environments:", ", ".join(e.upper() for e in envs))
    print("  Rounds per env:", args.rounds)
    print("=" * 60)

    all_results = []

    for env_name in envs:
        if args.model_dir:
            model_path = os.path.join(args.model_dir, env_name, "final.zip")
        else:
            model_path = os.path.join(BASE_DIR, "models", env_name, "final.zip")

        if not os.path.exists(model_path):
            print(f"\n  Skipping {env_name}: model not found at {model_path}")
            continue

        result = run_demo(env_name, model_path, args.rounds)
        all_results.append(result)

        # Pause between environments so the audience can see the transition
        if env_name != envs[-1]:
            print("  Next environment starting in 3 seconds...")
            time.sleep(3)

    # Final summary
    if len(all_results) > 1:
        print("=" * 60)
        print("  OVERALL SUMMARY")
        print("=" * 60)
        print(f"  {'Env':<10} {'Win Rate':<12} {'Wins':<10} {'Avg Reward'}")
        print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
        for r in all_results:
            print(f"  {r['env'].upper():<10} {r['win_rate']:>5.0f}%       "
                  f"{r['wins']}/{r['wins']+r['losses']+r['draws']:<7} "
                  f"{r['avg_reward']:+.3f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
