"""
Evaluate trained PPO agents and run live demos.

Usage:
    # Fast evaluation (no graphics)
    python src/evaluate.py --env simple --model models/simple/final.zip --rounds 100

    # Live demo mode (with graphics, real-time)
    python src/evaluate.py --env simple --model models/simple/final.zip --rounds 10 --demo

    # Evaluate all environments
    python src/evaluate.py --env all --rounds 10
"""
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from env_wrapper import make_env

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(env_name: str, model_path: str, num_rounds: int = 10,
             demo: bool = False, verbose: bool = True):
    """Evaluate a trained model on an environment.

    Args:
        env_name: "simple", "medium", or "hard"
        model_path: Path to saved .zip model
        num_rounds: Number of episodes to play
        demo: If True, run with graphics at real-time speed
        verbose: Print per-round results

    Returns:
        dict with win/loss/draw counts and average reward
    """
    time_scale = 1.0 if demo else 20.0
    no_graphics = not demo

    if verbose:
        mode = "DEMO (real-time)" if demo else "EVAL (fast)"
        print(f"\n{'='*50}")
        print(f"Evaluating: {env_name.upper()} | Mode: {mode}")
        print(f"Model: {model_path}")
        print(f"Rounds: {num_rounds}")
        print(f"{'='*50}")

    env = make_env(
        env_name=env_name,
        time_scale=time_scale,
        no_graphics=no_graphics,
    )
    model = PPO.load(model_path)

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

        if verbose:
            print(f"  Round {i+1:>3}/{num_rounds}: {outcome:>4} "
                  f"(reward={episode_reward:+.3f})")

    env.close()

    # Results
    results = {
        "env_name": env_name,
        "model_path": model_path,
        "num_rounds": num_rounds,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_rounds,
        "avg_reward": float(np.mean(rewards)),
        "projected_score": wins * 0.6,  # 0.6 pts per win
    }

    if verbose:
        print(f"\n--- Results ---")
        print(f"  Wins:   {wins}/{num_rounds} ({100*wins/num_rounds:.1f}%)")
        print(f"  Losses: {losses}/{num_rounds}")
        print(f"  Draws:  {draws}/{num_rounds}")
        print(f"  Avg reward: {results['avg_reward']:+.3f}")
        print(f"  Projected demo score: {results['projected_score']:.1f} / 6.0 pts")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agents")
    parser.add_argument("--env", type=str, required=True,
                        choices=["simple", "medium", "hard", "all"])
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .zip (default: models/{env}/final.zip)")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (graphics, real-time)")
    args = parser.parse_args()

    envs = ["simple", "medium", "hard"] if args.env == "all" else [args.env]
    all_results = []

    for env_name in envs:
        model_path = args.model or os.path.join(BASE_DIR, "models", env_name, "final.zip")
        if not os.path.exists(model_path):
            print(f"\nSkipping {env_name}: model not found at {model_path}")
            continue
        results = evaluate(env_name, model_path, args.rounds, args.demo)
        all_results.append(results)

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"{'Env':<10} {'Win%':<8} {'Wins':<8} {'Score':<10}")
        total_score = 0
        for r in all_results:
            print(f"{r['env_name']:<10} {100*r['win_rate']:<8.1f} "
                  f"{r['wins']}/{r['num_rounds']:<5} {r['projected_score']:.1f}/6.0")
            total_score += r['projected_score']
        print(f"{'TOTAL':<10} {'':8} {'':8} {total_score:.1f}/18.0")


if __name__ == "__main__":
    main()
