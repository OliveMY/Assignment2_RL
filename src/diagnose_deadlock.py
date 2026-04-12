"""
Diagnose the deadlock bug where both agents freeze in the demo.

Runs many episodes, tracking per-step observations and actions.
Flags episodes where the model outputs [0, 0] (no movement) and
analyzes what observation patterns trigger the deadlock.

Usage:
    python src/diagnose_deadlock.py --env medium --rounds 200
    python src/diagnose_deadlock.py --env simple --rounds 200
"""
import argparse
import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
from env_wrapper import make_env

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def diagnose(env_name: str, model_path: str, num_rounds: int):
    print(f"\n{'='*60}")
    print(f"DEADLOCK DIAGNOSIS: {env_name.upper()}")
    print(f"Model: {model_path}")
    print(f"Rounds: {num_rounds}")
    print(f"{'='*60}\n")

    env = make_env(env_name=env_name, time_scale=20.0, no_graphics=True)
    model = PPO.load(model_path)

    # Per-episode stats
    episode_stats = []

    for ep in range(num_rounds):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        no_move_steps = 0       # steps where model chose [0, 0]
        initial_obs = obs.copy()
        first_action = None
        actions_taken = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_taken.append(action.copy())

            if first_action is None:
                first_action = action.copy()

            # Check if model chose "no movement" (action 0 on both branches)
            if action[0] == 0 and action[1] == 0:
                no_move_steps += 1

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

        # Classify outcome
        if episode_reward > 0:
            outcome = "WIN"
        elif episode_reward < 0:
            outcome = "LOSS"
        else:
            outcome = "DRAW"

        # Detect deadlock: high ratio of no-move steps
        no_move_ratio = no_move_steps / max(steps, 1)
        is_deadlock = no_move_ratio > 0.8 and outcome == "DRAW"

        stat = {
            "episode": ep + 1,
            "outcome": outcome,
            "reward": episode_reward,
            "steps": steps,
            "no_move_steps": no_move_steps,
            "no_move_ratio": no_move_ratio,
            "is_deadlock": is_deadlock,
            "initial_obs": initial_obs,
            "first_action": first_action,
        }
        episode_stats.append(stat)

        if is_deadlock:
            print(f"  EP {ep+1:>4}: *** DEADLOCK *** steps={steps}, "
                  f"no_move={no_move_steps}/{steps} ({no_move_ratio:.0%})")
            print(f"          Initial obs: {initial_obs.round(3)}")
            print(f"          First action: {first_action}")
            # Show unique actions taken
            unique, counts = np.unique(
                np.array(actions_taken), axis=0, return_counts=True)
            for a, c in zip(unique, counts):
                print(f"          Action {a}: {c}x ({c/steps:.0%})")
        elif ep % 50 == 49:
            # Progress update every 50 episodes
            recent = episode_stats[-50:]
            wins = sum(1 for s in recent if s["outcome"] == "WIN")
            deadlocks = sum(1 for s in recent if s["is_deadlock"])
            print(f"  EP {ep+1:>4}: last 50 → {wins} wins, "
                  f"{deadlocks} deadlocks")

    env.close()

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total = len(episode_stats)
    wins = sum(1 for s in episode_stats if s["outcome"] == "WIN")
    losses = sum(1 for s in episode_stats if s["outcome"] == "LOSS")
    draws = sum(1 for s in episode_stats if s["outcome"] == "DRAW")
    deadlocks = sum(1 for s in episode_stats if s["is_deadlock"])

    print(f"  Wins:      {wins}/{total} ({wins/total*100:.1f}%)")
    print(f"  Losses:    {losses}/{total} ({losses/total*100:.1f}%)")
    print(f"  Draws:     {draws}/{total} ({draws/total*100:.1f}%)")
    print(f"  Deadlocks: {deadlocks}/{total} ({deadlocks/total*100:.1f}%)")

    # Analyze deadlock episodes
    dl_episodes = [s for s in episode_stats if s["is_deadlock"]]
    if dl_episodes:
        print(f"\n--- Deadlock Analysis ({len(dl_episodes)} episodes) ---")

        # Analyze initial observations in deadlock episodes
        dl_obs = np.array([s["initial_obs"] for s in dl_episodes])
        print(f"\n  Initial observation stats (deadlock episodes):")
        labels = [
            "self_pos_x", "self_pos_z",
            "opp_pos_x", "opp_pos_z",
            "self_vel_x", "self_vel_z",
            "opp_vel_x", "opp_vel_z",
            "self_angvel_x", "self_angvel_z",
            "opp_angvel_x", "opp_angvel_z",
        ]
        for i, label in enumerate(labels):
            vals = dl_obs[:, i]
            print(f"    {label:>15}: mean={vals.mean():+.3f}  "
                  f"std={vals.std():.3f}  "
                  f"range=[{vals.min():+.3f}, {vals.max():+.3f}]")

        # Compare with non-deadlock episodes
        ndl_episodes = [s for s in episode_stats if not s["is_deadlock"]]
        if ndl_episodes:
            ndl_obs = np.array([s["initial_obs"] for s in ndl_episodes])
            print(f"\n  Initial observation stats (normal episodes):")
            for i, label in enumerate(labels):
                vals = ndl_obs[:, i]
                print(f"    {label:>15}: mean={vals.mean():+.3f}  "
                      f"std={vals.std():.3f}  "
                      f"range=[{vals.min():+.3f}, {vals.max():+.3f}]")

        # Check what the model outputs for typical deadlock observations
        print(f"\n  First actions in deadlock episodes:")
        dl_actions = np.array([s["first_action"] for s in dl_episodes])
        unique, counts = np.unique(dl_actions, axis=0, return_counts=True)
        for a, c in sorted(zip(unique.tolist(), counts.tolist()),
                           key=lambda x: -x[1]):
            print(f"    Action {a}: {c}x ({c/len(dl_episodes):.0%})")

        # Analyze step counts
        dl_steps = [s["steps"] for s in dl_episodes]
        print(f"\n  Deadlock episode lengths: "
              f"mean={np.mean(dl_steps):.0f}, "
              f"min={min(dl_steps)}, max={max(dl_steps)}")
    else:
        print(f"\n  No deadlocks detected in {total} episodes!")

    # No-move analysis across all episodes
    all_no_move = [s["no_move_ratio"] for s in episode_stats]
    print(f"\n--- No-Move Ratio Distribution ---")
    for threshold in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        count = sum(1 for r in all_no_move if r >= threshold)
        print(f"  >= {threshold:.0%} no-move: {count}/{total} episodes")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose deadlock bug in trained agents")
    parser.add_argument("--env", type=str, required=True,
                        choices=["simple", "medium", "hard"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=200)
    args = parser.parse_args()

    model_path = args.model or os.path.join(
        BASE_DIR, "models", args.env, "final.zip")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    diagnose(args.env, model_path, args.rounds)


if __name__ == "__main__":
    main()
