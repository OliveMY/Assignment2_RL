"""
Hyperparameter grid sweep for PPO across Unity ML-Agents environments.

Usage:
    python src/sweep.py                    # Full sweep (all envs, all combos)
    python src/sweep.py --env simple       # Sweep on one environment only
    python src/sweep.py --dry-run          # Print all configurations without training
"""
import argparse
import itertools
import json
import os
import time
import traceback

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from config import get_config, get_ppo_kwargs
from env_wrapper import make_env
from callbacks import TrainingLogCallback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(BASE_DIR, "results", "sweep")

# === SWEEP GRID ===
SWEEP_PARAMS = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "ent_coef": [0.001, 0.01, 0.05],
    "n_steps": [512, 2048, 4096],
}

TIMESTEPS = {
    "simple": 500_000,
    "medium": 1_000_000,
    "hard": 2_000_000,
}


def make_grid():
    """Generate all hyperparameter combinations as list of dicts."""
    keys = sorted(SWEEP_PARAMS.keys())
    combos = list(itertools.product(*(SWEEP_PARAMS[k] for k in keys)))
    return [dict(zip(keys, vals)) for vals in combos]


def run_name(idx, params):
    """Generate a human-readable directory name for one run."""
    lr = params["learning_rate"]
    ent = params["ent_coef"]
    ns = params["n_steps"]
    return f"run_{idx:02d}_lr{lr:.0e}_ent{ent:.3f}_ns{ns}"


def run_single_experiment(env_name, params, run_idx, worker_id=0):
    """Train one PPO run with specific hyperparameters. Returns summary dict."""
    config = get_config(env_name)
    config.update(params)

    total_timesteps = TIMESTEPS[env_name]
    time_scale = config.pop("time_scale", 20.0)
    no_graphics = config.pop("no_graphics", True)
    config.pop("total_timesteps", None)

    name = run_name(run_idx, params)
    log_dir = os.path.join(SWEEP_DIR, env_name, name)
    model_dir = os.path.join(log_dir, "model")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(env_name, time_scale=time_scale,
                   no_graphics=no_graphics, worker_id=worker_id)
    env = Monitor(env)

    ppo_kwargs = get_ppo_kwargs(config)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=None, **ppo_kwargs)

    callbacks = [
        TrainingLogCallback(
            log_dir=log_dir,
            config={**config, "total_timesteps": total_timesteps},
            env_name=env_name,
            print_freq=200,
            verbose=0,
        ),
    ]

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks,
                progress_bar=False)
    elapsed = time.time() - start

    model.save(os.path.join(model_dir, "final"))
    env.close()

    # Read back episode log for summary
    log_path = os.path.join(log_dir, "episode_log.json")
    with open(log_path, "r") as f:
        ep_log = json.load(f)

    outcomes = ep_log["outcomes"]
    rewards = ep_log["rewards"]

    summary = {
        "env": env_name,
        "run_name": name,
        "params": {k: v for k, v in params.items()},
        "total_episodes": len(outcomes),
        "final_win_rate_100": float(np.mean(outcomes[-100:])) if len(outcomes) >= 100 else None,
        "final_win_rate_200": float(np.mean(outcomes[-200:])) if len(outcomes) >= 200 else None,
        "mean_reward": float(np.mean(rewards)),
        "mean_reward_last_100": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else None,
        "training_time_sec": elapsed,
        "status": "completed",
    }
    return summary


def load_existing_summary(env_name, run_idx, params):
    """Try to load summary from an already-completed run."""
    name = run_name(run_idx, params)
    log_path = os.path.join(SWEEP_DIR, env_name, name, "episode_log.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        ep_log = json.load(f)
    outcomes = ep_log["outcomes"]
    rewards = ep_log["rewards"]
    return {
        "env": env_name,
        "run_name": name,
        "params": {k: v for k, v in params.items()},
        "total_episodes": len(outcomes),
        "final_win_rate_100": float(np.mean(outcomes[-100:])) if len(outcomes) >= 100 else None,
        "final_win_rate_200": float(np.mean(outcomes[-200:])) if len(outcomes) >= 200 else None,
        "mean_reward": float(np.mean(rewards)),
        "mean_reward_last_100": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else None,
        "training_time_sec": None,
        "status": "completed (resumed)",
    }


def run_sweep(envs, dry_run=False):
    grid = make_grid()
    print(f"Sweep: {len(grid)} configurations x {len(envs)} environments "
          f"= {len(grid) * len(envs)} total runs\n")

    if dry_run:
        for i, params in enumerate(grid):
            print(f"  [{i:02d}] lr={params['learning_rate']:.0e}  "
                  f"ent={params['ent_coef']:.3f}  "
                  f"n_steps={params['n_steps']}")
        print(f"\nEnvironments: {', '.join(envs)}")
        print(f"Timesteps: {', '.join(f'{e}={TIMESTEPS[e]:,}' for e in envs)}")
        return

    all_results = []
    for env_name in envs:
        print(f"\n{'=' * 60}")
        print(f"  ENVIRONMENT: {env_name.upper()} ({TIMESTEPS[env_name]:,} steps)")
        print(f"{'=' * 60}")

        for i, params in enumerate(grid):
            name = run_name(i, params)

            # Resume support: skip completed runs
            existing = load_existing_summary(env_name, i, params)
            if existing is not None:
                all_results.append(existing)
                wr = existing.get("final_win_rate_100")
                wr_str = f"{100 * wr:.1f}%" if wr else "N/A"
                print(f"  [{i + 1:2d}/{len(grid)}] {name} -- SKIPPED (done, WR={wr_str})")
                continue

            print(f"  [{i + 1:2d}/{len(grid)}] {name} -- STARTING")
            try:
                summary = run_single_experiment(
                    env_name, params, run_idx=i, worker_id=i + 10)
                all_results.append(summary)
                wr = summary.get("final_win_rate_100")
                wr_str = f"{100 * wr:.1f}%" if wr else "N/A"
                print(f"             -> Win rate: {wr_str}, "
                      f"Time: {summary['training_time_sec'] / 60:.1f} min")
            except Exception as e:
                print(f"             -> FAILED: {e}")
                traceback.print_exc()
                all_results.append({
                    "env": env_name, "run_name": name,
                    "params": {k: v for k, v in params.items()},
                    "status": "failed", "error": str(e),
                })

    # Save all results
    os.makedirs(SWEEP_DIR, exist_ok=True)
    results_path = os.path.join(SWEEP_DIR, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSweep results saved to {results_path}")
    print(f"Total runs: {len(all_results)} "
          f"({sum(1 for r in all_results if r.get('status', '').startswith('completed'))} completed, "
          f"{sum(1 for r in all_results if r.get('status') == 'failed')} failed)")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid sweep for PPO on Unity ball game")
    parser.add_argument("--env", type=str, default=None,
                        choices=["simple", "medium", "hard"],
                        help="Run sweep on a single environment (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all configurations without training")
    args = parser.parse_args()

    envs = [args.env] if args.env else ["simple", "medium", "hard"]
    run_sweep(envs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
