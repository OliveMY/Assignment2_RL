"""
Single-variable ablation runner for PPO hyperparameter experiments.

Holds all parameters at baseline, varies one at a time.
Each run is labeled explicitly (e.g., clip_range_0.1) and saves
structured JSON results alongside optional PPO diagnostics.

Usage:
    python src/ablation.py --param clip_range --env simple
    python src/ablation.py --param ent_coef --env simple --seeds 0 42 123
    python src/ablation.py --param clip_range --env simple --values 0.1 0.2 0.3
    python src/ablation.py --list                # Show all ablation definitions
    python src/ablation.py --param clip_range --env simple --dry-run
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from config import get_config, get_ppo_kwargs
from env_wrapper import make_env
from callbacks import TrainingLogCallback
from diagnostics_callback import DiagnosticsCallback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION_DIR = os.path.join(BASE_DIR, "results", "ablations")

# Ablation definitions: parameter name -> list of values to test
# The baseline value is included in each list for comparison.
ABLATIONS = {
    "clip_range": {
        "values": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        "diagnostics": True,
    },
    "gae_lambda": {
        "values": [0.8, 0.9, 0.92, 0.95, 0.98, 1.0],
        "diagnostics": True,
    },
    "ent_coef": {
        "values": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        "diagnostics": True,
    },
    "learning_rate": {
        "values": [1e-4, 3e-4, 5e-4, 1e-3],
        "diagnostics": False,
    },
    "lr_schedule": {
        "values": ["constant", "linear_decay"],
        "diagnostics": False,
    },
    "n_epochs": {
        "values": [3, 5, 10, 15, 20],
        "diagnostics": True,
    },
    "gamma": {
        "values": [0.9, 0.95, 0.99, 0.999],
        "diagnostics": False,
    },
    "batch_size": {
        "values": [32, 64, 128, 256],
        "diagnostics": False,
    },
    "n_steps": {
        "values": [1024, 2048, 4096],
        "diagnostics": False,
    },
    "net_arch": {
        "values": [
            [64, 64],
            [128, 128],
            [256, 256],
            [128, 128, 128],
        ],
        "diagnostics": False,
    },
    "reward_shaping": {
        "values": ["none", "pbrs_proximity", "dense_contact", "dense_alive", "composite"],
        "diagnostics": False,
    },
    "shaping_scale": {
        "values": [0.01, 0.05, 0.1, 0.2, 0.5],
        "diagnostics": False,
    },
}

# Default timesteps per environment for ablation runs
ABLATION_TIMESTEPS = {
    "simple": 1_000_000,
    "medium": 2_000_000,
    "hard": 4_000_000,
}


def run_label(param_name, value):
    """Generate a human-readable label for one run."""
    if isinstance(value, list):
        return f"{param_name}_{'x'.join(str(v) for v in value)}"
    elif isinstance(value, float):
        if value < 0.01:
            return f"{param_name}_{value:.0e}"
        return f"{param_name}_{value}"
    else:
        return f"{param_name}_{value}"


def run_single_ablation(env_name, param_name, value, seed, timesteps,
                        worker_id, use_diagnostics, skip_eval,
                        baseline_overrides=None):
    """Train one PPO run with a single parameter override. Returns summary dict."""
    config = get_config(env_name)

    # Apply baseline overrides (e.g., lr, batch_size, lr_schedule)
    if baseline_overrides:
        config.update(baseline_overrides)

    # Apply the ablation override
    if param_name == "net_arch":
        config["policy_kwargs"] = {
            "net_arch": dict(pi=list(value), vf=list(value)),
        }
    elif param_name == "lr_schedule":
        config["lr_schedule"] = value
    elif param_name in ("reward_shaping", "shaping_scale"):
        config[param_name] = value
        # For shaping_scale ablation, use pbrs_proximity as the default strategy
        if param_name == "shaping_scale":
            config["reward_shaping"] = config.get("reward_shaping", "pbrs_proximity")
            if config["reward_shaping"] == "none":
                config["reward_shaping"] = "pbrs_proximity"
    else:
        config[param_name] = value

    total_timesteps = timesteps or ABLATION_TIMESTEPS[env_name]
    time_scale = config.pop("time_scale", 20.0)
    no_graphics = config.pop("no_graphics", True)
    config.pop("total_timesteps", None)

    label = run_label(param_name, value)
    seed_label = f"seed{seed}" if seed is not None else "noseed"
    log_dir = os.path.join(ABLATION_DIR, env_name, param_name, f"{label}_{seed_label}")
    model_dir = os.path.join(log_dir, "model")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Check for existing results (resume support)
    results_path = os.path.join(log_dir, "ablation_result.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            existing = json.load(f)
        if existing.get("status", "").startswith("completed"):
            return existing

    # Seed
    if seed is not None:
        set_random_seed(seed)

    # Build reward shaping config
    shaping_strategy = config.pop("reward_shaping", "none")
    shaping_scale = config.pop("shaping_scale", 0.1)
    reward_shaping_config = None
    if shaping_strategy != "none":
        reward_shaping_config = {
            "strategy": shaping_strategy,
            "shaping_scale": shaping_scale,
            "gamma": config.get("gamma", 0.99),
        }

    # Create environment
    env = make_env(env_name, time_scale=time_scale,
                   no_graphics=no_graphics, worker_id=worker_id,
                   reward_shaping_config=reward_shaping_config)
    env = Monitor(env)

    # Create model
    ppo_kwargs = get_ppo_kwargs(config)
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=None, **ppo_kwargs)

    # Callbacks
    callbacks = [
        TrainingLogCallback(
            log_dir=log_dir,
            config={**config, "total_timesteps": total_timesteps},
            env_name=env_name,
            print_freq=200,
            verbose=0,
        ),
    ]
    if use_diagnostics:
        callbacks.append(DiagnosticsCallback(log_dir=log_dir, verbose=0))

    # Train
    start = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    progress_bar=False)
    except KeyboardInterrupt:
        print(f"\n  [{label}] Training interrupted.")
    finally:
        elapsed = time.time() - start
        model.save(os.path.join(model_dir, "final"))
        env.close()

    # Read episode log for summary
    ep_log_path = os.path.join(log_dir, "episode_log.json")
    if os.path.exists(ep_log_path):
        with open(ep_log_path, "r") as f:
            ep_log = json.load(f)
        outcomes = ep_log.get("outcomes", [])
        rewards = ep_log.get("rewards", [])
    else:
        outcomes, rewards = [], []

    # Compute win rate (wins only, not draws)
    wins = sum(1 for o in outcomes if o == 1)
    total = len(outcomes)

    summary = {
        "env": env_name,
        "param": param_name,
        "value": value,
        "seed": seed,
        "label": label,
        "total_episodes": total,
        "wins": wins,
        "win_rate": wins / total if total > 0 else None,
        "win_rate_last_100": sum(1 for o in outcomes[-100:] if o == 1) / min(100, len(outcomes[-100:])) if outcomes else None,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "mean_reward_last_100": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else None,
        "training_time_sec": elapsed,
        "total_timesteps": total_timesteps,
        "status": "completed",
        "has_diagnostics": use_diagnostics,
    }

    # Save result
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_ablation_phase(param_name, env_name, seeds, timesteps, worker_id_base,
                       values=None, skip_eval=True, dry_run=False,
                       baseline_overrides=None):
    """Run all values for one parameter ablation."""
    if param_name not in ABLATIONS:
        print(f"Unknown ablation parameter: {param_name}")
        print(f"Available: {', '.join(sorted(ABLATIONS.keys()))}")
        return []

    ablation_def = ABLATIONS[param_name]
    test_values = values if values is not None else ablation_def["values"]
    use_diagnostics = ablation_def["diagnostics"]

    total_runs = len(test_values) * len(seeds)
    ts = timesteps or ABLATION_TIMESTEPS[env_name]

    print(f"\n{'='*60}")
    print(f"  ABLATION: {param_name} on {env_name.upper()}")
    print(f"  Values: {test_values}")
    print(f"  Seeds: {seeds}")
    print(f"  Runs: {total_runs} ({len(test_values)} values x {len(seeds)} seeds)")
    print(f"  Timesteps per run: {ts:,}")
    print(f"  Diagnostics: {'Yes' if use_diagnostics else 'No'}")
    if baseline_overrides:
        print(f"  Baseline overrides: {baseline_overrides}")
    print(f"{'='*60}")

    if dry_run:
        for i, val in enumerate(test_values):
            for seed in seeds:
                label = run_label(param_name, val)
                print(f"  [{i+1}] {label} seed={seed}")
        return []

    all_results = []
    first_run_speed = None

    for i, val in enumerate(test_values):
        for j, seed in enumerate(seeds):
            run_num = i * len(seeds) + j + 1
            label = run_label(param_name, val)
            wid = worker_id_base + i * 10 + j

            # Time estimate after first run
            if first_run_speed is not None:
                remaining = total_runs - run_num + 1
                est_minutes = remaining * (ts / first_run_speed) / 60
                print(f"\n  [{run_num}/{total_runs}] {label} seed={seed} "
                      f"(~{est_minutes:.0f} min remaining)")
            else:
                print(f"\n  [{run_num}/{total_runs}] {label} seed={seed}")

            try:
                summary = run_single_ablation(
                    env_name=env_name,
                    param_name=param_name,
                    value=val,
                    seed=seed,
                    timesteps=timesteps,
                    worker_id=wid,
                    use_diagnostics=use_diagnostics,
                    skip_eval=skip_eval,
                    baseline_overrides=baseline_overrides,
                )
                all_results.append(summary)

                if summary.get("status", "").startswith("completed"):
                    wr = summary.get("win_rate_last_100")
                    elapsed = summary.get("training_time_sec", 0)
                    wr_str = f"{100*wr:.1f}%" if wr is not None else "N/A"

                    if first_run_speed is None and elapsed > 0:
                        first_run_speed = ts / elapsed  # steps/sec

                    print(f"      -> WR={wr_str}, Time={elapsed/60:.1f}min, "
                          f"Speed={ts/elapsed:.0f} steps/s" if elapsed > 0 else "")
                else:
                    print(f"      -> RESUMED (already completed)")

            except Exception as e:
                print(f"      -> FAILED: {e}")
                all_results.append({
                    "env": env_name, "param": param_name,
                    "value": val, "seed": seed, "label": label,
                    "status": "failed", "error": str(e),
                })

    # Save combined results
    combined_path = os.path.join(ABLATION_DIR, env_name, param_name,
                                 "ablation_summary.json")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n--- {param_name} Ablation Summary ({env_name}) ---")
    print(f"  {'Value':<20} {'WR(last 100)':<15} {'Mean Reward':<15} {'Time':<10}")
    print(f"  {'-'*60}")
    for r in all_results:
        if r.get("status", "").startswith("completed"):
            val_str = str(r["value"])
            wr = r.get("win_rate_last_100")
            mr = r.get("mean_reward_last_100")
            t = r.get("training_time_sec", 0)
            print(f"  {val_str:<20} "
                  f"{f'{100*wr:.1f}%' if wr is not None else 'N/A':<15} "
                  f"{f'{mr:.3f}' if mr is not None else 'N/A':<15} "
                  f"{t/60:.1f}m")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Single-variable PPO ablation runner")
    parser.add_argument("--param", type=str, default=None,
                        help="Parameter to ablate (e.g., clip_range, ent_coef)")
    parser.add_argument("--env", type=str, default="simple",
                        choices=["simple", "medium", "hard"])
    parser.add_argument("--values", nargs="+", default=None,
                        help="Override default values (space-separated)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Random seeds to use (default: 0)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps per run")
    parser.add_argument("--worker-id", type=int, default=10,
                        help="Base worker ID (default: 10)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override baseline learning rate")
    parser.add_argument("--lr-schedule", type=str, default=None,
                        choices=["constant", "linear_decay"],
                        help="Override baseline LR schedule")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override baseline batch size")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="Override baseline entropy coefficient")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configurations without running")
    parser.add_argument("--list", action="store_true",
                        help="List all available ablation parameters")
    args = parser.parse_args()

    if args.list:
        print("Available ablation parameters:")
        for name, defn in sorted(ABLATIONS.items()):
            vals = defn["values"]
            diag = "with diagnostics" if defn["diagnostics"] else "no diagnostics"
            print(f"  {name:<20} {len(vals)} values  ({diag})")
            print(f"    values: {vals}")
        return

    if args.param is None:
        parser.error("--param is required (use --list to see available parameters)")

    # Parse values from CLI if provided
    values = None
    if args.values:
        param_def = ABLATIONS.get(args.param, {})
        sample = param_def.get("values", [None])[0] if param_def else None
        if isinstance(sample, float) or args.param in ("clip_range", "gae_lambda",
                                                        "ent_coef", "learning_rate",
                                                        "gamma", "shaping_scale"):
            values = [float(v) for v in args.values]
        elif isinstance(sample, int) or args.param in ("n_epochs", "batch_size", "n_steps"):
            values = [int(v) for v in args.values]
        elif isinstance(sample, list) or args.param == "net_arch":
            # Parse "64x64" format
            values = [[int(x) for x in v.split("x")] for v in args.values]
        else:
            values = args.values  # strings (e.g., lr_schedule, reward_shaping)

    # Build baseline overrides from CLI flags
    baseline_overrides = {}
    if args.lr is not None:
        baseline_overrides["learning_rate"] = args.lr
    if args.lr_schedule is not None:
        baseline_overrides["lr_schedule"] = args.lr_schedule
    if args.batch_size is not None:
        baseline_overrides["batch_size"] = args.batch_size
    if args.ent_coef is not None:
        baseline_overrides["ent_coef"] = args.ent_coef

    run_ablation_phase(
        param_name=args.param,
        env_name=args.env,
        seeds=args.seeds,
        timesteps=args.timesteps,
        worker_id_base=args.worker_id,
        values=values,
        dry_run=args.dry_run,
        baseline_overrides=baseline_overrides or None,
    )


if __name__ == "__main__":
    main()
