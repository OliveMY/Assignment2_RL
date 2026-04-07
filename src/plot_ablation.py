"""
Analysis and plotting for PPO ablation experiments.

Reads results from results/ablations/{env}/{param}/ and generates:
- Win rate vs timesteps plots (one line per parameter value)
- PPO diagnostics plots (clip_fraction, approx_kl, etc.) when available
- Summary tables in CSV and terminal output

Usage:
    python src/plot_ablation.py --param clip_range --env simple
    python src/plot_ablation.py --param clip_range --env simple --no-plots
    python src/plot_ablation.py --all --env simple
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION_DIR = os.path.join(BASE_DIR, "results", "ablations")


def load_episode_log(run_dir):
    """Load episode-level data from a run directory."""
    path = os.path.join(run_dir, "episode_log.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_diagnostics(run_dir):
    """Load JSONL diagnostics from a run directory."""
    path = os.path.join(run_dir, "diagnostics.jsonl")
    if not os.path.exists(path):
        return None
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries if entries else None


def load_ablation_result(run_dir):
    """Load the ablation result summary."""
    path = os.path.join(run_dir, "ablation_result.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def rolling_win_rate(outcomes, window=100):
    """Compute rolling win rate over a list of outcomes (1=win, 0=loss, 0.5=draw)."""
    if not outcomes:
        return [], []
    rates = []
    indices = []
    for i in range(window, len(outcomes) + 1):
        chunk = outcomes[i - window:i]
        wins = sum(1 for o in chunk if o == 1)
        rates.append(wins / window)
        indices.append(i)
    return indices, rates


def discover_runs(env_name, param_name):
    """Find all run directories for a given ablation."""
    param_dir = os.path.join(ABLATION_DIR, env_name, param_name)
    if not os.path.isdir(param_dir):
        return []
    runs = []
    for name in sorted(os.listdir(param_dir)):
        run_dir = os.path.join(param_dir, name)
        if os.path.isdir(run_dir):
            result = load_ablation_result(run_dir)
            if result:
                runs.append({"dir": run_dir, "name": name, "result": result})
    return runs


def print_summary_table(runs, param_name, env_name):
    """Print a summary table of ablation results."""
    print(f"\n{'='*70}")
    print(f"  {param_name.upper()} Ablation — {env_name.upper()}")
    print(f"{'='*70}")
    print(f"  {'Value':<20} {'Seed':<6} {'WR(100)':<10} {'Reward':<10} {'Time':<8} {'Episodes':<8}")
    print(f"  {'-'*65}")

    for r in runs:
        res = r["result"]
        if not res.get("status", "").startswith("completed"):
            continue
        val = str(res.get("value", "?"))
        seed = str(res.get("seed", "?"))
        wr = res.get("win_rate_last_100")
        mr = res.get("mean_reward_last_100")
        t = res.get("training_time_sec", 0)
        ep = res.get("total_episodes", 0)
        print(f"  {val:<20} {seed:<6} "
              f"{f'{100*wr:.1f}%' if wr is not None else 'N/A':<10} "
              f"{f'{mr:.3f}' if mr is not None else 'N/A':<10} "
              f"{f'{t/60:.1f}m' if t else 'N/A':<8} "
              f"{ep:<8}")

    # Aggregate by value (average across seeds)
    by_value = defaultdict(list)
    for r in runs:
        res = r["result"]
        if res.get("status", "").startswith("completed") and res.get("win_rate_last_100") is not None:
            by_value[str(res["value"])].append(res["win_rate_last_100"])

    if any(len(v) > 1 for v in by_value.values()):
        print(f"\n  Aggregated (mean +/- std across seeds):")
        print(f"  {'Value':<20} {'Mean WR':<10} {'Std':<10} {'N':<4}")
        print(f"  {'-'*45}")
        for val, wrs in sorted(by_value.items()):
            mean = np.mean(wrs)
            std = np.std(wrs)
            print(f"  {val:<20} {100*mean:<10.1f}% {100*std:<10.1f} {len(wrs):<4}")


def export_csv(runs, param_name, env_name):
    """Export ablation results as CSV."""
    csv_dir = os.path.join(ABLATION_DIR, env_name, param_name)
    csv_path = os.path.join(csv_dir, "ablation_results.csv")
    fieldnames = [
        "value", "seed", "win_rate", "win_rate_last_100",
        "mean_reward", "mean_reward_last_100",
        "total_episodes", "training_time_sec", "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            res = r["result"]
            writer.writerow({k: res.get(k, "") for k in fieldnames})
    print(f"\n  CSV exported: {csv_path}")


def make_plots(runs, param_name, env_name):
    """Generate matplotlib plots for an ablation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not available, skipping plots.")
        return

    plot_dir = os.path.join(ABLATION_DIR, env_name, param_name, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Group runs by value
    by_value = defaultdict(list)
    for r in runs:
        res = r["result"]
        if res.get("status", "").startswith("completed"):
            by_value[str(res["value"])].append(r)

    # --- Win rate learning curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for val_str, val_runs in sorted(by_value.items()):
        for vr in val_runs:
            ep_log = load_episode_log(vr["dir"])
            if ep_log and ep_log.get("outcomes"):
                indices, rates = rolling_win_rate(ep_log["outcomes"])
                label = f"{val_str}" if len(val_runs) == 1 else f"{val_str} (s{vr['result'].get('seed', '?')})"
                ax.plot(indices, [r * 100 for r in rates], label=label, alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (rolling 100) %")
    ax.set_title(f"{param_name} Ablation — {env_name.upper()}")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "win_rate_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")

    # --- Bar chart of final win rates ---
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    heights = []
    for val_str, val_runs in sorted(by_value.items()):
        wrs = [vr["result"].get("win_rate_last_100", 0) or 0 for vr in val_runs]
        labels.append(val_str)
        heights.append(np.mean(wrs) * 100)
    colors = plt.cm.RdYlGn([h / 100 for h in heights])
    ax.bar(range(len(labels)), heights, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Win Rate (last 100) %")
    ax.set_title(f"{param_name} — Final Win Rate Comparison ({env_name.upper()})")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    path = os.path.join(plot_dir, "win_rate_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")

    # --- Diagnostics plots (if available) ---
    has_diag = False
    for val_runs in by_value.values():
        for vr in val_runs:
            if load_diagnostics(vr["dir"]):
                has_diag = True
                break

    if has_diag:
        diag_keys = ["clip_fraction", "approx_kl", "explained_variance",
                     "entropy_loss", "policy_gradient_loss", "value_loss"]
        for dkey in diag_keys:
            fig, ax = plt.subplots(figsize=(12, 5))
            found_data = False
            for val_str, val_runs in sorted(by_value.items()):
                for vr in val_runs:
                    diag = load_diagnostics(vr["dir"])
                    if diag:
                        steps = [d["step"] for d in diag if dkey in d]
                        values = [d[dkey] for d in diag if dkey in d]
                        if steps:
                            label = f"{val_str}" if len(val_runs) == 1 else f"{val_str} (s{vr['result'].get('seed', '?')})"
                            ax.plot(steps, values, label=label, alpha=0.7)
                            found_data = True
            if found_data:
                ax.set_xlabel("Timestep")
                ax.set_ylabel(dkey)
                ax.set_title(f"{param_name} — {dkey} ({env_name.upper()})")
                ax.legend(fontsize=8, loc="best")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                path = os.path.join(plot_dir, f"diag_{dkey}.png")
                fig.savefig(path, dpi=150)
                print(f"  Plot: {path}")
            plt.close(fig)


def analyze_param(param_name, env_name, skip_plots=False):
    """Full analysis of one ablation parameter."""
    runs = discover_runs(env_name, param_name)
    if not runs:
        print(f"\n  No results found for {param_name} on {env_name}")
        print(f"  Expected at: {os.path.join(ABLATION_DIR, env_name, param_name)}/")
        return

    print_summary_table(runs, param_name, env_name)
    export_csv(runs, param_name, env_name)
    if not skip_plots:
        make_plots(runs, param_name, env_name)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PPO ablation results and generate plots")
    parser.add_argument("--param", type=str, default=None,
                        help="Parameter to analyze (e.g., clip_range)")
    parser.add_argument("--env", type=str, default="simple",
                        choices=["simple", "medium", "hard"])
    parser.add_argument("--all", action="store_true",
                        help="Analyze all parameters that have results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    if args.all:
        env_dir = os.path.join(ABLATION_DIR, args.env)
        if not os.path.isdir(env_dir):
            print(f"No ablation results found for {args.env}")
            return
        for param_name in sorted(os.listdir(env_dir)):
            if os.path.isdir(os.path.join(env_dir, param_name)):
                analyze_param(param_name, args.env, skip_plots=args.no_plots)
    elif args.param:
        analyze_param(args.param, args.env, skip_plots=args.no_plots)
    else:
        parser.error("--param or --all is required")


if __name__ == "__main__":
    main()
