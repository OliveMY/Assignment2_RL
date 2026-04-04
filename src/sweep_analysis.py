"""
Analyze hyperparameter sweep results and produce tables, CSV, and plots.

Usage:
    python src/sweep_analysis.py                # Full analysis
    python src/sweep_analysis.py --no-plots     # Skip matplotlib plots
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(BASE_DIR, "results", "sweep")
RESULTS_PATH = os.path.join(SWEEP_DIR, "sweep_results.json")

SWEEP_PARAMS = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "ent_coef": [0.001, 0.01, 0.05],
    "n_steps": [512, 2048, 4096],
}


def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def print_ranked_table(results, env_name):
    """Print runs ranked by final win rate for one environment."""
    env_runs = [r for r in results
                if r["env"] == env_name and r.get("status", "").startswith("completed")]
    if not env_runs:
        print(f"  No completed runs for {env_name}")
        return

    env_runs.sort(key=lambda r: r.get("final_win_rate_100") or 0, reverse=True)

    print(f"\n{'=' * 80}")
    print(f"  {env_name.upper()} — Ranked by Final Win Rate (last 100 episodes)")
    print(f"{'=' * 80}")
    print(f"  {'Rank':<5} {'LR':<10} {'Ent Coef':<10} {'N Steps':<8} "
          f"{'WR-100':<8} {'WR-200':<8} {'Reward':<10} {'Episodes':<8}")
    print(f"  {'-' * 75}")

    for rank, r in enumerate(env_runs, 1):
        p = r["params"]
        wr100 = r.get("final_win_rate_100")
        wr200 = r.get("final_win_rate_200")
        mr = r.get("mean_reward_last_100")
        print(f"  {rank:<5} {p['learning_rate']:<10.0e} {p['ent_coef']:<10.3f} "
              f"{p['n_steps']:<8} "
              f"{f'{100*wr100:.1f}%' if wr100 is not None else 'N/A':<8} "
              f"{f'{100*wr200:.1f}%' if wr200 is not None else 'N/A':<8} "
              f"{f'{mr:.3f}' if mr is not None else 'N/A':<10} "
              f"{r['total_episodes']:<8}")

    best = env_runs[0]
    p = best["params"]
    print(f"\n  Best: lr={p['learning_rate']:.0e}, ent_coef={p['ent_coef']:.3f}, "
          f"n_steps={p['n_steps']} "
          f"-> Win Rate = {100 * (best.get('final_win_rate_100') or 0):.1f}%")


def print_sensitivity(results, env_name):
    """Show marginal effect of each hyperparameter."""
    env_runs = [r for r in results
                if r["env"] == env_name and r.get("status", "").startswith("completed")]
    if not env_runs:
        return

    print(f"\n  Parameter Sensitivity ({env_name}):")
    print(f"  {'Parameter':<15} {'Value':<12} {'Mean WR-100':<12} {'Std':<8} {'N':<4}")
    print(f"  {'-' * 55}")

    for param_name, values in sorted(SWEEP_PARAMS.items()):
        for val in values:
            matching = [r for r in env_runs
                        if abs(r["params"][param_name] - val) < 1e-10]
            win_rates = [r["final_win_rate_100"] for r in matching
                         if r.get("final_win_rate_100") is not None]
            if win_rates:
                mean_wr = np.mean(win_rates)
                std_wr = np.std(win_rates)
                print(f"  {param_name:<15} {val:<12.4g} {100*mean_wr:<12.1f}% "
                      f"{100*std_wr:<8.1f} {len(win_rates):<4}")
        print()


def export_csv(results):
    """Write sweep_summary.csv for report import."""
    csv_path = os.path.join(SWEEP_DIR, "sweep_summary.csv")
    fieldnames = [
        "env", "learning_rate", "ent_coef", "n_steps",
        "final_win_rate_100", "final_win_rate_200",
        "mean_reward", "mean_reward_last_100",
        "total_episodes", "training_time_sec", "status",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "env": r["env"],
                "status": r.get("status", "unknown"),
            }
            if "params" in r:
                row.update(r["params"])
            for key in ["final_win_rate_100", "final_win_rate_200",
                        "mean_reward", "mean_reward_last_100",
                        "total_episodes", "training_time_sec"]:
                row[key] = r.get(key, "")
            writer.writerow(row)

    print(f"\nCSV exported to {csv_path}")


def make_plots(results):
    """Generate matplotlib plots for the sweep results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    envs = sorted(set(r["env"] for r in results))
    plot_dir = os.path.join(SWEEP_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for env_name in envs:
        env_runs = [r for r in results
                    if r["env"] == env_name and r.get("status", "").startswith("completed")]
        if not env_runs:
            continue

        env_runs.sort(key=lambda r: r.get("final_win_rate_100") or 0, reverse=True)

        # --- Bar chart of win rates ---
        fig, ax = plt.subplots(figsize=(14, 6))
        names = [r["run_name"].replace(f"run_", "").split("_", 1)[1] for r in env_runs]
        win_rates = [100 * (r.get("final_win_rate_100") or 0) for r in env_runs]
        colors = plt.cm.RdYlGn([wr / 100 for wr in win_rates])
        ax.bar(range(len(names)), win_rates, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Win Rate (last 100 ep) %")
        ax.set_title(f"{env_name.upper()} — Sweep Win Rates")
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(plot_dir, f"{env_name}_win_rates.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {path}")

        # --- Pairwise heatmaps ---
        param_names = sorted(SWEEP_PARAMS.keys())
        for i, p1 in enumerate(param_names):
            for j, p2 in enumerate(param_names):
                if j <= i:
                    continue
                fig, ax = plt.subplots(figsize=(6, 5))
                v1 = sorted(SWEEP_PARAMS[p1])
                v2 = sorted(SWEEP_PARAMS[p2])
                heatmap = np.zeros((len(v2), len(v1)))
                for r in env_runs:
                    wr = r.get("final_win_rate_100") or 0
                    idx1 = v1.index(r["params"][p1])
                    idx2 = v2.index(r["params"][p2])
                    # Average over the third parameter
                    heatmap[idx2, idx1] += wr
                # Each cell averages over 3 values of the third param
                heatmap /= len(list(SWEEP_PARAMS.values())[0])
                heatmap *= 100

                im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0, vmax=100,
                               aspect="auto", origin="lower")
                ax.set_xticks(range(len(v1)))
                ax.set_xticklabels([f"{v:.4g}" for v in v1])
                ax.set_yticks(range(len(v2)))
                ax.set_yticklabels([f"{v:.4g}" for v in v2])
                ax.set_xlabel(p1)
                ax.set_ylabel(p2)
                ax.set_title(f"{env_name.upper()} — {p1} vs {p2}")
                for yi in range(len(v2)):
                    for xi in range(len(v1)):
                        ax.text(xi, yi, f"{heatmap[yi, xi]:.0f}%",
                                ha="center", va="center", fontsize=10,
                                color="black" if 30 < heatmap[yi, xi] < 70 else "white")
                fig.colorbar(im, label="Win Rate %")
                plt.tight_layout()
                path = os.path.join(plot_dir, f"{env_name}_{p1}_vs_{p2}.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)
                print(f"  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    args = parser.parse_args()

    if not os.path.exists(RESULTS_PATH):
        print(f"No sweep results found at {RESULTS_PATH}")
        print("Run src/sweep.py first.")
        return

    results = load_results()
    completed = [r for r in results if r.get("status", "").startswith("completed")]
    failed = [r for r in results if r.get("status") == "failed"]
    print(f"Loaded {len(results)} runs ({len(completed)} completed, {len(failed)} failed)")

    envs = sorted(set(r["env"] for r in results))
    for env_name in envs:
        print_ranked_table(results, env_name)
        print_sensitivity(results, env_name)

    export_csv(results)

    if not args.no_plots:
        make_plots(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
