"""
Auto-generate experiment reports from training data and human annotations.

Reads experiments.yaml (human-edited registry of experiments with annotations)
and TensorBoard event files (auto-extracted training metrics) to produce a
structured Markdown report.

Usage:
    python src/document_experiments.py                    # Generate EXPERIMENTS.md
    python src/document_experiments.py --init             # Create template experiments.yaml
    python src/document_experiments.py --add              # Add new experiment interactively
    python src/document_experiments.py --env hard         # Filter to one environment
    python src/document_experiments.py --output report.md # Custom output path
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REGISTRY_PATH = os.path.join(BASE_DIR, "experiments.yaml")
OUTPUT_PATH = os.path.join(BASE_DIR, "EXPERIMENTS.md")

MILESTONE_THRESHOLDS = [50, 90, 95, 100]

DIAG_TAGS = [
    "train/clip_fraction",
    "train/approx_kl",
    "train/entropy_loss",
    "train/explained_variance",
    "train/policy_gradient_loss",
    "train/value_loss",
]

DIAG_LABELS = {
    "train/clip_fraction": "Clip Fraction",
    "train/approx_kl": "Approx KL",
    "train/entropy_loss": "Entropy Loss",
    "train/explained_variance": "Explained Variance",
    "train/policy_gradient_loss": "Policy Gradient Loss",
    "train/value_loss": "Value Loss",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentData:
    """All data for one experiment, from registry + auto-extracted."""
    # From registry
    id: str
    run: str
    env: str
    description: str
    config_overrides: dict
    hypothesis: str
    findings: str
    decision: str
    baseline: Optional[str] = None
    tags: list = field(default_factory=list)

    # Auto-extracted
    eval_curve: list = field(default_factory=list)       # [(step, win_rate)]
    reward_curve: list = field(default_factory=list)      # [(step, avg_reward)]
    diagnostics: dict = field(default_factory=dict)       # tag -> [(step, value)]
    rollout_win_rate: list = field(default_factory=list)  # [(step, wr)]

    # Derived
    milestones: dict = field(default_factory=dict)   # {50: step, 90: step, ...}
    final_win_rate: Optional[float] = None
    final_avg_reward: Optional[float] = None
    final_step: Optional[int] = None
    training_wall_time: Optional[float] = None       # seconds
    config_diff: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TensorBoard reading
# ---------------------------------------------------------------------------

def read_tensorboard(event_dir):
    """Read all scalar tags from a TensorBoard event directory.

    Returns {tag: [(step, value), ...]} sorted by step.
    Returns empty dict if no event files found.
    """
    if not os.path.isdir(event_dir):
        return {}, None, None

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
            SCALARS,
        )
    except ImportError:
        print("Warning: tensorboard not installed, skipping TensorBoard data.")
        return {}, None, None

    ea = EventAccumulator(event_dir, size_guidance={SCALARS: 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        return {}, None, None

    data = {}
    first_wall = None
    last_wall = None

    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            data[tag] = [(e.step, e.value) for e in events]
            if first_wall is None or events[0].wall_time < first_wall:
                first_wall = events[0].wall_time
            if last_wall is None or events[-1].wall_time > last_wall:
                last_wall = events[-1].wall_time

    wall_time = (last_wall - first_wall) if first_wall and last_wall else None
    return data, wall_time, len(tags)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_milestones(eval_curve, thresholds=None):
    """Find the first step where win_rate >= each threshold.

    Args:
        eval_curve: [(step, win_rate), ...] sorted by step
        thresholds: list of ints (50, 90, 95, 100)

    Returns: {50: 327680, 90: 1200384, 100: None, ...}
    """
    if thresholds is None:
        thresholds = MILESTONE_THRESHOLDS
    milestones = {t: None for t in thresholds}
    for step, wr in eval_curve:
        for t in thresholds:
            if milestones[t] is None and wr >= t / 100.0:
                milestones[t] = step
    return milestones


def compute_config_diff(base_config, overrides):
    """Compute what changed from base_config.

    Returns: {param: {"base": old_val, "value": new_val}, ...}
    """
    diff = {}
    for key, new_val in overrides.items():
        base_val = base_config.get(key)
        if base_val != new_val:
            diff[key] = {"base": base_val, "value": new_val}
    return diff


def summarize_diagnostics(tag_data):
    """Summarize a single diagnostics time series.

    Returns: {"mean", "early_mean", "late_mean", "trend"}
    """
    if not tag_data or len(tag_data) < 2:
        return None

    values = [v for _, v in tag_data]
    n = len(values)
    cutoff = max(1, n // 5)

    early = values[:cutoff]
    late = values[-cutoff:]

    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    mean = sum(values) / len(values)

    delta = late_mean - early_mean
    threshold = abs(mean) * 0.1 if abs(mean) > 0.001 else 0.01
    if delta > threshold:
        trend = "increasing"
    elif delta < -threshold:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "mean": mean,
        "early_mean": early_mean,
        "late_mean": late_mean,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# Experiment loading
# ---------------------------------------------------------------------------

def load_experiment(entry, base_config):
    """Load all data for one experiment from registry + TensorBoard."""
    exp = ExperimentData(
        id=entry["id"],
        run=entry["run"],
        env=entry["env"],
        description=entry.get("description", ""),
        config_overrides=entry.get("config_overrides", {}),
        hypothesis=entry.get("hypothesis", ""),
        findings=entry.get("findings", ""),
        decision=entry.get("decision", ""),
        baseline=entry.get("baseline"),
        tags=entry.get("tags", []),
    )

    # Read TensorBoard
    event_dir = os.path.join(RESULTS_DIR, exp.env, exp.run)
    tb_data, wall_time, n_tags = read_tensorboard(event_dir)

    exp.training_wall_time = wall_time

    # Extract eval curve
    if "eval/win_rate" in tb_data:
        exp.eval_curve = tb_data["eval/win_rate"]
        if exp.eval_curve:
            exp.final_win_rate = exp.eval_curve[-1][1]
            exp.final_step = exp.eval_curve[-1][0]

    if "eval/avg_reward" in tb_data:
        exp.reward_curve = tb_data["eval/avg_reward"]
        if exp.reward_curve:
            exp.final_avg_reward = exp.reward_curve[-1][1]

    # Rollout win rate (training-time, only available post env_wrapper fix)
    if "rollout/win_rate" in tb_data:
        exp.rollout_win_rate = tb_data["rollout/win_rate"]

    # Diagnostics
    for tag in DIAG_TAGS:
        if tag in tb_data:
            exp.diagnostics[tag] = tb_data[tag]

    # Compute derived metrics
    exp.milestones = compute_milestones(exp.eval_curve)
    exp.config_diff = compute_config_diff(base_config, exp.config_overrides)

    return exp


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------

def load_registry(path):
    """Load and validate experiments.yaml."""
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run with --init to create it.")
        sys.exit(1)

    with open(path, "r") as f:
        registry = yaml.safe_load(f)

    if not registry or "experiments" not in registry:
        print(f"Error: {path} must have an 'experiments' key.")
        sys.exit(1)

    base_config = registry.get("base_config", {})
    experiments = registry["experiments"]

    for i, exp in enumerate(experiments):
        for field in ["id", "run", "env"]:
            if field not in exp:
                print(f"Error: experiment #{i+1} is missing required field '{field}'")
                sys.exit(1)

    return base_config, experiments


def discover_runs():
    """Find all PPO_N directories across all environments."""
    runs = []
    if not os.path.isdir(RESULTS_DIR):
        return runs

    for env_name in sorted(os.listdir(RESULTS_DIR)):
        env_dir = os.path.join(RESULTS_DIR, env_name)
        if not os.path.isdir(env_dir):
            continue
        for run_name in sorted(os.listdir(env_dir)):
            run_dir = os.path.join(env_dir, run_name)
            if os.path.isdir(run_dir) and run_name.startswith("PPO_"):
                events = [f for f in os.listdir(run_dir) if f.startswith("events.")]
                if events:
                    runs.append({"env": env_name, "run": run_name, "dir": run_dir})
    return runs


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def fmt_steps(steps):
    """Format step count as human-readable string."""
    if steps is None:
        return "—"
    if steps >= 1_000_000:
        return f"{steps/1_000_000:.1f}M"
    elif steps >= 1_000:
        return f"{steps/1_000:.0f}K"
    return str(steps)


def fmt_pct(value):
    """Format a fraction as percentage."""
    if value is None:
        return "—"
    return f"{value * 100:.0f}%"


def fmt_time(seconds):
    """Format seconds as human-readable duration."""
    if seconds is None:
        return "—"
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/60:.0f}min"


def fmt_float(value, precision=4):
    """Format a float."""
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


def generate_summary_table(experiments):
    """Generate the overview summary table."""
    lines = []
    lines.append("| # | ID | Env | Description | Final WR | Steps to 90% | Steps to 100% | Time |")
    lines.append("|---|-----|-----|-------------|----------|--------------|---------------|------|")

    for i, exp in enumerate(experiments, 1):
        lines.append(
            f"| {i} | {exp.id} | {exp.env} | {exp.description} "
            f"| {fmt_pct(exp.final_win_rate)} "
            f"| {fmt_steps(exp.milestones.get(90))} "
            f"| {fmt_steps(exp.milestones.get(100))} "
            f"| {fmt_time(exp.training_wall_time)} |"
        )

    return "\n".join(lines)


def generate_config_comparison(experiments, base_config):
    """Generate a config comparison table showing only changed params."""
    # Collect all changed params across all experiments
    all_changed = set()
    for exp in experiments:
        all_changed.update(exp.config_diff.keys())
        all_changed.update(exp.config_overrides.keys())

    if not all_changed:
        return "*All experiments use identical configuration.*"

    params = sorted(all_changed)
    header = "| Parameter | Baseline |"
    sep = "|-----------|----------|"
    for exp in experiments:
        if "baseline" not in exp.tags:
            header += f" {exp.id} |"
            sep += "----------|"

    lines = [header, sep]
    for param in params:
        base_val = base_config.get(param, "—")
        row = f"| {param} | {base_val} |"
        for exp in experiments:
            if "baseline" not in exp.tags:
                if param in exp.config_diff:
                    val = exp.config_diff[param]["value"]
                    row += f" **{val}** |"
                elif param in exp.config_overrides:
                    val = exp.config_overrides[param]
                    row += f" **{val}** |"
                else:
                    row += f" {base_val} |"
        lines.append(row)

    lines.append("")
    lines.append("*(Bold = changed from baseline)*")
    return "\n".join(lines)


def generate_experiment_section(exp, baseline_exp, index):
    """Generate the detailed section for one experiment."""
    lines = []
    lines.append(f"### {index}. {exp.id} ({exp.run})")
    lines.append(f"**{exp.description}**")
    lines.append("")

    # Config diff
    if exp.config_diff:
        lines.append("#### Configuration Changes")
        lines.append("| Parameter | Baseline | This Run |")
        lines.append("|-----------|----------|----------|")
        for param, vals in sorted(exp.config_diff.items()):
            lines.append(f"| {param} | {vals['base']} | **{vals['value']}** |")
        lines.append("")
    elif "baseline" in exp.tags:
        lines.append("#### Configuration")
        lines.append("*Baseline config — no changes.*")
        lines.append("")
    else:
        lines.append("#### Configuration")
        lines.append("*No changes from baseline.*")
        lines.append("")

    # Hypothesis
    if exp.hypothesis:
        lines.append("#### Hypothesis")
        lines.append(exp.hypothesis)
        lines.append("")

    # Milestones
    if exp.eval_curve:
        lines.append("#### Training Milestones")
        lines.append("| Threshold | Steps | Eval WR at that point |")
        lines.append("|-----------|-------|-----------------------|")
        for t in MILESTONE_THRESHOLDS:
            step = exp.milestones.get(t)
            lines.append(f"| {t}% WR | {fmt_steps(step)} | {fmt_pct(t/100 if step else None)} |")
        lines.append("")

        # Comparison vs baseline
        if baseline_exp and baseline_exp.id != exp.id and baseline_exp.eval_curve:
            lines.append("#### Comparison vs Baseline")
            lines.append("| Metric | Baseline | This Run | Delta |")
            lines.append("|--------|----------|----------|-------|")

            for t in [50, 90, 100]:
                b_step = baseline_exp.milestones.get(t)
                e_step = exp.milestones.get(t)
                if b_step and e_step:
                    pct = (e_step - b_step) / b_step * 100
                    delta = f"{pct:+.0f}%"
                elif b_step and not e_step:
                    delta = "not reached"
                elif not b_step and e_step:
                    delta = "NEW"
                else:
                    delta = "—"
                lines.append(f"| Steps to {t}% WR | {fmt_steps(b_step)} | {fmt_steps(e_step)} | {delta} |")

            # Final WR comparison
            lines.append(
                f"| Final WR | {fmt_pct(baseline_exp.final_win_rate)} "
                f"| {fmt_pct(exp.final_win_rate)} | — |"
            )
            lines.append("")
    else:
        lines.append("#### Training Data")
        lines.append("*No evaluation data available for this run.*")
        lines.append("")

    # PPO Diagnostics
    diag_summaries = {}
    for tag in DIAG_TAGS:
        if tag in exp.diagnostics:
            s = summarize_diagnostics(exp.diagnostics[tag])
            if s:
                diag_summaries[tag] = s

    if diag_summaries:
        lines.append("#### PPO Diagnostics")
        lines.append("| Metric | Mean | Early (0-20%) | Late (80-100%) | Trend |")
        lines.append("|--------|------|---------------|----------------|-------|")
        for tag in DIAG_TAGS:
            if tag in diag_summaries:
                s = diag_summaries[tag]
                label = DIAG_LABELS.get(tag, tag)
                lines.append(
                    f"| {label} | {fmt_float(s['mean'])} "
                    f"| {fmt_float(s['early_mean'])} "
                    f"| {fmt_float(s['late_mean'])} "
                    f"| {s['trend']} |"
                )
        lines.append("")

    # Findings
    if exp.findings:
        lines.append("#### Findings")
        lines.append(exp.findings)
        lines.append("")

    # Decision
    if exp.decision:
        lines.append("#### Decision")
        lines.append(exp.decision)
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def generate_markdown(experiments, base_config, env_filter=None):
    """Generate the full EXPERIMENTS.md content."""
    if env_filter:
        experiments = [e for e in experiments if e.env == env_filter]

    lines = []
    lines.append("# Experiment Report: PPO on Unity Ball Games")
    lines.append(f"> Auto-generated by `document_experiments.py` on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(generate_summary_table(experiments))
    lines.append("")

    # Config comparison
    lines.append("## Configuration Comparison")
    lines.append("")
    lines.append(generate_config_comparison(experiments, base_config))
    lines.append("")

    # Per-experiment sections
    lines.append("## Experiments")
    lines.append("")

    # Find baselines per env
    baselines = {}
    for exp in experiments:
        if "baseline" in exp.tags and exp.env not in baselines:
            baselines[exp.env] = exp

    for i, exp in enumerate(experiments, 1):
        # Resolve baseline for comparison
        baseline_exp = None
        if exp.baseline:
            baseline_exp = next((e for e in experiments if e.id == exp.baseline), None)
        if baseline_exp is None:
            baseline_exp = baselines.get(exp.env)

        lines.append(generate_experiment_section(exp, baseline_exp, i))

    # Appendix
    lines.append("## Appendix")
    lines.append("")
    lines.append("### Base Configuration")
    lines.append("```yaml")
    for key, val in sorted(base_config.items()):
        lines.append(f"{key}: {val}")
    lines.append("```")
    lines.append("")

    lines.append("### PPO Diagnostics Glossary")
    lines.append("| Metric | What It Measures |")
    lines.append("|--------|-----------------|")
    lines.append("| Clip Fraction | Fraction of policy updates that hit the clipping bound. High = policy trying to change faster than clip_range allows. |")
    lines.append("| Approx KL | KL divergence between old and new policy. High = large policy updates per step. |")
    lines.append("| Entropy Loss | Negative entropy of the policy. Decreasing entropy = agent becoming more deterministic/confident. |")
    lines.append("| Explained Variance | How well the value function predicts returns. 1.0 = perfect, <0 = worse than mean prediction. |")
    lines.append("| Policy Gradient Loss | The PPO surrogate objective. More negative = larger policy improvement. |")
    lines.append("| Value Loss | MSE of value function predictions. Higher during active learning, lower when stable. |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Init and Add commands
# ---------------------------------------------------------------------------

def init_registry():
    """Create a template experiments.yaml with auto-discovered runs."""
    if os.path.exists(REGISTRY_PATH):
        print(f"{REGISTRY_PATH} already exists. Delete it first or edit manually.")
        return

    runs = discover_runs()
    print(f"Found {len(runs)} PPO runs:")
    for r in runs:
        print(f"  {r['env']}/{r['run']}")

    # Read base config from config.py if possible
    base_config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "net_arch": [128, 128],
        "total_timesteps": 4000000,
        "n_envs": 1,
    }

    experiments = []
    for r in runs:
        experiments.append({
            "id": f"{r['env']}_{r['run'].lower()}",
            "run": r["run"],
            "env": r["env"],
            "description": "TODO: describe this experiment",
            "config_overrides": {},
            "hypothesis": "TODO",
            "findings": "TODO",
            "decision": "TODO",
            "tags": [],
        })

    registry = {
        "base_config": base_config,
        "experiments": experiments,
    }

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated {REGISTRY_PATH} with {len(experiments)} experiments.")
    print("Edit the file to add hypotheses, findings, and decisions.")


def add_experiment():
    """Interactive: add a new experiment to the registry."""
    if not os.path.exists(REGISTRY_PATH):
        print(f"No {REGISTRY_PATH} found. Run --init first.")
        return

    with open(REGISTRY_PATH, "r") as f:
        registry = yaml.safe_load(f)

    existing_runs = {(e["env"], e["run"]) for e in registry.get("experiments", [])}
    available = [r for r in discover_runs() if (r["env"], r["run"]) not in existing_runs]

    if available:
        print("Available runs not yet documented:")
        for i, r in enumerate(available):
            print(f"  {i+1}. {r['env']}/{r['run']}")
    else:
        print("All discovered runs are already in the registry.")

    print()
    exp = {}
    exp["id"] = input("Experiment ID (e.g., hard_low_entropy): ").strip()
    exp["run"] = input("PPO run directory (e.g., PPO_5): ").strip()
    exp["env"] = input("Environment (simple/medium/hard): ").strip()
    exp["description"] = input("Description: ").strip()

    overrides_str = input("Config overrides as key=value pairs (comma-separated, or empty): ").strip()
    if overrides_str:
        exp["config_overrides"] = {}
        for pair in overrides_str.split(","):
            k, v = pair.strip().split("=")
            try:
                v = float(v)
                if v == int(v):
                    v = int(v)
            except ValueError:
                pass
            exp["config_overrides"][k.strip()] = v
    else:
        exp["config_overrides"] = {}

    exp["hypothesis"] = input("Hypothesis: ").strip()
    exp["findings"] = input("Findings: ").strip()
    exp["decision"] = input("Decision: ").strip()
    exp["tags"] = []

    baseline = input("Baseline experiment ID (or empty for default): ").strip()
    if baseline:
        exp["baseline"] = baseline

    registry["experiments"].append(exp)

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    print(f"\nAdded '{exp['id']}' to {REGISTRY_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment reports from training data and annotations")
    parser.add_argument("--init", action="store_true",
                        help="Create template experiments.yaml")
    parser.add_argument("--add", action="store_true",
                        help="Interactively add a new experiment")
    parser.add_argument("--env", type=str, default=None,
                        help="Filter report to one environment")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: EXPERIMENTS.md)")
    args = parser.parse_args()

    if args.init:
        init_registry()
        return

    if args.add:
        add_experiment()
        return

    # Generate report
    base_config, exp_entries = load_registry(REGISTRY_PATH)

    experiments = []
    for entry in exp_entries:
        if args.env and entry["env"] != args.env:
            continue
        print(f"Loading {entry['id']} ({entry['env']}/{entry['run']})...")
        exp = load_experiment(entry, base_config)
        if exp.eval_curve:
            print(f"  {len(exp.eval_curve)} eval points, "
                  f"final WR={fmt_pct(exp.final_win_rate)}, "
                  f"milestones: {', '.join(f'{t}%={fmt_steps(s)}' for t, s in exp.milestones.items() if s)}")
        else:
            print(f"  No eval data")
        experiments.append(exp)

    if not experiments:
        print("No experiments to report.")
        return

    # Generate markdown
    report = generate_markdown(experiments, base_config, env_filter=args.env)

    output_path = args.output or OUTPUT_PATH
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport written to {output_path}")
    print(f"  {len(experiments)} experiments documented")


if __name__ == "__main__":
    main()
