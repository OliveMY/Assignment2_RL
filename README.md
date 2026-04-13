# CE6127 Assignment 2 — Reinforcement Learning

PPO agent trained to play a Unity sumo wrestling ball game at three difficulty levels (Simple, Medium, Hard) using Stable Baselines 3 and Unity ML-Agents. The agent learns to push an opponent off a platform while staying on itself.

All three environments reach **100% win rate** with the optimized configuration:
- **Simple**: 262K steps (~6 min)
- **Medium**: 918K steps (~29 min)
- **Hard**: 1.31M steps (~29 min)

## Project Structure

```
├── src/
│   ├── train.py                # Main training script (supports --env, --resume, etc.)
│   ├── train_medium.py         # Shorthand training for Medium environment
│   ├── train_hard.py           # Shorthand training for Hard environment
│   ├── evaluate.py             # Evaluation & live demo
│   ├── demo.py                 # Interactive live demonstration across all envs
│   ├── record.py               # Record gameplay videos from checkpoints
│   ├── config.py               # Hyperparameter configs per environment
│   ├── env_wrapper.py          # Gymnasium wrapper for Unity ML-Agents
│   ├── callbacks.py            # Training callbacks (logging, early stopping, eval)
│   ├── diagnostics_callback.py # PPO diagnostic metrics (clip_fraction, KL, entropy)
│   ├── reward_shaping.py       # Reward shaping strategies (PBRS, dense, composite)
│   ├── ablation.py             # Single-variable ablation runner
│   ├── plot_ablation.py        # Ablation result analysis & visualization
│   ├── sweep.py                # Grid search over hyperparameters
│   ├── sweep_analysis.py       # Sweep result analysis
│   ├── document_experiments.py # Experiment registry & EXPERIMENTS.md generator
│   ├── export_onnx.py          # Export SB3 models to ONNX for Unity inference
│   ├── probe_env.py            # Environment introspection utility
│   └── diagnose_deadlock.py    # Debugging utility for environment lockups
├── tests/
│   ├── test_config.py          # Config system tests
│   ├── test_diagnostics.py     # Diagnostics callback tests
│   └── test_reward_shaping.py  # Reward shaping tests
├── Games/                      # Unity game executables (.app on macOS)
│   ├── Simple.app
│   ├── Medium.app
│   └── Hard.app
├── models/                     # Saved checkpoints and final models
│   ├── simple/
│   ├── medium/
│   ├── hard/
│   └── onnx/                   # Exported ONNX models for Unity
├── results/                    # Training logs, TensorBoard events, episode data
│   ├── simple/
│   ├── medium/
│   ├── hard/
│   └── sweep/
├── recordings/                 # Recorded gameplay videos (MP4)
├── docs/
│   ├── environment_observation_structure.md  # Observation/action/reward specs
│   └── DEMO_BUILD_GUIDE.md                  # Guide for building Windows demo
├── Ref_doc/                    # Reference documentation
├── EXPERIMENTS.md              # Comprehensive experiment registry (14 experiments)
├── PLAN.md                     # Ablation methodology guide
├── experiments.yaml            # Experiment metadata in YAML format
├── requirements.txt            # Python dependencies
└── setup.sh                    # Automated setup script
```

## Setup

### Automated

```bash
chmod +x setup.sh
./setup.sh
```

This creates a virtual environment, installs dependencies, sets permissions on Unity builds, and verifies the installation. Requires Python 3.9-3.12.

### Manual

```bash
pip install -r requirements.txt
```

Dependencies: `mlagents-envs==0.28.0`, `stable-baselines3[extra]>=2.3.0`, `gymnasium`, `tensorboard`, `numpy`, `mss`, `imageio[ffmpeg]`.

The Unity game executables (`Games/*.app`) must be present but are not tracked in git.

## Training

```bash
# Train on a specific environment
python src/train.py --env simple
python src/train.py --env medium
python src/train.py --env hard

# Resume from a checkpoint
python src/train.py --env simple --resume models/simple/PPO_1/checkpoint_50000_steps.zip

# Override hyperparameters
python src/train.py --env hard --lr 5e-4 --lr-schedule linear_decay --batch-size 256

# Parallel environments (faster, smoother gradients)
python src/train.py --env hard --n-envs 32

# Reward shaping experiments
python src/train.py --env hard --reward-shaping pbrs_proximity --shaping-scale 0.1

# Custom evaluation settings
python src/train.py --env hard --eval-freq 50000 --eval-episodes 50 --stop-win-rate 1.0

# Seed for reproducibility
python src/train.py --env hard --seed 0
```

Checkpoints are saved to `models/{env}/PPO_N/` every ~50K steps. Training stops early when the rolling win rate exceeds 92%.

### Hyperparameters

**Base configuration** (shared across all environments):

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| N steps | 2048 |
| Batch size | 64 |
| N epochs | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| VF coef | 0.5 |
| Max grad norm | 0.5 |
| Network | [128, 128] (policy and value) |

**Per-environment overrides:**

| Parameter | Simple | Medium | Hard |
|-----------|--------|--------|------|
| Total timesteps | 1M | 2M | 4M |
| Entropy coef | 0.01 | 0.015 | 0.02 |

**Optimized configuration** (from 14 experiments on Hard, transferred to all envs):

| Parameter | Baseline | Optimized |
|-----------|----------|-----------|
| Learning rate | 3e-4 | 5e-4 |
| LR schedule | constant | linear_decay |
| Batch size | 64 | 256 |
| N envs | 1 | 32 |

See `src/config.py` for the full configuration and [EXPERIMENTS.md](EXPERIMENTS.md) for the complete experiment history.

## Evaluation & Visualization

```bash
# Fast evaluation (no graphics, accelerated) — prints win rate and stats
python src/evaluate.py --env simple --model models/simple/final.zip --rounds 100

# Live demo mode (graphics, real-time) — opens Unity window
python src/evaluate.py --env simple --model models/simple/final.zip --rounds 10 --demo

# Evaluate all environments (uses models/{env}/final.zip by default)
python src/evaluate.py --env all --rounds 10
```

### Live Demo

The `demo.py` script runs a presentation-ready demo across all environments with real-time graphics:

```bash
# Demo all environments (Simple → Medium → Hard), 10 rounds each
python src/demo.py

# Demo a single environment
python src/demo.py --env hard --rounds 5
```

## Ablation & Hyperparameter Search

### Single-variable ablation

```bash
# List available parameters
python src/ablation.py --list

# Run ablation with multiple seeds
python src/ablation.py --param clip_range --env simple --seeds 0 42 123

# Dry run to preview experiments
python src/ablation.py --param clip_range --env simple --dry-run

# Analyze and plot results
python src/plot_ablation.py --param clip_range --env simple
python src/plot_ablation.py --all --env simple
```

### Grid search

```bash
python src/sweep.py --env simple
python src/sweep_analysis.py --env simple
```

## Recording Videos

Record gameplay from a checkpoint as MP4 video files:

```bash
# Record 3 episodes from the final model
python src/record.py --env simple --model models/simple/final.zip --rounds 3

# Record from a specific checkpoint
python src/record.py --env hard --model models/hard/PPO_16/final.zip --rounds 5

# Capture a specific screen region
python src/record.py --env simple --model models/simple/final.zip --rounds 2 --region "100,100,800,600"
```

Videos are saved to `recordings/{env}/`. The Unity game window should be visible on screen during recording. macOS requires Screen Recording permission for your terminal app.

## ONNX Export

Export trained models to ONNX format for standalone Unity inference (no Python needed):

```bash
# Export all three models
python src/export_onnx.py --env all

# Export directly into Unity project
python src/export_onnx.py --env all --output-dir /path/to/Unity/Assets/Models
```

See [docs/DEMO_BUILD_GUIDE.md](docs/DEMO_BUILD_GUIDE.md) for building standalone Windows demo executables.

## Monitoring

```bash
# Monitor training with TensorBoard
tensorboard --logdir results/
```

TensorBoard tracks learning curves, PPO diagnostics (clip_fraction, approx_kl, entropy_loss, explained_variance), episode rewards, and win rates.

## Environment

The game is a Unity ML-Agents sumo wrestling ball game:
- **Observation space**: 12-dimensional vector (self/opponent positions, velocities, angular velocities)
- **Action space**: MultiDiscrete([3, 3]) — 9 possible actions (movement on two axes)
- **Reward**: +1 for pushing opponent off, -1 for falling off, -0.001 per step

Three difficulty levels correspond to different heuristic opponents (ConservativeCenter, OrbitLimitCycle, OscillatingAxes).

See [docs/environment_observation_structure.md](docs/environment_observation_structure.md) for the full specification.

### Environment Probing

```bash
python src/probe_env.py --env simple
python src/probe_env.py --env all
```

## Tests

```bash
python -m pytest tests/
```

## Experiment Documentation

The project includes a comprehensive experiment tracking system:

```bash
# Generate EXPERIMENTS.md from experiments.yaml + TensorBoard data
python src/document_experiments.py

# Add a new experiment interactively
python src/document_experiments.py --add
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full registry of 14 experiments documenting the progression from baseline to optimized configuration.
