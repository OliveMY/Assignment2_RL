# CE6127 Assignment 2 — Reinforcement Learning

PPO agent trained to play a Unity ball game at three difficulty levels (Simple, Medium, Hard) using Stable Baselines 3 and Unity ML-Agents.

## Project Structure

```
├── src/
│   ├── train.py            # Main training script (supports --env, --resume, etc.)
│   ├── train_medium.py     # Direct training script for Medium environment
│   ├── train_hard.py       # Direct training script for Hard environment
│   ├── evaluate.py         # Evaluation & live demo
│   ├── record.py           # Record gameplay videos from checkpoints
│   ├── config.py           # Hyperparameter configs per environment
│   ├── env_wrapper.py      # Gymnasium wrapper for Unity ML-Agents
│   ├── callbacks.py        # Training callbacks (logging, early stopping)
│   └── probe_env.py        # Environment introspection utility
├── Games/                  # Unity game executables (.app)
│   ├── Simple.app
│   ├── Medium.app
│   └── Hard.app
├── models/                 # Saved checkpoints and final models
│   ├── simple/
│   ├── medium/
│   └── hard/
├── results/                # Training logs and episode data
├── recordings/             # Recorded gameplay videos
└── requirements.txt
```

## Setup

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
python src/train.py --env simple --resume models/simple/checkpoint_50000_steps.zip

# Override hyperparameters
python src/train.py --env hard --timesteps 3000000 --lr 1e-4

# Run parallel training (use different worker IDs)
python src/train.py --env simple --worker-id 0 &
python src/train.py --env medium --worker-id 1 &
```

Checkpoints are saved to `models/{env}/` every 10K steps (50K for hard). Training stops early when the rolling win rate exceeds 92% (90% for hard).

### Hyperparameters

| Parameter | Simple | Medium | Hard |
|-----------|--------|--------|------|
| Total timesteps | 500K | 1M | 2M |
| Entropy coef | 0.01 | 0.015 | 0.02 |
| Network | [128, 128] | [128, 128] | [128, 128] |
| Learning rate | 3e-4 | 3e-4 | 3e-4 |
| Batch size | 64 | 64 | 64 |

All environments use PPO with MlpPolicy. See `src/config.py` for the full configuration.

## Evaluation

```bash
# Fast evaluation (no graphics, accelerated)
python src/evaluate.py --env simple --model models/simple/final.zip --rounds 100

# Live demo (graphics, real-time)
python src/evaluate.py --env simple --model models/simple/final.zip --rounds 10 --demo

# Evaluate all environments
python src/evaluate.py --env all --rounds 10
```

## Recording Videos

Record gameplay from a checkpoint as MP4 video files:

```bash
# Record 3 episodes (defaults to models/{env}/final.zip)
python src/record.py --env simple --rounds 3

# Record from a specific checkpoint
python src/record.py --env hard --model models/hard/checkpoint_550000_steps.zip --rounds 5

# Use a different worker ID if training is running concurrently
python src/record.py --env simple --rounds 3 --worker-id 1

# Capture a specific screen region
python src/record.py --env simple --rounds 2 --region "100,100,800,600"
```

Videos are saved to `recordings/{env}/`. The Unity game window should be visible on screen during recording (maximize it for best results). macOS requires Screen Recording permission for your terminal app.

## Environment

The game is a Unity ML-Agents ball game with:
- **Observation space**: 12-dimensional vector (ball positions, velocities)
- **Action space**: MultiDiscrete([3, 3]) — 9 possible actions (movement on two axes)
- **Reward**: Positive for winning, negative for losing

Use `src/probe_env.py` to inspect environment details:

```bash
python src/probe_env.py --env simple
python src/probe_env.py --env all
```
