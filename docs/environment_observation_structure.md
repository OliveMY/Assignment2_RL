# RL Environment Observation Structure

## Overview

The environment is a **sumo wrestling** game built in Unity ML-Agents. Two balls (agents) compete on a square platform, each trying to push the opponent off while staying on the platform.

Unity source code: `Assignment2_v6/Assets/Scripts/` (separate Unity project, not in this repo)
Python wrapper: `src/env_wrapper.py` | Reward shaping: `src/reward_shaping.py`

## Observation Space (12 continuous values)

| Index | Observation               | Normalization                        |
|-------|---------------------------|--------------------------------------|
| 0     | Self X position           | Normalized by `arenaHalfSize` → [-1, 1] |
| 1     | Self Z position           | Normalized by `arenaHalfSize` → [-1, 1] |
| 2     | Opponent X position       | Normalized by `arenaHalfSize` → [-1, 1] |
| 3     | Opponent Z position       | Normalized by `arenaHalfSize` → [-1, 1] |
| 4     | Self X velocity           | Raw (unbounded)                      |
| 5     | Self Z velocity           | Raw (unbounded)                      |
| 6     | Opponent X velocity       | Raw (unbounded)                      |
| 7     | Opponent Z velocity       | Raw (unbounded)                      |
| 8     | Self X angular velocity   | Raw (unbounded)                      |
| 9     | Self Z angular velocity   | Raw (unbounded)                      |
| 10    | Opponent X angular velocity | Raw (unbounded)                    |
| 11    | Opponent Z angular velocity | Raw (unbounded)                    |

Observations are collected in `RollerAgent.CollectObservations()`. Positions are normalized relative to `arenaHalfSize` (default 4.0), while velocities and angular velocities are passed as raw values.

## Action Space (Discrete)

Two discrete action branches, each with 3 options:

| Branch | Axis | 0      | 1        | 2        |
|--------|------|--------|----------|----------|
| 0      | X    | No move | Negative | Positive |
| 1      | Z    | No move | Negative | Positive |

Discrete actions are converted to continuous force via `DiscreteToAxis()` and applied as:

```
rBody.AddForce(controlSignal * forceMultiplier)
```

Default `forceMultiplier = 10`.

## Reward Structure

| Event                  | Reward  |
|------------------------|---------|
| Each step              | -0.001  |
| Opponent falls off     | +1.0    |
| Agent falls off        | -1.0    |
| Timeout (max episode)  | 0.0     |

## Environment Parameters

| Parameter          | Default | Description                              |
|--------------------|---------|------------------------------------------|
| `arenaHalfSize`    | 4.0     | Half-length of the square arena (8×8)    |
| `fallYThreshold`   | 0.0     | Y position below which an agent has fallen |
| `maxEpisodeTime`   | 10.0s   | Maximum episode duration before draw     |
| `minSpawnDistance`  | 2.0     | Minimum distance between spawn positions |
| `spawnY`           | 0.5     | Y position at spawn                      |
| `forceMultiplier`  | 10.0    | Force applied per action step            |

## Collision Mechanics

Agent-agent collisions use a knockback system (`AggressiveCollision.cs`):

- **Speed threshold:** 1.0 — minimum speed difference to trigger knockback
- **Force multiplier:** 5.0 — scales the impulse based on speed advantage
- Impulse direction is along the collision normal

## Heuristic Opponents

Four built-in heuristic strategies are available for training or evaluation:

1. **PlayerInput** — manual keyboard control
2. **ConservativeCenter** — position/velocity controller pulling toward center
3. **OrbitLimitCycle** — PID-based orbital movement
4. **OscillatingAxes** — per-frame lookahead safety policy

## Reward Shaping Strategies

Five reward shaping strategies are available via `src/reward_shaping.py` (configured with `--reward-shaping` flag):

| Strategy | Signal | Outcome (vs no shaping) |
|----------|--------|-------------------------|
| `none` | Raw Unity rewards only | **Best.** Fastest, smoothest convergence |
| `pbrs_proximity` | PBRS bonus for closing distance to opponent (Ng 1999) | Slower convergence, +200K steps |
| `dense_contact` | Small bonus when within contact radius (0.5) of opponent | Noisier training, WR drops |
| `dense_alive` | Tiny per-step bonus for staying on platform | Worst — conflicts with offensive objective |
| `composite` | Weighted combination of all three | Not tested in experiments |

All strategies tested at scale=0.1. None outperformed the sparse +1/-1 reward. See EXPERIMENTS.md #7-#9 for details.

## Key Source Files

### Unity (C#)

| File                                  | Purpose                          |
|---------------------------------------|----------------------------------|
| `RollerAgent.cs`                      | Agent observations, actions, rewards, heuristics |
| `RollerGameController.cs`            | Episode lifecycle, spawn logic, reward assignment |
| `DeterministicRollerGameController.cs`| Fixed spawn positions variant    |
| `AggressiveCollision.cs`             | Collision knockback mechanics    |
| `GameStatsUI.cs`                     | Win statistics UI display        |

### Python (this repo)

| File                        | Purpose                                          |
|-----------------------------|--------------------------------------------------|
| `src/env_wrapper.py`       | Gymnasium wrapper for Unity ML-Agents environments |
| `src/reward_shaping.py`    | Reward shaping wrapper (PBRS, dense, composite)  |
| `src/config.py`            | PPO hyperparameter configuration per environment |
| `src/probe_env.py`         | Environment introspection and diagnostics        |
