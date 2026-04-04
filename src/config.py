"""
Hyperparameter configurations for each environment.

All choices are documented here for the assignment report.
Using the same algorithm (PPO) across all environments,
with environment-specific tuning.

=== Design Choices ===
Algorithm: PPO (Proximal Policy Optimization)
  - Why: Standard for ML-Agents, stable training, works well with both
    continuous and discrete action spaces, good sample efficiency for
    on-policy methods.

Policy: MlpPolicy (Multi-Layer Perceptron)
  - Why: Observation space is a low-dimensional vector (ball positions,
    velocities), not images. MLP is appropriate and fast to train.

Network: [128, 128] hidden layers
  - Why: Two hidden layers provide enough capacity for this task.
    128 units per layer is reasonable for 12-dim observation vector.

Action Space: Discrete MultiDiscrete([3, 3])
  - Two movement axes, 3 choices each (left/none/right, forward/none/back)
  - Both branches have 3 options → 9 possible actions total
"""

# Base config shared across all environments
BASE_CONFIG = {
    # PPO hyperparameters
    "learning_rate": 3e-4,
    "n_steps": 2048,           # Steps per rollout before update
    "batch_size": 64,          # Minibatch size for PPO updates
    "n_epochs": 10,            # Number of passes through rollout buffer
    "gamma": 0.99,             # Discount factor
    "gae_lambda": 0.95,        # GAE lambda for advantage estimation
    "clip_range": 0.2,         # PPO clipping parameter
    "ent_coef": 0.01,          # Entropy coefficient (encourages exploration)
    "vf_coef": 0.5,            # Value function loss coefficient
    "max_grad_norm": 0.5,      # Gradient clipping
    "policy_kwargs": {
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    },
    # Training
    "device": "auto",          # Will use MPS on Apple Silicon if beneficial
    # Environment
    "time_scale": 20.0,        # Unity time acceleration for training
    "no_graphics": True,       # Disable rendering during training
}

# Per-environment configs (override base)
ENV_CONFIGS = {
    "simple": {
        "total_timesteps": 1_000_000,
        # Simple opponent: likely predictable pattern
        # Standard params should suffice
    },
    "medium": {
        "total_timesteps": 2_000_000,
        # Medium opponent: more varied behavior
        # Slightly more exploration might help
        "ent_coef": 0.015,
    },
    "hard": {
        "total_timesteps": 4_000_000,
        # Hard opponent: complex/adaptive behavior
        # More exploration + longer training
        "ent_coef": 0.02,
    },
}


def get_config(env_name: str) -> dict:
    """Get the full config for an environment, merging base + overrides."""
    config = BASE_CONFIG.copy()
    config["policy_kwargs"] = BASE_CONFIG["policy_kwargs"].copy()
    if env_name in ENV_CONFIGS:
        overrides = ENV_CONFIGS[env_name]
        config.update(overrides)
    return config


def get_ppo_kwargs(config: dict) -> dict:
    """Extract only the kwargs that PPO.__init__ accepts."""
    ppo_keys = [
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "max_grad_norm", "policy_kwargs", "device",
    ]
    return {k: config[k] for k in ppo_keys if k in config}
