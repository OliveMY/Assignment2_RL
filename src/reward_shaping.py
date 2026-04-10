"""
Reward shaping wrapper for Unity sumo wrestling environments.

The game: two balls on a square platform, each trying to push the opponent
off while staying on. Rewards from Unity are sparse (+1 opponent falls,
-1 agent falls, -0.001/step). Shaping adds dense intermediate signals.

Strategies:
- none: Pass through raw rewards unchanged (baseline).
- pbrs_proximity: PBRS using agent-opponent distance.
    F(s,s') = gamma * Phi(s') - Phi(s), where Phi(s) = -dist(self, opponent).
    Rewards closing distance to engage. Preserves optimal policy (Ng 1999).
- dense_contact: Small bonus when agent is close to opponent (engaging).
- dense_alive: Tiny per-step bonus for staying on the platform.
- composite: Weighted combination of pbrs_proximity + dense_contact + dense_alive.

Usage:
    env = UnityGymnasiumWrapper(...)
    env = RewardShapingWrapper(env, strategy="pbrs_proximity", shaping_scale=0.1)
    env = Monitor(env)  # SB3 Monitor sees shaped rewards
"""
import numpy as np
import gymnasium as gym


# Observation indices (from Unity source: RollerAgent.CollectObservations)
#
# This is a SUMO WRESTLING game: two balls on a platform, push opponent off.
# Observations (12 dims):
#   [0] self_x       (normalized by arenaHalfSize, range [-1, 1])
#   [1] self_z       (normalized)
#   [2] opponent_x   (normalized)
#   [3] opponent_z   (normalized)
#   [4] self_vel_x   (raw)
#   [5] self_vel_z   (raw)
#   [6] opp_vel_x    (raw)
#   [7] opp_vel_z    (raw)
#   [8] self_ang_vel_x  (raw)
#   [9] self_ang_vel_z  (raw)
#   [10] opp_ang_vel_x  (raw)
#   [11] opp_ang_vel_z  (raw)
#
# Rewards from Unity: -0.001/step, +1 opponent falls, -1 agent falls, 0 timeout.
SELF_POS_IDX = (0, 1)
OPP_POS_IDX = (2, 3)
SELF_VEL_IDX = (4, 5)
OPP_VEL_IDX = (6, 7)

STRATEGIES = ["none", "pbrs_proximity", "dense_contact", "dense_alive", "composite"]


class RewardShapingWrapper(gym.Wrapper):
    """Gymnasium wrapper that adds shaped rewards on top of raw environment rewards.

    Sits between UnityGymnasiumWrapper and SB3's Monitor in the wrapper chain.
    Tracks both raw and shaped rewards separately for logging.
    """

    def __init__(
        self,
        env: gym.Env,
        strategy: str = "none",
        gamma: float = 0.99,
        shaping_scale: float = 0.1,
        # Observation index params
        self_pos_idx: tuple = SELF_POS_IDX,
        opp_pos_idx: tuple = OPP_POS_IDX,
        # Dense contact params
        contact_radius: float = 0.5,
        contact_reward: float = 0.05,
        # Dense alive params
        alive_reward: float = 0.001,
        # Composite weights
        proximity_weight: float = 1.0,
        contact_weight: float = 1.0,
        alive_weight: float = 1.0,
    ):
        super().__init__(env)
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown reward shaping strategy: {strategy!r}. "
                f"Available: {STRATEGIES}"
            )
        self.strategy = strategy
        self.gamma = gamma
        self.shaping_scale = shaping_scale

        self.self_pos_idx = list(self_pos_idx)
        self.opp_pos_idx = list(opp_pos_idx)
        self.contact_radius = contact_radius
        self.contact_reward = contact_reward
        self.alive_reward = alive_reward

        self.proximity_weight = proximity_weight
        self.contact_weight = contact_weight
        self.alive_weight = alive_weight

        self._prev_obs = None
        self._raw_episode_reward = 0.0
        self._shaping_episode_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs.copy()
        self._raw_episode_reward = 0.0
        self._shaping_episode_reward = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping_bonus = self._compute_shaping(self._prev_obs, obs, terminated)
        self._prev_obs = obs.copy()

        self._raw_episode_reward += reward
        self._shaping_episode_reward += shaping_bonus

        info["raw_reward"] = reward
        info["shaping_reward"] = shaping_bonus

        if terminated or truncated:
            info["raw_episode_reward"] = self._raw_episode_reward
            info["shaping_episode_reward"] = self._shaping_episode_reward

        return obs, reward + shaping_bonus, terminated, truncated, info

    def _compute_shaping(self, prev_obs, obs, terminated):
        """Compute the shaping bonus for the current transition."""
        if self.strategy == "none" or prev_obs is None:
            return 0.0

        if self.strategy == "pbrs_proximity":
            return self._pbrs_proximity(prev_obs, obs, terminated)
        elif self.strategy == "dense_contact":
            return self._dense_contact(obs, terminated)
        elif self.strategy == "dense_alive":
            return self._dense_alive(terminated)
        elif self.strategy == "composite":
            return self._composite(prev_obs, obs, terminated)
        return 0.0

    # --- Potential-based reward shaping ---

    def _self_opp_distance(self, obs):
        """Compute distance between self and opponent."""
        self_pos = obs[self.self_pos_idx]
        opp_pos = obs[self.opp_pos_idx]
        return np.linalg.norm(self_pos - opp_pos)

    def _phi_proximity(self, obs):
        """Potential: negative distance from self to opponent.

        Higher (less negative) when closer to opponent = rewards engaging.
        """
        return -self._self_opp_distance(obs)

    def _pbrs_proximity(self, prev_obs, obs, terminated):
        """PBRS using agent-ball proximity potential.

        F(s, s') = gamma * Phi(s') - Phi(s)
        At terminal states, Phi(s') = 0 by convention.
        """
        phi_prev = self._phi_proximity(prev_obs)
        phi_next = 0.0 if terminated else self._phi_proximity(obs)
        return self.shaping_scale * (self.gamma * phi_next - phi_prev)

    # --- Dense strategies ---

    def _dense_contact(self, obs, terminated):
        """Small bonus when close to opponent (engaging in combat)."""
        if terminated:
            return 0.0
        dist = self._self_opp_distance(obs)
        if dist < self.contact_radius:
            return self.shaping_scale * self.contact_reward
        return 0.0

    def _dense_alive(self, terminated):
        """Tiny per-step bonus for keeping the episode going."""
        if terminated:
            return 0.0
        return self.shaping_scale * self.alive_reward

    # --- Composite ---

    def _composite(self, prev_obs, obs, terminated):
        """Weighted combination of all shaping signals."""
        total = 0.0
        if self.proximity_weight > 0:
            total += self.proximity_weight * self._pbrs_proximity(prev_obs, obs, terminated)
        if self.contact_weight > 0:
            total += self.contact_weight * self._dense_contact(obs, terminated)
        if self.alive_weight > 0:
            total += self.alive_weight * self._dense_alive(terminated)
        return total
