"""Tests for the RewardShapingWrapper."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reward_shaping import RewardShapingWrapper, STRATEGIES


class MockEnv(gym.Env):
    """Minimal gym env for testing reward shaping.

    Obs is 12-dim: [ball_pos(3), ball_vel(3), agent_pos(3), agent_vel(3)].
    Actions are MultiDiscrete([3, 3]).

    Call set_sequence(reset_obs, step_obs_list, reward_list, terminated_list)
    to program the env. reset() returns reset_obs. step() returns from the
    step sequences starting at index 0.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3, 3])
        self._step_count = 0
        self._reset_obs = np.zeros(12, dtype=np.float32)
        self._step_obs = []
        self._reward_sequence = []
        self._terminated_sequence = []

    def set_sequence(self, reset_obs, step_obs_list, reward_list, terminated_list):
        """Pre-program env outputs.

        Args:
            reset_obs: observation returned by reset()
            step_obs_list: observations returned by successive step() calls
            reward_list: rewards returned by successive step() calls
            terminated_list: terminated flags returned by successive step() calls
        """
        self._reset_obs = reset_obs
        self._step_obs = step_obs_list
        self._reward_sequence = reward_list
        self._terminated_sequence = terminated_list
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return self._reset_obs.copy(), {}

    def step(self, action):
        idx = min(self._step_count, len(self._step_obs) - 1)
        obs = self._step_obs[idx]
        reward = self._reward_sequence[idx]
        terminated = self._terminated_sequence[idx]
        self._step_count += 1
        return obs, reward, terminated, False, {}


def _make_obs(self_pos=(0, 0), opp_pos=(0, 0), self_vel=(0, 0),
              opp_vel=(0, 0), self_ang_vel=(0, 0), opp_ang_vel=(0, 0)):
    """Helper to build a 12-dim sumo wrestling obs vector.

    Layout: [self_x, self_z, opp_x, opp_z, self_vx, self_vz,
             opp_vx, opp_vz, self_angvx, self_angvz, opp_angvx, opp_angvz]
    Positions are normalized by arenaHalfSize (range [-1,1]).
    """
    return np.array(
        list(self_pos) + list(opp_pos) + list(self_vel) +
        list(opp_vel) + list(self_ang_vel) + list(opp_ang_vel),
        dtype=np.float32,
    )


class TestRewardShapingNone:
    """Test that strategy='none' is a pure passthrough."""

    def test_passthrough_reward(self):
        env = MockEnv()
        obs1 = _make_obs(self_pos=(0, 0), opp_pos=(0.5, 0))
        obs2 = _make_obs(self_pos=(0.1, 0), opp_pos=(0.5, 0))
        env.set_sequence(obs1, [obs1, obs2], [0.0, 1.0], [False, True])

        wrapped = RewardShapingWrapper(env, strategy="none")
        wrapped.reset()

        _, reward, _, _, info = wrapped.step(0)
        assert reward == 0.0
        assert info["raw_reward"] == 0.0
        assert info["shaping_reward"] == 0.0

        _, reward, _, _, info = wrapped.step(0)
        assert reward == 1.0

    def test_no_episode_info_on_nonterminal(self):
        env = MockEnv()
        obs = _make_obs()
        env.set_sequence(obs, [obs], [0.0], [False])
        wrapped = RewardShapingWrapper(env, strategy="none")
        wrapped.reset()
        _, _, _, _, info = wrapped.step(0)
        assert "raw_episode_reward" not in info


class TestPBRSProximity:
    """Test potential-based reward shaping with proximity potential.

    Phi(s) = -dist(self, opponent) = -norm(obs[0:2] - obs[2:4]).
    Rewards closing distance to engage the opponent.
    """

    def test_closing_distance_positive(self):
        """Getting closer to opponent = positive shaping."""
        # Start: self at origin, opp at (0.5, 0) -> dist=0.5
        obs_reset = _make_obs(self_pos=(0, 0), opp_pos=(0.5, 0))
        # After: self moved toward opp -> dist=0.2
        obs1 = _make_obs(self_pos=(0.3, 0), opp_pos=(0.5, 0))

        env = MockEnv()
        env.set_sequence(obs_reset, [obs1], [0.0], [False])

        wrapped = RewardShapingWrapper(
            env, strategy="pbrs_proximity", gamma=0.99, shaping_scale=1.0
        )
        wrapped.reset()

        _, reward, _, _, info = wrapped.step(0)
        # Phi(reset) = -0.5, Phi(obs1) = -0.2
        # F = 0.99 * (-0.2) - (-0.5) = 0.302
        expected = 0.99 * (-0.2) - (-0.5)
        assert info["shaping_reward"] == pytest.approx(expected, abs=1e-4)
        assert reward == pytest.approx(expected, abs=1e-4)

    def test_retreating_negative(self):
        """Moving away from opponent = negative shaping."""
        obs_reset = _make_obs(self_pos=(0.3, 0), opp_pos=(0.5, 0))  # dist=0.2
        obs1 = _make_obs(self_pos=(0, 0), opp_pos=(0.5, 0))          # dist=0.5

        env = MockEnv()
        env.set_sequence(obs_reset, [obs1], [0.0], [False])

        wrapped = RewardShapingWrapper(
            env, strategy="pbrs_proximity", gamma=0.99, shaping_scale=1.0
        )
        wrapped.reset()

        _, reward, _, _, info = wrapped.step(0)
        # Phi(reset) = -0.2, Phi(obs1) = -0.5
        # F = 0.99 * (-0.5) - (-0.2) = -0.295
        expected = 0.99 * (-0.5) - (-0.2)
        assert info["shaping_reward"] < 0
        assert info["shaping_reward"] == pytest.approx(expected, abs=1e-4)

    def test_terminal_phi_zero(self):
        """At terminal state, Phi(s') = 0."""
        obs_reset = _make_obs(self_pos=(0, 0), opp_pos=(0.5, 0))  # dist=0.5
        obs_term = _make_obs(self_pos=(0.4, 0), opp_pos=(0.5, 0))  # dist=0.1

        env = MockEnv()
        env.set_sequence(obs_reset, [obs_term], [1.0], [True])

        wrapped = RewardShapingWrapper(
            env, strategy="pbrs_proximity", gamma=0.99, shaping_scale=1.0
        )
        wrapped.reset()

        _, reward, terminated, _, info = wrapped.step(0)
        assert terminated
        # Phi(prev) = -0.5, Phi(terminal) = 0
        # F = 0.99 * 0 - (-0.5) = 0.5
        expected_shaping = 0.99 * 0.0 - (-0.5)
        assert info["shaping_reward"] == pytest.approx(expected_shaping, abs=1e-4)
        assert reward == pytest.approx(1.0 + expected_shaping, abs=1e-4)

    def test_shaping_scale(self):
        """Shaping scale multiplies the bonus."""
        obs_reset = _make_obs(self_pos=(0, 0), opp_pos=(0.5, 0))  # dist=0.5
        obs1 = _make_obs(self_pos=(0.3, 0), opp_pos=(0.5, 0))     # dist=0.2

        env = MockEnv()
        env.set_sequence(obs_reset, [obs1], [0.0], [False])

        wrapped = RewardShapingWrapper(
            env, strategy="pbrs_proximity", gamma=0.99, shaping_scale=0.1
        )
        wrapped.reset()

        _, _, _, _, info = wrapped.step(0)
        unscaled = 0.99 * (-0.2) - (-0.5)
        assert info["shaping_reward"] == pytest.approx(0.1 * unscaled, abs=1e-4)


class TestDenseContact:
    """Test dense contact reward strategy (close to opponent = bonus)."""

    def test_close_to_opponent_gets_bonus(self):
        obs_reset = _make_obs(self_pos=(0.3, 0), opp_pos=(0.5, 0))  # dist=0.2
        obs1 = _make_obs(self_pos=(0.3, 0), opp_pos=(0.5, 0))

        env = MockEnv()
        env.set_sequence(obs_reset, [obs1], [0.0], [False])

        wrapped = RewardShapingWrapper(
            env, strategy="dense_contact", shaping_scale=1.0,
            contact_radius=0.5, contact_reward=0.05,
        )
        wrapped.reset()

        _, reward, _, _, info = wrapped.step(0)
        assert info["shaping_reward"] == pytest.approx(0.05)
        assert reward == pytest.approx(0.05)

    def test_far_from_opponent_no_bonus(self):
        obs_reset = _make_obs(self_pos=(-0.5, 0), opp_pos=(0.5, 0))  # dist=1.0
        obs1 = _make_obs(self_pos=(-0.5, 0), opp_pos=(0.5, 0))

        env = MockEnv()
        env.set_sequence(obs_reset, [obs1], [0.0], [False])

        wrapped = RewardShapingWrapper(
            env, strategy="dense_contact", shaping_scale=1.0,
            contact_radius=0.5, contact_reward=0.05,
        )
        wrapped.reset()

        _, reward, _, _, info = wrapped.step(0)
        assert info["shaping_reward"] == 0.0

    def test_no_bonus_on_terminal(self):
        obs_reset = _make_obs(self_pos=(0.4, 0), opp_pos=(0.5, 0))  # dist=0.1
        obs_term = _make_obs(self_pos=(0.4, 0), opp_pos=(0.5, 0))

        env = MockEnv()
        env.set_sequence(obs_reset, [obs_term], [1.0], [True])

        wrapped = RewardShapingWrapper(
            env, strategy="dense_contact", shaping_scale=1.0,
            contact_radius=0.5, contact_reward=0.05,
        )
        wrapped.reset()

        _, _, _, _, info = wrapped.step(0)
        assert info["shaping_reward"] == 0.0


class TestDenseAlive:
    """Test dense alive reward strategy."""

    def test_alive_bonus_nonterminal(self):
        env = MockEnv()
        obs = _make_obs()
        env.set_sequence(obs, [obs], [0.0], [False])

        wrapped = RewardShapingWrapper(
            env, strategy="dense_alive", shaping_scale=1.0, alive_reward=0.001
        )
        wrapped.reset()

        _, reward, _, _, info = wrapped.step(0)
        assert info["shaping_reward"] == pytest.approx(0.001)

    def test_no_alive_bonus_terminal(self):
        env = MockEnv()
        obs = _make_obs()
        env.set_sequence(obs, [obs], [1.0], [True])

        wrapped = RewardShapingWrapper(
            env, strategy="dense_alive", shaping_scale=1.0, alive_reward=0.001
        )
        wrapped.reset()

        _, _, _, _, info = wrapped.step(0)
        assert info["shaping_reward"] == 0.0


class TestComposite:
    """Test composite strategy combining all signals."""

    def test_composite_sums_components(self):
        # Close to opponent and closing distance
        obs_reset = _make_obs(self_pos=(0.2, 0), opp_pos=(0.5, 0))  # dist=0.3
        obs1 = _make_obs(self_pos=(0.4, 0), opp_pos=(0.5, 0))       # dist=0.1

        env = MockEnv()
        env.set_sequence(obs_reset, [obs1], [0.0], [False])

        wrapped = RewardShapingWrapper(
            env, strategy="composite", gamma=0.99, shaping_scale=1.0,
            contact_radius=0.5, contact_reward=0.05, alive_reward=0.001,
            proximity_weight=1.0, contact_weight=1.0, alive_weight=1.0,
        )
        wrapped.reset()

        _, reward, _, _, info = wrapped.step(0)

        # PBRS: Phi(reset) = -0.3, Phi(obs1) = -0.1
        # F = 0.99 * (-0.1) - (-0.3) = 0.201
        pbrs = 0.99 * (-0.1) - (-0.3)
        # Dense contact: dist = 0.1 < 0.5, so +0.05
        contact = 0.05
        # Dense alive: +0.001
        alive = 0.001
        expected = pbrs + contact + alive
        assert info["shaping_reward"] == pytest.approx(expected, abs=1e-3)


class TestEpisodeTracking:
    """Test that raw and shaped episode rewards are tracked correctly."""

    def test_episode_reward_accumulation(self):
        obs_reset = _make_obs(self_pos=(0, 0), opp_pos=(0.5, 0))    # dist=0.5
        obs2 = _make_obs(self_pos=(0.2, 0), opp_pos=(0.5, 0))       # dist=0.3
        obs_term = _make_obs(self_pos=(0.4, 0), opp_pos=(0.5, 0))   # dist=0.1

        env = MockEnv()
        env.set_sequence(
            obs_reset,
            [obs2, obs_term, obs_term],
            [0.0, 0.0, 1.0],
            [False, False, True],
        )

        wrapped = RewardShapingWrapper(
            env, strategy="pbrs_proximity", gamma=0.99, shaping_scale=1.0
        )
        wrapped.reset()

        # Step through episode
        wrapped.step(0)
        wrapped.step(0)
        _, _, _, _, info = wrapped.step(0)

        # Episode ended — check tracking
        assert "raw_episode_reward" in info
        assert info["raw_episode_reward"] == pytest.approx(1.0)  # 0 + 0 + 1
        assert "shaping_episode_reward" in info
        # Shaping reward should be nonzero (agent moved toward ball)
        assert info["shaping_episode_reward"] != 0.0

    def test_reset_clears_accumulators(self):
        env = MockEnv()
        obs = _make_obs()
        env.set_sequence(obs, [obs], [0.5], [True])

        wrapped = RewardShapingWrapper(env, strategy="dense_alive", shaping_scale=1.0)
        wrapped.reset()
        wrapped.step(0)

        # Reset should clear
        wrapped.reset()
        assert wrapped._raw_episode_reward == 0.0
        assert wrapped._shaping_episode_reward == 0.0


class TestInvalidStrategy:
    """Test error handling for invalid strategy."""

    def test_unknown_strategy_raises(self):
        env = MockEnv()
        with pytest.raises(ValueError, match="Unknown reward shaping strategy"):
            RewardShapingWrapper(env, strategy="invalid_strategy")


class TestStrategyList:
    """Test that STRATEGIES constant is correct."""

    def test_all_strategies_listed(self):
        expected = {"none", "pbrs_proximity", "dense_contact", "dense_alive", "composite"}
        assert set(STRATEGIES) == expected
