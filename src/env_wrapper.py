"""
Gymnasium-compatible wrapper for Unity ML-Agents environments.

Bypasses the old UnityToGymWrapper and wraps UnityEnvironment directly,
providing the modern gymnasium API that Stable Baselines 3 expects.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import platform

if platform.system() == "Darwin":
    ENV_PATHS = {
        "simple": os.path.join(BASE_DIR, "Games", "Simple.app"),
        "medium": os.path.join(BASE_DIR, "Games", "Medium.app"),
        "hard": os.path.join(BASE_DIR, "Games", "Hard.app"),
    }
else:
    ENV_PATHS = {
        "simple": os.path.join(BASE_DIR, "Games", "Simple_Linux.x86_64"),
        "medium": os.path.join(BASE_DIR, "Games", "Medium_Linux.x86_64"),
        "hard": os.path.join(BASE_DIR, "Games", "Hard_Linux.x86_64"),
    }


class UnityGymnasiumWrapper(gym.Env):
    """Wraps a Unity ML-Agents environment as a Gymnasium environment.

    Handles the low-level Unity API (behavior specs, decision/terminal steps)
    and exposes standard reset()/step() with gymnasium conventions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_name: str,
        time_scale: float = 20.0,
        no_graphics: bool = True,
        worker_id: int = 0,
    ):
        super().__init__()
        self.env_name = env_name
        self.time_scale = time_scale
        self.no_graphics = no_graphics
        self.worker_id = worker_id

        self._engine_channel = EngineConfigurationChannel()
        self._unity_env = UnityEnvironment(
            file_name=ENV_PATHS[env_name],
            side_channels=[self._engine_channel],
            no_graphics=no_graphics,
            worker_id=worker_id,
        )
        self._engine_channel.set_configuration_parameters(time_scale=time_scale)
        self._unity_env.reset()

        # Discover behavior spec (use first behavior)
        self._behavior_name = list(self._unity_env.behavior_specs.keys())[0]
        self._spec = self._unity_env.behavior_specs[self._behavior_name]

        # Build observation space: concatenate all observation specs
        obs_shapes = [obs_spec.shape for obs_spec in self._spec.observation_specs]
        total_obs_size = sum(s[0] for s in obs_shapes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        # Build action space
        action_spec = self._spec.action_spec
        if action_spec.continuous_size > 0 and len(action_spec.discrete_branches) == 0:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(action_spec.continuous_size,),
                dtype=np.float32,
            )
            self._action_type = "continuous"
        elif action_spec.continuous_size == 0 and len(action_spec.discrete_branches) > 0:
            if len(action_spec.discrete_branches) == 1:
                self.action_space = spaces.Discrete(action_spec.discrete_branches[0])
            else:
                self.action_space = spaces.MultiDiscrete(action_spec.discrete_branches)
            self._action_type = "discrete"
        else:
            # Hybrid: continuous + discrete — treat as continuous for simplicity
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(action_spec.continuous_size,),
                dtype=np.float32,
            )
            self._action_type = "continuous"

        self._agent_id = None
        self._episode_reward = 0.0
        self._episode_length = 0

    def _get_obs(self, steps, agent_idx: int) -> np.ndarray:
        """Concatenate all observation tensors for a single agent."""
        obs_list = [obs[agent_idx] for obs in steps.obs]
        return np.concatenate(obs_list).astype(np.float32)

    def _make_action(self, action):
        """Convert gymnasium action to Unity ActionTuple."""
        from mlagents_envs.base_env import ActionTuple

        action_spec = self._spec.action_spec
        if self._action_type == "continuous":
            continuous = np.array(action, dtype=np.float32).reshape(1, -1)
            return ActionTuple(continuous=continuous)
        else:
            if isinstance(action, (int, np.integer)):
                discrete = np.array([[action]], dtype=np.int32)
            else:
                discrete = np.array(action, dtype=np.int32).reshape(1, -1)
            return ActionTuple(discrete=discrete)

    def _make_episode_info(self):
        """Build the episode info dict that SB3's Monitor/callbacks expect."""
        return {"episode": {"r": self._episode_reward, "l": self._episode_length}}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._unity_env.reset()
        self._episode_reward = 0.0
        self._episode_length = 0

        # Get first decision step
        decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)

        # Wait until we have a decision-requesting agent
        while len(decision_steps) == 0:
            self._unity_env.step()
            decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)

        self._agent_id = decision_steps.agent_id[0]
        obs = self._get_obs(decision_steps, 0)
        return obs, {}

    def step(self, action):
        # Set action for our agent
        action_tuple = self._make_action(action)
        self._unity_env.set_actions(self._behavior_name, action_tuple)
        self._unity_env.step()

        # Check results
        decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)

        # Check if episode terminated
        if len(terminal_steps) > 0:
            agent_idx = None
            for i, aid in enumerate(terminal_steps.agent_id):
                if aid == self._agent_id:
                    agent_idx = i
                    break

            if agent_idx is not None:
                obs = self._get_obs(terminal_steps, agent_idx)
                reward = float(terminal_steps.reward[agent_idx])
                self._episode_reward += reward
                self._episode_length += 1
                info = self._make_episode_info()
                return obs, reward, True, False, info

        # Check if our agent needs a new decision
        if len(decision_steps) > 0:
            agent_idx = None
            for i, aid in enumerate(decision_steps.agent_id):
                if aid == self._agent_id:
                    agent_idx = i
                    break

            if agent_idx is not None:
                obs = self._get_obs(decision_steps, agent_idx)
                reward = float(decision_steps.reward[agent_idx])
                self._episode_reward += reward
                self._episode_length += 1
                return obs, reward, False, False, {}

        # Agent not found in either — Unity auto-reset consumed the terminal step.
        # The previous episode ended but we missed it. Treat as termination.
        ep_info = self._make_episode_info()

        # Wait for the new agent from the next episode
        while len(decision_steps) == 0:
            self._unity_env.step()
            decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)
            if len(terminal_steps) > 0:
                obs = self._get_obs(terminal_steps, 0)
                reward = float(terminal_steps.reward[0])
                return obs, reward, True, False, ep_info

        # New episode started — return terminal for the OLD episode
        self._agent_id = decision_steps.agent_id[0]
        obs = self._get_obs(decision_steps, 0)
        # Reset counters for the new episode
        self._episode_reward = 0.0
        self._episode_length = 0
        return obs, 0.0, True, False, ep_info

    def close(self):
        self._unity_env.close()

    def render(self):
        pass  # Rendering is handled by Unity when no_graphics=False


def make_env(env_name: str, time_scale: float = 20.0, no_graphics: bool = True,
             worker_id: int = 0, norm_path: str = None,
             reward_shaping_config: dict = None) -> UnityGymnasiumWrapper:
    """Factory function for creating wrapped Unity environments.

    Args:
        norm_path: If provided, wraps the environment in VecNormalize and loads
            saved normalization stats from this path. Use for evaluating models
            that were trained with observation normalization.
        reward_shaping_config: If provided, wraps with RewardShapingWrapper.
            Dict is passed as kwargs, e.g. {"strategy": "pbrs_proximity", "shaping_scale": 0.1}.
    """
    env = UnityGymnasiumWrapper(
        env_name=env_name,
        time_scale=time_scale,
        no_graphics=no_graphics,
        worker_id=worker_id,
    )
    if reward_shaping_config and reward_shaping_config.get("strategy", "none") != "none":
        from reward_shaping import RewardShapingWrapper
        env = RewardShapingWrapper(env, **reward_shaping_config)
    if norm_path is not None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        return vec_env
    return env


def make_vec_env(env_name: str, n_envs: int = 4, time_scale: float = 20.0,
                 no_graphics: bool = True, base_worker_id: int = 0,
                 reward_shaping_config: dict = None):
    """Create a SubprocVecEnv with n_envs parallel Unity environments.

    Each sub-environment gets a unique worker_id (base_worker_id + i)
    so they bind to different gRPC ports.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def _make_env_fn(wid):
        def _init():
            env = UnityGymnasiumWrapper(
                env_name=env_name,
                time_scale=time_scale,
                no_graphics=no_graphics,
                worker_id=wid,
            )
            if reward_shaping_config and reward_shaping_config.get("strategy", "none") != "none":
                from reward_shaping import RewardShapingWrapper
                return RewardShapingWrapper(env, **reward_shaping_config)
            return env
        return _init

    env_fns = [_make_env_fn(base_worker_id + i) for i in range(n_envs)]
    return SubprocVecEnv(env_fns)
