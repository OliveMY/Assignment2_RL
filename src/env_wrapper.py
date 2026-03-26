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

ENV_PATHS = {
    "simple": os.path.join(BASE_DIR, "Games", "Simple.app"),
    "medium": os.path.join(BASE_DIR, "Games", "Medium.app"),
    "hard": os.path.join(BASE_DIR, "Games", "Hard.app"),
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._unity_env.reset()

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
            # Find our agent in terminal steps
            agent_idx = None
            for i, aid in enumerate(terminal_steps.agent_id):
                if aid == self._agent_id:
                    agent_idx = i
                    break

            if agent_idx is not None:
                obs = self._get_obs(terminal_steps, agent_idx)
                reward = float(terminal_steps.reward[agent_idx])
                return obs, reward, True, False, {}

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
                return obs, reward, False, False, {}

        # Agent not found in either — environment may have auto-reset
        # Try to get new agent
        while len(decision_steps) == 0:
            self._unity_env.step()
            decision_steps, terminal_steps = self._unity_env.get_steps(self._behavior_name)
            if len(terminal_steps) > 0:
                obs = self._get_obs(terminal_steps, 0)
                reward = float(terminal_steps.reward[0])
                return obs, reward, True, False, {}

        self._agent_id = decision_steps.agent_id[0]
        obs = self._get_obs(decision_steps, 0)
        reward = float(decision_steps.reward[0])
        return obs, reward, False, False, {}

    def close(self):
        self._unity_env.close()

    def render(self):
        pass  # Rendering is handled by Unity when no_graphics=False


def make_env(env_name: str, time_scale: float = 20.0, no_graphics: bool = True,
             worker_id: int = 0) -> UnityGymnasiumWrapper:
    """Factory function for creating wrapped Unity environments."""
    return UnityGymnasiumWrapper(
        env_name=env_name,
        time_scale=time_scale,
        no_graphics=no_graphics,
        worker_id=worker_id,
    )
