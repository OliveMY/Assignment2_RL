"""
Probe Unity ML-Agents environments to discover observation/action spaces.
Run this FIRST before writing any training code.

Usage:
    python src/probe_env.py --env simple
    python src/probe_env.py --env all
"""
import argparse
import os
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_PATHS = {
    "simple": os.path.join(BASE_DIR, "Games", "Simple.app"),
    "medium": os.path.join(BASE_DIR, "Games", "Medium.app"),
    "hard": os.path.join(BASE_DIR, "Games", "Hard.app"),
}


def probe_environment(env_name: str, num_episodes: int = 20):
    """Connect to a Unity environment and discover its spaces."""
    env_path = ENV_PATHS[env_name]
    print(f"\n{'='*60}")
    print(f"Probing: {env_name.upper()} ({env_path})")
    print(f"{'='*60}")

    engine_channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        file_name=env_path,
        side_channels=[engine_channel],
        no_graphics=True,
        worker_id=0,
    )
    engine_channel.set_configuration_parameters(time_scale=20.0)

    unity_env.reset()

    # Discover behavior specs
    behavior_names = list(unity_env.behavior_specs.keys())
    print(f"\nBehavior names: {behavior_names}")

    for bname in behavior_names:
        spec = unity_env.behavior_specs[bname]
        print(f"\n--- Behavior: {bname} ---")

        # Observation specs
        print(f"  Observation specs ({len(spec.observation_specs)}):")
        for i, obs_spec in enumerate(spec.observation_specs):
            print(f"    [{i}] shape={obs_spec.shape}, type={obs_spec.observation_type}, name={obs_spec.name}")

        # Action spec
        action_spec = spec.action_spec
        print(f"  Action spec:")
        print(f"    continuous_size={action_spec.continuous_size}")
        print(f"    discrete_branches={action_spec.discrete_branches}")
        if action_spec.continuous_size > 0:
            print(f"    -> Continuous action space of size {action_spec.continuous_size}")
        if len(action_spec.discrete_branches) > 0:
            print(f"    -> Discrete branches: {action_spec.discrete_branches}")

    # Run episodes and collect statistics
    print(f"\nRunning {num_episodes} episodes with random actions...")
    rewards_all = []
    lengths_all = []

    target_behavior = behavior_names[0]
    spec = unity_env.behavior_specs[target_behavior]
    action_spec = spec.action_spec

    episode_rewards = {}
    episode_lengths = {}
    completed = 0

    while completed < num_episodes:
        decision_steps, terminal_steps = unity_env.get_steps(target_behavior)

        # Track new agents
        for agent_id in decision_steps.agent_id:
            if agent_id not in episode_rewards:
                episode_rewards[agent_id] = 0.0
                episode_lengths[agent_id] = 0

        # Generate random actions for decision-requesting agents
        n_agents = len(decision_steps)
        if n_agents > 0:
            actions = action_spec.random_action(n_agents)
            unity_env.set_actions(target_behavior, actions)

            for i, agent_id in enumerate(decision_steps.agent_id):
                episode_rewards[agent_id] += decision_steps.reward[i]
                episode_lengths[agent_id] += 1

        # Handle terminal agents
        for i, agent_id in enumerate(terminal_steps.agent_id):
            if agent_id in episode_rewards:
                final_reward = episode_rewards[agent_id] + terminal_steps.reward[i]
                final_length = episode_lengths[agent_id] + 1
                rewards_all.append(final_reward)
                lengths_all.append(final_length)
                del episode_rewards[agent_id]
                del episode_lengths[agent_id]
                completed += 1

                if completed <= 5:
                    print(f"  Episode {completed}: reward={final_reward:.3f}, length={final_length}")

        unity_env.step()

    unity_env.close()

    # Print statistics
    rewards = np.array(rewards_all)
    lengths = np.array(lengths_all)
    wins = np.sum(rewards > 0)
    losses = np.sum(rewards < 0)

    print(f"\n--- Statistics ({env_name.upper()}, {num_episodes} episodes) ---")
    print(f"  Reward: mean={rewards.mean():.3f}, std={rewards.std():.3f}, "
          f"min={rewards.min():.3f}, max={rewards.max():.3f}")
    print(f"  Length: mean={lengths.mean():.1f}, std={lengths.std():.1f}, "
          f"min={lengths.min()}, max={lengths.max()}")
    print(f"  Wins: {wins}/{num_episodes} ({100*wins/num_episodes:.1f}%)")
    print(f"  Losses: {losses}/{num_episodes} ({100*losses/num_episodes:.1f}%)")

    # Print sample observations
    print(f"\n  Sample observation (first episode, first step):")
    unity_env2 = UnityEnvironment(
        file_name=env_path,
        side_channels=[EngineConfigurationChannel()],
        no_graphics=True,
        worker_id=1,
    )
    unity_env2.reset()
    decision_steps, _ = unity_env2.get_steps(target_behavior)
    if len(decision_steps) > 0:
        for i, obs in enumerate(decision_steps.obs):
            print(f"    obs[{i}] shape={obs.shape}: {obs[0][:10]}{'...' if obs.shape[-1] > 10 else ''}")
    unity_env2.close()

    return {
        "env_name": env_name,
        "behavior_names": behavior_names,
        "obs_specs": [(s.shape, str(s.observation_type), s.name) for s in spec.observation_specs],
        "action_continuous": action_spec.continuous_size,
        "action_discrete": list(action_spec.discrete_branches),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "episode_length_mean": float(lengths.mean()),
        "win_rate": float(wins / num_episodes),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe Unity ML-Agents environments")
    parser.add_argument("--env", type=str, default="all",
                        choices=["simple", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    envs = ["simple", "medium", "hard"] if args.env == "all" else [args.env]
    results = []

    for env_name in envs:
        result = probe_environment(env_name, args.episodes)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Env':<10} {'Obs Shape':<20} {'Act (cont/disc)':<20} {'Reward':<15} {'Win%':<8} {'Ep Len':<8}")
    for r in results:
        obs_shape = str(r["obs_specs"][0][0])
        act = f"C:{r['action_continuous']}/D:{r['action_discrete']}"
        print(f"{r['env_name']:<10} {obs_shape:<20} {act:<20} "
              f"{r['reward_mean']:+.3f}±{r['reward_std']:.3f} "
              f"{100*r['win_rate']:<8.1f} {r['episode_length_mean']:<8.1f}")


if __name__ == "__main__":
    main()
