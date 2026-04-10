"""
Probe Unity ML-Agents environments to discover observation/action spaces.
Run this FIRST before writing any training code.

Usage:
    python src/probe_env.py --env simple
    python src/probe_env.py --env all
    python src/probe_env.py --env simple --analyze   # validate reward shaping obs indices
"""
import argparse
import os
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

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


def analyze_observation_indices(env_name, env_path, behavior_name):
    """Analyze observation dimensions to identify ball, agent, and opponent indices.

    Strategy:
    1. Collect obs across many steps with controlled actions.
    2. Compare obs deltas when taking action=0 (no movement) vs action=1/2.
       Dimensions that change in response to actions are agent-controlled.
    3. Dimensions that change independently of actions are ball/opponent.
    4. Use variance and cross-step deltas to distinguish position vs velocity.
    5. Validate against reward_shaping.py default assumptions.
    """
    from mlagents_envs.base_env import ActionTuple

    print(f"\n{'='*60}")
    print(f"  OBSERVATION INDEX ANALYSIS: {env_name.upper()}")
    print(f"{'='*60}")

    engine_channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        file_name=env_path,
        side_channels=[engine_channel],
        no_graphics=True,
        worker_id=2,
    )
    engine_channel.set_configuration_parameters(time_scale=20.0)
    unity_env.reset()
    spec = unity_env.behavior_specs[behavior_name]
    action_spec = spec.action_spec
    obs_shapes = [s.shape for s in spec.observation_specs]
    total_dim = sum(s[0] for s in obs_shapes)

    # --- Phase 1: Collect obs with different action strategies ---
    # Strategy A: all-zero actions (no movement)
    # Strategy B: random actions
    # Compare per-dim variance under each to find agent-controlled dims.

    n_collect_steps = 200

    def collect_obs_with_action(unity_env, behavior_name, action_fn, n_steps):
        """Collect observations while applying a fixed action strategy."""
        all_obs = []
        steps_collected = 0
        unity_env.reset()
        while steps_collected < n_steps:
            decision_steps, terminal_steps = unity_env.get_steps(behavior_name)
            if len(decision_steps) > 0:
                obs_list = [obs[0] for obs in decision_steps.obs]
                obs_vec = np.concatenate(obs_list).astype(np.float32)
                all_obs.append(obs_vec)
                steps_collected += 1

                # Set action
                n_agents = len(decision_steps)
                action = action_fn(n_agents)
                unity_env.set_actions(behavior_name, action)

            if len(terminal_steps) > 0:
                pass  # Unity auto-resets

            unity_env.step()
        return np.array(all_obs)

    # Action strategy: no movement (action [1, 1] = center, center for MultiDiscrete([3,3]))
    def no_move_action(n):
        discrete = np.ones((n, len(action_spec.discrete_branches)), dtype=np.int32)
        return ActionTuple(discrete=discrete)

    # Action strategy: always move in one direction (action [0, 0])
    def move_action(n):
        discrete = np.zeros((n, len(action_spec.discrete_branches)), dtype=np.int32)
        return ActionTuple(discrete=discrete)

    # Action strategy: random
    def random_action(n):
        return action_spec.random_action(n)

    print(f"\n  Collecting {n_collect_steps} steps with no-movement action...")
    obs_no_move = collect_obs_with_action(unity_env, behavior_name, no_move_action, n_collect_steps)

    unity_env.reset()
    print(f"  Collecting {n_collect_steps} steps with fixed-movement action...")
    obs_move = collect_obs_with_action(unity_env, behavior_name, move_action, n_collect_steps)

    unity_env.reset()
    print(f"  Collecting {n_collect_steps} steps with random actions...")
    obs_random = collect_obs_with_action(unity_env, behavior_name, random_action, n_collect_steps)

    unity_env.close()

    # --- Phase 2: Per-dimension statistics ---
    print(f"\n  --- Per-Dimension Statistics (random actions, {n_collect_steps} steps) ---")
    print(f"  {'Dim':<5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Delta_Std':>10} {'Label'}")
    print(f"  {'-'*65}")

    deltas_random = np.diff(obs_random, axis=0)
    deltas_no_move = np.diff(obs_no_move, axis=0)
    deltas_move = np.diff(obs_move, axis=0)

    dim_labels = []
    for d in range(total_dim):
        mean_val = obs_random[:, d].mean()
        std_val = obs_random[:, d].std()
        min_val = obs_random[:, d].min()
        max_val = obs_random[:, d].max()
        delta_std = deltas_random[:, d].std()

        # Heuristic labeling
        label = ""
        # If the dimension barely changes regardless of action, it might be constant/unused
        if std_val < 0.01:
            label = "constant?"
        dim_labels.append(label)

        print(f"  [{d:<3}] {mean_val:>8.3f} {std_val:>8.3f} {min_val:>8.3f} {max_val:>8.3f} {delta_std:>10.4f} {label}")

    # --- Phase 3: Action-responsiveness analysis ---
    # Compare delta variance between no-movement and movement strategies.
    # Agent-controlled dimensions will have LARGER deltas under movement.
    print(f"\n  --- Action-Responsiveness (movement vs no-movement delta std) ---")
    print(f"  {'Dim':<5} {'NoMove_dStd':>12} {'Move_dStd':>12} {'Ratio':>8} {'Responsive?'}")
    print(f"  {'-'*55}")

    responsive_dims = []
    unresponsive_dims = []
    for d in range(total_dim):
        no_move_dstd = deltas_no_move[:, d].std()
        move_dstd = deltas_move[:, d].std()
        # Avoid division by zero
        ratio = move_dstd / no_move_dstd if no_move_dstd > 1e-6 else (
            float('inf') if move_dstd > 1e-6 else 1.0
        )
        responsive = ratio > 1.5 or (move_dstd > 0.01 and no_move_dstd < 0.001)
        tag = "AGENT" if responsive else "ball/opponent"

        if responsive:
            responsive_dims.append(d)
        else:
            unresponsive_dims.append(d)

        print(f"  [{d:<3}] {no_move_dstd:>12.5f} {move_dstd:>12.5f} {ratio:>8.2f} {tag}")

    # --- Phase 4: Position vs velocity heuristic ---
    # Velocity dimensions tend to have higher step-to-step delta variance
    # relative to their overall range, and can be negative.
    print(f"\n  --- Position vs Velocity Heuristic ---")
    print(f"  {'Dim':<5} {'Autocorr(1)':>12} {'Likely Type'}")
    print(f"  {'-'*35}")

    for d in range(total_dim):
        series = obs_random[:, d]
        if series.std() < 1e-6:
            ptype = "constant"
        else:
            # Autocorrelation at lag 1: high = position (smooth), low = velocity (noisy)
            centered = series - series.mean()
            autocorr = np.correlate(centered[:-1], centered[1:])[0] / (np.sum(centered**2) + 1e-10)
            if autocorr > 0.8:
                ptype = "position (smooth)"
            elif autocorr > 0.3:
                ptype = "position or slow-changing"
            else:
                ptype = "velocity (noisy)"
        print(f"  [{d:<3}] {autocorr if series.std() >= 1e-6 else 0:>12.4f} {ptype}")

    # --- Phase 5: Validate reward_shaping.py assumptions ---
    # Known layout from Unity source (RollerAgent.CollectObservations):
    #   [0,1] self_pos (normalized), [2,3] opp_pos (normalized),
    #   [4,5] self_vel, [6,7] opp_vel, [8,9] self_ang_vel, [10,11] opp_ang_vel
    print(f"\n  {'='*60}")
    print(f"  REWARD SHAPING INDEX VALIDATION")
    print(f"  {'='*60}")
    print(f"  Known obs layout (from Unity source):")
    print(f"    [0,1]   self position   (normalized [-1,1])")
    print(f"    [2,3]   opponent position (normalized [-1,1])")
    print(f"    [4,5]   self velocity    (raw)")
    print(f"    [6,7]   opponent velocity (raw)")
    print(f"    [8,9]   self angular vel  (raw)")
    print(f"    [10,11] opponent angular vel (raw)")
    print()
    print(f"  reward_shaping.py uses:")
    print(f"    self_pos_idx = (0, 1)")
    print(f"    opp_pos_idx  = (2, 3)")
    print(f"    distance = norm(obs[0:2] - obs[2:4])")

    self_idx = [0, 1]
    opp_idx = [2, 3]

    # Validate: position dims should have small, bounded range ~ [-1, 1]
    stds = [obs_random[:, d].std() for d in range(total_dim)]
    delta_stds = [deltas_random[:, d].std() for d in range(total_dim)]

    self_std = np.mean([stds[d] for d in self_idx])
    opp_std = np.mean([stds[d] for d in opp_idx])
    self_range = np.mean([obs_random[:, d].max() - obs_random[:, d].min() for d in self_idx])
    opp_range = np.mean([obs_random[:, d].max() - obs_random[:, d].min() for d in opp_idx])

    print(f"\n  Self position dims {self_idx}:")
    print(f"    std={self_std:.3f}, range={self_range:.3f}")
    if self_range < 4.0:
        print(f"    PASS - Bounded range, consistent with normalized position")
    else:
        print(f"    WARN - Large range ({self_range:.3f}), expected ~2.0 for [-1,1]")

    print(f"  Opponent position dims {opp_idx}:")
    print(f"    std={opp_std:.3f}, range={opp_range:.3f}")
    if opp_range < 4.0:
        print(f"    PASS - Bounded range, consistent with normalized position")
    else:
        print(f"    WARN - Large range ({opp_range:.3f}), expected ~2.0 for [-1,1]")

    # Velocity dims should have larger variance
    vel_idx = [4, 5, 6, 7]
    vel_std = np.mean([stds[d] for d in vel_idx])
    print(f"  Velocity dims {vel_idx}:")
    print(f"    std={vel_std:.3f}")
    if vel_std > self_std:
        print(f"    PASS - Higher variance than position dims (as expected)")
    else:
        print(f"    WARN - Lower variance than position dims")

    # Identify likely observation groups by variance clustering
    print(f"\n  Dim pairs sorted by scale (std):")
    pairs = []
    for d in range(0, total_dim - 1, 2):
        pair_std = (stds[d] + stds[d+1]) / 2
        pair_delta = (delta_stds[d] + delta_stds[d+1]) / 2
        pairs.append((d, d+1, pair_std, pair_delta))

    pairs_sorted = sorted(pairs, key=lambda x: x[2])
    known_labels = {
        (0, 1): "self_pos", (2, 3): "opp_pos",
        (4, 5): "self_vel", (6, 7): "opp_vel",
        (8, 9): "self_ang_vel", (10, 11): "opp_ang_vel",
    }
    for d0, d1, pstd, pdelta in pairs_sorted:
        label = known_labels.get((d0, d1), "?")
        print(f"    [{d0},{d1}]  std={pstd:.3f}  delta_std={pdelta:.3f}  ({label})")

    # --- Phase 6: Distance analysis for contact_radius tuning ---
    print(f"\n  --- Distance Statistics (for contact_radius tuning) ---")
    self_obs = obs_random[:, self_idx]
    opp_obs = obs_random[:, opp_idx]
    distances = np.linalg.norm(self_obs - opp_obs, axis=1)
    print(f"  Self-Opponent distance (random policy, normalized coords):")
    print(f"    mean={distances.mean():.3f}, std={distances.std():.3f}, "
          f"min={distances.min():.3f}, max={distances.max():.3f}")
    print(f"    10th percentile: {np.percentile(distances, 10):.3f}")
    print(f"    25th percentile: {np.percentile(distances, 25):.3f}")
    print(f"    Suggested contact_radius: {np.percentile(distances, 10):.2f} "
          f"(10th percentile of distance)")

    return {
        "responsive_dims": responsive_dims,
        "unresponsive_dims": unresponsive_dims,
        "distance_mean": float(distances.mean()),
        "distance_p10": float(np.percentile(distances, 10)),
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

    # --- Observation index analysis ---
    obs_analysis = None

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
        "obs_analysis": obs_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="Probe Unity ML-Agents environments")
    parser.add_argument("--env", type=str, default="all",
                        choices=["simple", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--analyze", action="store_true",
                        help="Run observation index analysis for reward shaping validation")
    args = parser.parse_args()

    envs = ["simple", "medium", "hard"] if args.env == "all" else [args.env]
    results = []

    for env_name in envs:
        result = probe_environment(env_name, args.episodes)
        if args.analyze:
            env_path = ENV_PATHS[env_name]
            behavior_name = result["behavior_names"][0]
            result["obs_analysis"] = analyze_observation_indices(
                env_name, env_path, behavior_name
            )
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
