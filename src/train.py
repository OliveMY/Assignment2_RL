"""
Main training script for PPO agents on Unity ball game environments.

Usage:
    python src/train.py --env simple
    python src/train.py --env medium --timesteps 1000000
    python src/train.py --env hard --timesteps 2000000 --time-scale 20.0
    python src/train.py --env simple --resume models/simple/checkpoint_100000_steps.zip

Training logs are saved to results/{env_name}/ and models to models/{env_name}/.
"""
import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from config import get_config, get_ppo_kwargs
from env_wrapper import make_env, make_vec_env
from callbacks import TrainingLogCallback, WinRateStoppingCallback, EvalCallback
from diagnostics_callback import DiagnosticsCallback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train(args):
    config = get_config(args.env)

    # Set random seed for reproducibility
    if args.seed is not None:
        set_random_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    # CLI overrides
    if args.timesteps:
        config["total_timesteps"] = args.timesteps
    if args.time_scale:
        config["time_scale"] = args.time_scale
    if args.lr:
        config["learning_rate"] = args.lr
    if args.ent_coef is not None:
        config["ent_coef"] = args.ent_coef
    if args.lr_schedule:
        config["lr_schedule"] = args.lr_schedule
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.reward_shaping:
        config["reward_shaping"] = args.reward_shaping
    if args.shaping_scale is not None:
        config["shaping_scale"] = args.shaping_scale

    total_timesteps = config.pop("total_timesteps")
    time_scale = config.pop("time_scale", 20.0)
    no_graphics = config.pop("no_graphics", True)
    config["total_timesteps"] = total_timesteps  # keep for callback logging

    log_dir = os.path.join(BASE_DIR, "results", args.env)
    os.makedirs(log_dir, exist_ok=True)

    # Determine the next PPO_N run number (matches TensorBoard's auto-naming)
    existing_runs = [d for d in os.listdir(log_dir)
                     if d.startswith("PPO_") and os.path.isdir(os.path.join(log_dir, d))]
    run_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
    next_run = max(run_numbers, default=0) + 1
    run_name = f"PPO_{next_run}"

    # Save checkpoints per-experiment so runs don't overwrite each other
    model_dir = os.path.join(BASE_DIR, "models", args.env, run_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Run: {run_name} | Checkpoints: {model_dir}")

    # Build reward shaping config
    shaping_strategy = config.pop("reward_shaping", "none")
    shaping_scale = config.pop("shaping_scale", 0.1)
    reward_shaping_config = None
    if shaping_strategy != "none":
        reward_shaping_config = {
            "strategy": shaping_strategy,
            "shaping_scale": shaping_scale,
            "gamma": config.get("gamma", 0.99),
        }
        print(f"Reward shaping: {shaping_strategy} (scale={shaping_scale})")

    # Create environment
    n_envs = args.n_envs
    if n_envs > 1:
        print(f"Creating {n_envs} parallel {args.env} environments (time_scale={time_scale})...")
        env = make_vec_env(
            env_name=args.env,
            n_envs=n_envs,
            time_scale=time_scale,
            no_graphics=no_graphics,
            base_worker_id=args.worker_id,
            reward_shaping_config=reward_shaping_config,
        )
    else:
        print(f"Creating {args.env} environment (time_scale={time_scale})...")
        env = make_env(
            env_name=args.env,
            time_scale=time_scale,
            no_graphics=no_graphics,
            worker_id=args.worker_id,
            reward_shaping_config=reward_shaping_config,
        )
        env = Monitor(env)

    # Create or load model
    ppo_kwargs = get_ppo_kwargs(config)
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, **ppo_kwargs)
    else:
        print(f"Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **ppo_kwargs,
        )

    # Print model summary
    print(f"\n--- Model Summary ---")
    print(f"  Algorithm: PPO")
    print(f"  Policy: MlpPolicy")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Network: {config.get('policy_kwargs', {})}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Device: {model.device}")
    print(f"  Key hyperparams: lr={config['learning_rate']}, "
          f"ent_coef={config['ent_coef']}, gamma={config['gamma']}")

    # Callbacks
    # save_freq is in _on_step calls; with n_envs, each call = n_envs timesteps
    checkpoint_freq = max(1, 50_000 // n_envs)
    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=model_dir,
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        TrainingLogCallback(
            log_dir=log_dir,
            config=config,
            env_name=args.env,
            print_freq=50,
        ),
        DiagnosticsCallback(
            log_dir=log_dir,
            verbose=1,
        ),
    ]
    if not args.skip_eval:
        callbacks.append(EvalCallback(
            env_name=args.env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            log_dir=log_dir,
            worker_id=args.worker_id + 100,
            stop_on_win_rate=args.stop_win_rate,
        ))
    if not args.no_early_stop:
        callbacks.append(WinRateStoppingCallback(
            win_rate_threshold=0.92,
            min_episodes=200,
        ))

    # Train
    print(f"\nStarting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        # Save final model
        final_path = os.path.join(model_dir, "final")
        model.save(final_path)
        print(f"Model saved to {final_path}.zip")
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Unity ball game")
    parser.add_argument("--env", type=str, required=True,
                        choices=["simple", "medium", "hard"])
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--time-scale", type=float, default=None,
                        help="Unity time scale (default: 20.0)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="Override entropy coefficient")
    parser.add_argument("--lr-schedule", type=str, default=None,
                        choices=["constant", "linear_decay"],
                        help="Learning rate schedule (default: constant)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override minibatch size")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Unity worker ID (use different IDs for parallel training)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel Unity environments (default: 1)")
    parser.add_argument("--reward-shaping", type=str, default=None,
                        choices=["none", "pbrs_proximity", "dense_contact",
                                 "dense_alive", "composite"],
                        help="Reward shaping strategy (default: from config)")
    parser.add_argument("--shaping-scale", type=float, default=None,
                        help="Scale factor for shaping rewards (default: 0.1)")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping on win rate")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="Evaluate every N timesteps (default: 50000)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of games per evaluation (default: 50)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Disable periodic evaluation")
    parser.add_argument("--stop-win-rate", type=float, default=None,
                        help="Stop training when eval win rate reaches this threshold for 3 consecutive evals")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
