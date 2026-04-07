"""
Custom Stable Baselines 3 callbacks for training logging and monitoring.

Provides:
- TrainingLogCallback: logs per-episode stats, win rates, saves config JSON
- WinRateStoppingCallback: early stopping when win rate is high enough
"""
import json
import os
import time
import tempfile
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLogCallback(BaseCallback):
    """Logs training progress with episode-level statistics.

    Tracks:
    - Per-episode cumulative reward
    - Rolling win rate (reward > 0 = win)
    - Training speed (steps/second)
    - Saves a training_config.json at the start of training
    """

    def __init__(
        self,
        log_dir: str,
        config: dict,
        env_name: str,
        print_freq: int = 50,
        rolling_window: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.config = config
        self.env_name = env_name
        self.print_freq = print_freq
        self.rolling_window = rolling_window

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []  # 1=win, 0=loss, 0.5=draw
        self.start_time = None
        self._episode_count = 0

    def _on_training_start(self):
        self.start_time = time.time()
        os.makedirs(self.log_dir, exist_ok=True)

        # Save training config as JSON
        config_path = os.path.join(self.log_dir, "training_config.json")
        config_to_save = {
            "env_name": self.env_name,
            "algorithm": "PPO",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": {k: str(v) if not isinstance(v, (int, float, str, bool, list)) else v
                                for k, v in self.config.items()},
        }
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)

        if self.verbose:
            print(f"\n[TrainingLog] Config saved to {config_path}")
            print(f"[TrainingLog] Training {self.env_name} with PPO")
            print(f"[TrainingLog] Total timesteps: {self.config.get('total_timesteps', 'N/A')}")

    def _on_step(self):
        # Check for completed episodes in the info buffer
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                if ep_reward > 0:
                    outcome = 1    # win
                elif ep_reward < 0:
                    outcome = 0    # loss
                else:
                    outcome = 0.5  # draw
                self.episode_outcomes.append(outcome)
                self._episode_count += 1

                # Log to TensorBoard
                self.logger.record("rollout/ep_reward", ep_reward)
                self.logger.record("rollout/ep_length", ep_length)
                self.logger.record("rollout/win", 1 if ep_reward > 0 else 0)

                # Rolling win rate
                if len(self.episode_outcomes) >= 10:
                    recent = self.episode_outcomes[-self.rolling_window:]
                    win_rate = np.mean(recent)
                    self.logger.record("rollout/win_rate", win_rate)

                # Print periodic summary
                if self.verbose and self._episode_count % self.print_freq == 0:
                    self._print_summary()

        return True

    def _print_summary(self):
        elapsed = time.time() - self.start_time
        steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0

        recent_rewards = self.episode_rewards[-self.rolling_window:]
        recent_outcomes = self.episode_outcomes[-self.rolling_window:]

        print(f"\n[{self.env_name}] Episode {self._episode_count} | "
              f"Step {self.num_timesteps:,} | "
              f"Speed: {steps_per_sec:.0f} steps/s")
        print(f"  Reward (last {len(recent_rewards)}): "
              f"mean={np.mean(recent_rewards):.3f}, "
              f"std={np.std(recent_rewards):.3f}")
        print(f"  Win rate (last {len(recent_outcomes)}): "
              f"{100*np.mean(recent_outcomes):.1f}%")

    def _on_training_end(self):
        if self.verbose and len(self.episode_rewards) > 0:
            print(f"\n{'='*50}")
            print(f"Training Complete: {self.env_name}")
            print(f"{'='*50}")
            print(f"  Total episodes: {self._episode_count}")
            print(f"  Total timesteps: {self.num_timesteps:,}")
            print(f"  Final win rate (last 100): "
                  f"{100*np.mean(self.episode_outcomes[-100:]):.1f}%")
            print(f"  Best rolling win rate: "
                  f"{100*max(np.mean(self.episode_outcomes[max(0,i-100):i]) for i in range(10, len(self.episode_outcomes)+1)):.1f}%"
                  if len(self.episode_outcomes) >= 10 else "N/A")
            elapsed = time.time() - self.start_time
            print(f"  Training time: {elapsed/60:.1f} minutes")

        # Save episode log
        log_path = os.path.join(self.log_dir, "episode_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "outcomes": self.episode_outcomes,
            }, f)


class EvalCallback(BaseCallback):
    """Periodically evaluate the model by playing games in a separate environment.

    Every eval_freq timesteps, saves the current model, loads it in a fresh
    environment, and plays n_eval_episodes games to measure win rate.
    Results are logged to TensorBoard and saved as JSON.
    """

    def __init__(
        self,
        env_name: str,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 50,
        log_dir: str = ".",
        worker_id: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env_name = env_name
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir
        self.worker_id = worker_id
        self._last_eval_step = 0

    def _on_step(self):
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        self._run_eval()
        return True

    def _run_eval(self):
        import tempfile
        from stable_baselines3 import PPO
        from env_wrapper import make_env

        if self.verbose:
            print(f"\n[Eval] Running {self.n_eval_episodes}-game evaluation "
                  f"at step {self.num_timesteps:,}...")

        # Save current model to a temp file
        tmp_path = os.path.join(self.log_dir, "_eval_tmp_model")
        self.model.save(tmp_path)

        # Create a separate eval environment (with retry for transient Unity spawn failures)
        env = None
        for attempt in range(3):
            try:
                env = make_env(
                    env_name=self.env_name,
                    time_scale=20.0,
                    no_graphics=True,
                    worker_id=self.worker_id,
                )
                break
            except Exception as e:
                if attempt < 2:
                    if self.verbose:
                        print(f"[Eval] Unity spawn failed (attempt {attempt+1}/3): {e}. Retrying in 5s...")
                    import time
                    time.sleep(5)
                else:
                    if self.verbose:
                        print(f"[Eval] Unity spawn failed after 3 attempts. Skipping eval at step {self.num_timesteps}.")
                    return

        eval_model = PPO.load(tmp_path)

        wins, losses, draws = 0, 0, 0
        rewards = []

        try:
            for _ in range(self.n_eval_episodes):
                obs, info = env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    action, _ = eval_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated

                rewards.append(episode_reward)
                if episode_reward > 0:
                    wins += 1
                elif episode_reward < 0:
                    losses += 1
                else:
                    draws += 1
        finally:
            env.close()
            # Clean up temp model file
            tmp_file = tmp_path + ".zip"
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        win_rate = wins / self.n_eval_episodes
        avg_reward = float(np.mean(rewards))

        # Log to TensorBoard
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/avg_reward", avg_reward)
        self.logger.record("eval/wins", wins)
        self.logger.record("eval/losses", losses)
        self.logger.record("eval/draws", draws)

        # Save JSON
        results = {
            "step": self.num_timesteps,
            "env_name": self.env_name,
            "n_episodes": self.n_eval_episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
        }
        results_path = os.path.join(
            self.log_dir, f"eval_step_{self.num_timesteps}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"[Eval] Step {self.num_timesteps:,}: "
                  f"Win rate {100*win_rate:.1f}% "
                  f"({wins}W/{losses}L/{draws}D), "
                  f"Avg reward {avg_reward:+.3f}")
            print(f"[Eval] Results saved to {results_path}")


class WinRateStoppingCallback(BaseCallback):
    """Stop training early when rolling win rate exceeds a threshold."""

    def __init__(
        self,
        win_rate_threshold: float = 0.90,
        min_episodes: int = 200,
        rolling_window: int = 200,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.win_rate_threshold = win_rate_threshold
        self.min_episodes = min_episodes
        self.rolling_window = rolling_window
        self.episode_outcomes = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = info["episode"]["r"]
                self.episode_outcomes.append(1 if r > 0 else (0 if r < 0 else 0.5))

        if len(self.episode_outcomes) >= self.min_episodes:
            recent = self.episode_outcomes[-self.rolling_window:]
            win_rate = np.mean(recent)
            if win_rate >= self.win_rate_threshold:
                if self.verbose:
                    print(f"\n[EarlyStop] Win rate {100*win_rate:.1f}% >= "
                          f"{100*self.win_rate_threshold:.1f}% threshold. Stopping.")
                return False

        return True
