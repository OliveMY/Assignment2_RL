"""
PPO diagnostics callback for capturing internal training statistics.

Hooks into _on_rollout_end to read PPO's internal metrics (clip_fraction,
approx_kl, explained_variance, etc.) from the logger after each train() call.

Note: _on_rollout_end fires after collect_rollouts() but before train(),
so the values read are from the PREVIOUS training update (one-step lag).
This is fine for tracking trends across thousands of updates.

Output format: JSONL (one JSON object per line), crash-safe via append.
"""
import json
import os

from stable_baselines3.common.callbacks import BaseCallback


# Keys that SB3's PPO writes to the logger during train()
PPO_DIAGNOSTIC_KEYS = [
    "train/clip_fraction",
    "train/approx_kl",
    "train/explained_variance",
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
]


class DiagnosticsCallback(BaseCallback):
    """Captures PPO internal training stats and writes them as JSONL.

    Each line in the output file is a JSON object with the timestep
    and all available PPO diagnostic values from that training update.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self._log_path = os.path.join(log_dir, "diagnostics.jsonl")
        self._file = None

    def _on_training_start(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._file = open(self._log_path, "a")
        if self.verbose:
            print(f"[Diagnostics] Logging to {self._log_path}")

    def _on_rollout_end(self):
        logger_values = getattr(self.model.logger, "name_to_value", {})
        if not logger_values:
            return

        entry = {"step": self.num_timesteps}
        for key in PPO_DIAGNOSTIC_KEYS:
            val = logger_values.get(key)
            if val is not None:
                short_key = key.split("/", 1)[1]
                entry[short_key] = float(val)

        if len(entry) > 1:  # has at least one diagnostic value beyond "step"
            line = json.dumps(entry)
            self._file.write(line + "\n")
            self._file.flush()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()
            self._file = None
        if self.verbose:
            print(f"[Diagnostics] Saved to {self._log_path}")
