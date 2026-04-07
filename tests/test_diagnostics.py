"""Tests for diagnostics_callback.py — JSONL output, schema, edge cases."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diagnostics_callback import DiagnosticsCallback, PPO_DIAGNOSTIC_KEYS


class FakeLogger:
    """Mimics SB3's Logger.name_to_value dict."""
    def __init__(self, values=None):
        self.name_to_value = values or {}


class FakeModel:
    """Minimal mock of an SB3 model for testing the callback."""
    def __init__(self, logger_values=None):
        self.logger = FakeLogger(logger_values)


def test_diagnostics_writes_valid_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = DiagnosticsCallback(log_dir=tmpdir, verbose=0)
        cb.model = FakeModel({
            "train/clip_fraction": 0.15,
            "train/approx_kl": 0.008,
            "train/explained_variance": 0.72,
            "train/entropy_loss": -1.2,
            "train/policy_gradient_loss": -0.03,
            "train/value_loss": 0.45,
        })
        cb.num_timesteps = 2048

        # Simulate lifecycle
        cb._on_training_start()
        cb._on_rollout_end()
        cb._on_training_end()

        # Verify file
        path = os.path.join(tmpdir, "diagnostics.jsonl")
        assert os.path.exists(path)

        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["step"] == 2048
        assert abs(entry["clip_fraction"] - 0.15) < 1e-10
        assert abs(entry["approx_kl"] - 0.008) < 1e-10
        assert abs(entry["explained_variance"] - 0.72) < 1e-10


def test_diagnostics_handles_empty_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = DiagnosticsCallback(log_dir=tmpdir, verbose=0)
        cb.model = FakeModel({})  # empty logger
        cb.num_timesteps = 0

        cb._on_training_start()
        cb._on_rollout_end()
        cb._on_training_end()

        path = os.path.join(tmpdir, "diagnostics.jsonl")
        assert os.path.exists(path)
        with open(path, "r") as f:
            content = f.read().strip()
        assert content == ""  # no entries written


def test_diagnostics_handles_partial_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = DiagnosticsCallback(log_dir=tmpdir, verbose=0)
        cb.model = FakeModel({
            "train/clip_fraction": 0.1,
            # other keys missing
        })
        cb.num_timesteps = 4096

        cb._on_training_start()
        cb._on_rollout_end()
        cb._on_training_end()

        path = os.path.join(tmpdir, "diagnostics.jsonl")
        with open(path, "r") as f:
            entry = json.loads(f.readline())
        assert entry["step"] == 4096
        assert entry["clip_fraction"] == 0.1
        assert "approx_kl" not in entry  # missing key not written


def test_diagnostics_multiple_rollouts():
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = DiagnosticsCallback(log_dir=tmpdir, verbose=0)
        cb.model = FakeModel({"train/clip_fraction": 0.1})

        cb._on_training_start()

        cb.num_timesteps = 2048
        cb._on_rollout_end()

        cb.model.logger.name_to_value["train/clip_fraction"] = 0.2
        cb.num_timesteps = 4096
        cb._on_rollout_end()

        cb._on_training_end()

        path = os.path.join(tmpdir, "diagnostics.jsonl")
        with open(path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["clip_fraction"] == 0.1
        assert json.loads(lines[1])["clip_fraction"] == 0.2


def test_diagnostics_schema_keys():
    """Verify the expected diagnostic keys match SB3's PPO logger keys."""
    expected_short_keys = [
        "clip_fraction", "approx_kl", "explained_variance",
        "entropy_loss", "policy_gradient_loss", "value_loss",
    ]
    for key in PPO_DIAGNOSTIC_KEYS:
        short = key.split("/", 1)[1]
        assert short in expected_short_keys, f"Unexpected key: {key}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
