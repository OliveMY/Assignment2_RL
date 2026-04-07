"""Tests for config.py — config merging, deepcopy safety, LR schedule."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import get_config, get_ppo_kwargs, make_lr_schedule, BASE_CONFIG


def test_get_config_returns_expected_defaults():
    config = get_config("simple")
    assert config["learning_rate"] == 3e-4
    assert config["gamma"] == 0.99
    assert config["clip_range"] == 0.2
    assert config["n_steps"] == 2048
    assert config["batch_size"] == 64


def test_get_config_applies_env_overrides():
    config = get_config("hard")
    assert config["ent_coef"] == 0.02  # hard-specific override
    assert config["gamma"] == 0.99     # base value preserved


def test_get_config_deepcopy_isolation():
    """Mutating a returned config must not affect BASE_CONFIG."""
    config = get_config("simple")
    config["policy_kwargs"]["net_arch"] = {"pi": [999], "vf": [999]}

    # BASE_CONFIG should be untouched
    fresh = get_config("simple")
    assert fresh["policy_kwargs"]["net_arch"]["pi"] == [128, 128]


def test_get_config_unknown_env_returns_base():
    config = get_config("nonexistent")
    assert config["learning_rate"] == BASE_CONFIG["learning_rate"]


def test_lr_schedule_constant():
    config = {"learning_rate": 3e-4, "lr_schedule": "constant"}
    result = make_lr_schedule(config)
    assert result == 3e-4


def test_lr_schedule_linear_decay():
    config = {"learning_rate": 3e-4, "lr_schedule": "linear_decay"}
    schedule = make_lr_schedule(config)
    assert callable(schedule)
    assert abs(schedule(1.0) - 3e-4) < 1e-10
    assert abs(schedule(0.5) - 1.5e-4) < 1e-10
    assert abs(schedule(0.0) - 0.0) < 1e-10


def test_lr_schedule_default_is_constant():
    config = {"learning_rate": 3e-4}
    result = make_lr_schedule(config)
    assert result == 3e-4


def test_lr_schedule_invalid_raises():
    config = {"learning_rate": 3e-4, "lr_schedule": "cosine"}
    try:
        make_lr_schedule(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "cosine" in str(e)


def test_get_ppo_kwargs_uses_lr_schedule():
    config = get_config("simple")
    config["lr_schedule"] = "linear_decay"
    kwargs = get_ppo_kwargs(config)
    assert callable(kwargs["learning_rate"])


def test_get_ppo_kwargs_constant_lr():
    config = get_config("simple")
    kwargs = get_ppo_kwargs(config)
    assert kwargs["learning_rate"] == 3e-4


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
