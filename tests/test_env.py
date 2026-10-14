"""Fast pytest checks for the Milestone 3 Gymnasium environment."""
import numpy as np
import pytest

from env.biped_env import BipedEnv


@pytest.fixture
def env():
    e = BipedEnv(seed=0)
    e.reset(seed=0)
    return e


def test_spaces(env):
    assert env.observation_space.shape == (47,)
    assert env.action_space.shape == (12,)


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (47,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert "cmd" in info


def test_step_returns_valid_tuple(env):
    obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
    assert obs.shape == (47,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_terms" in info


def test_determinism():
    a = np.full(12, 0.1, dtype=np.float32)

    def rollout():
        e = BipedEnv(seed=5)
        obs, _ = e.reset(seed=5)
        for _ in range(5):
            obs, *_ = e.step(a)
        return obs

    np.testing.assert_array_equal(rollout(), rollout())


def test_zero_action_eventually_terminates(env):
    zero = np.zeros(12, dtype=np.float32)
    for _ in range(300):
        _, _, terminated, truncated, _ = env.step(zero)
        if terminated or truncated:
            return
    pytest.fail("episode never ended under zero action (expected a fall ~1.8s in)")


def test_reset_can_pin_command(env):
    _, info = env.reset(seed=0, options={"cmd": [0.3, 0.0, 0.0]})
    np.testing.assert_allclose(info["cmd"], [0.3, 0.0, 0.0])
    assert env._stepping()


def test_lin_vel_sigma_only_narrows_forward_tracking():
    from env.rewards import compute_reward
    state = {
        "base_linvel": np.zeros(3), "base_angvel_z": 0.0, "cmd": np.array([0.2, 0.0, 0.0]),
        "projected_gravity": np.array([0.0, 0.0, -1.0]), "base_height": 1.0, "height_target": 1.0,
        "torque": np.zeros(12), "torque_limit": np.ones(12), "action": np.zeros(12),
        "prev_action": np.zeros(12), "qvel": np.zeros(12), "prev_qvel": np.zeros(12), "dt": 0.02,
        "foot_contact": [True, True], "foot_xy_vel": [np.zeros(2)] * 2,
        "non_foot_contact": False, "terminated_by_fall": False,
    }
    w = {"lin_vel_track": 1.0, "upright": 1.0}
    _, wide = compute_reward(state, w, 0.25, 10.0)
    _, narrow = compute_reward(state, w, 0.25, 10.0, lin_vel_sigma=0.02)
    assert wide["lin_vel_track"] == pytest.approx(np.exp(-0.04 / 0.25))
    assert narrow["lin_vel_track"] == pytest.approx(np.exp(-0.04 / 0.02))
    assert wide["upright"] == narrow["upright"]
