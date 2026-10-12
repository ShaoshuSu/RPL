"""CPU tests for the FAR-FALCON-Terrains observation parity presets / terms."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from holosoma.utils.safe_torch_import import torch

from rpl_training.config_values.observation import (
    g1_29dof_falcon_hm_depth_distill_falcon,
    g1_29dof_falcon_hm_history_wolinvel,
    g1_29dof_falcon_hm_history_wolinvel_falcon,
    g1_29dof_falcon_hm_history_wolinvel_falcon_hm_noise,
)
from rpl_training.terms.observation import command_base_height, ee_apply_force


def test_ee_apply_force_returns_cached_base_frame_forces():
    # The force manager caches the forces already rotated into the base frame; a non-identity
    # base yaw must not rotate them a second time.
    s = 0.5**0.5
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        base_quat=torch.tensor([[0.0, 0.0, s, s]]),  # +90 deg yaw, w last
        left_ee_apply_force=torch.tensor([[1.0, 2.0, -30.0]]),
        right_ee_apply_force=torch.tensor([[-0.5, 0.25, -10.0]]),
    )
    torch.testing.assert_close(ee_apply_force(env), torch.tensor([[1.0, 2.0, -30.0, -0.5, 0.25, -10.0]]))
    assert ee_apply_force(SimpleNamespace(num_envs=3, device="cpu")).shape == (3, 6)


def test_command_base_height_absolute():
    cmds = torch.zeros(2, 9)
    cmds[1, 8] = -0.2
    env = SimpleNamespace(command_manager=SimpleNamespace(commands=cmds), command_base_height_nominal=0.75)
    torch.testing.assert_close(command_base_height(env), torch.tensor([[0.0], [-0.2]]))
    torch.testing.assert_close(command_base_height(env, absolute=True), torch.tensor([[0.75], [0.55]]))


def test_falcon_expert_obs_preset():
    base = g1_29dof_falcon_hm_history_wolinvel.groups
    groups = g1_29dof_falcon_hm_history_wolinvel_falcon.groups
    assert list(groups) == list(base)
    for name, group in groups.items():
        # FALCON never clears history on reset; layout and history lengths are unchanged.
        assert group.reset_history_on_env_reset is False, name
        assert list(group.terms) == list(base[name].terms), name
        assert group.history_length == base[name].history_length, name
    actor, critic = groups["actor_state_obs"].terms, groups["critic_state_obs"].terms
    assert "command_stand" in actor
    assert actor["command_base_height"].params == {"absolute": True}
    assert critic["command_base_height"].params == {"absolute": True}
    assert groups["critic_state_obs"].enable_noise
    assert (critic["dof_pos"].noise, critic["dof_vel"].noise) == (0.001, 0.01)
    # IMU noise of FALCON's final expert stages; critic at noise_level 0.1.
    assert (actor["base_ang_vel"].noise, actor["projected_gravity"].noise) == (0.2, 0.05)
    assert critic["base_ang_vel"].noise == pytest.approx(0.02)
    assert critic["projected_gravity"].noise == pytest.approx(0.005)
    noisy = ("dof_pos", "dof_vel", "base_ang_vel", "projected_gravity")
    assert all(t.noise == 0.0 for k, t in critic.items() if k not in noisy)
    assert groups["actor_state_obs"].terms["dof_vel"].noise == 0.1
    for name in ("perception_obs", "perception_obs_wo_noise"):
        assert groups[name].terms["height_measurements"].noise == 0.0
    # Stones / slopes experts: heightmap noise 0.1, critic heightmap at noise_level 0.1.
    hm = g1_29dof_falcon_hm_history_wolinvel_falcon_hm_noise.groups
    assert hm["perception_obs"].terms["height_measurements"].noise == 0.1
    assert hm["perception_obs_wo_noise"].enable_noise
    assert hm["perception_obs_wo_noise"].terms["height_measurements"].noise == pytest.approx(0.01)
    assert hm["actor_state_obs"] == groups["actor_state_obs"]
    # The distill teacher obs keep FALCON's dagger_moe settings (no IMU noise).
    teacher = g1_29dof_falcon_hm_depth_distill_falcon.groups["actor_teacher_state_obs"].terms
    assert (teacher["base_ang_vel"].noise, teacher["projected_gravity"].noise) == (0.0, 0.0)
    # Base preset untouched.
    assert base["actor_state_obs"].reset_history_on_env_reset is True
    assert base["actor_state_obs"].terms["command_base_height"].params in (None, {})
