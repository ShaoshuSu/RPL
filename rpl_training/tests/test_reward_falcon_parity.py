"""FAR-FALCON-Terrains parity of the loco-manip reward terms used by the FALCON RL experts (no simulator)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from holosoma.managers.reward.terms import locomotion as loco_terms
from holosoma.utils.safe_torch_import import torch

from rpl_training.config_values.reward import (
    g1_29dof_falcon_hm_reward,
    g1_29dof_falcon_hm_reward_falcon,
)
from rpl_training.terms import reward as lm

TERRAINS = ["slopes", "stairs_up", "stairs_down", "stepping_stones_columns"]
DOF_NAMES = [f"lower_{i}" for i in range(15)] + [f"upper_{i}" for i in range(14)]


class _TerrainState:
    def __init__(self, idx, base_heights=None, heights=None, levels=None):
        self.terrain_type_names = TERRAINS
        self.terrain_type_idx = idx
        n = idx.shape[0]
        self.base_heights = torch.full((n,), 0.75) if base_heights is None else base_heights
        self._heights = torch.zeros(n, 4) if heights is None else heights
        self.terrain_levels = torch.zeros(n, dtype=torch.long) if levels is None else levels
        self.max_terrain_level = 10

    def height_measurements(self):
        return self._heights


def _make_env(terrain_idx, **terrain_kwargs):
    n = terrain_idx.shape[0]
    state = _TerrainState(terrain_idx, **terrain_kwargs)
    commands = torch.zeros(n, 9)
    commands[:, 4] = 1.0
    return SimpleNamespace(
        num_envs=n,
        device="cpu",
        dt=0.02,
        common_step_counter=0,
        episode_length_buf=torch.full((n,), 5, dtype=torch.long),
        command_manager=SimpleNamespace(commands=commands),
        terrain_manager=SimpleNamespace(get_state=lambda _name: state),
        command_base_height_nominal=0.75,
        robot_config=SimpleNamespace(
            dof_names=DOF_NAMES,
            lower_dof_names=DOF_NAMES[:15],
            upper_dof_names=DOF_NAMES[15:],
            lower_body_acc_penalty_dof_names=DOF_NAMES[:2],
            lower_body_torque_penalty_dof_names=DOF_NAMES[:2],
            lower_body_vel_penalty_dof_names=DOF_NAMES[:2],
            knee_dof_names=DOF_NAMES[3:5],
            knee_joint_min_threshold=0.2,
        ),
        simulator=SimpleNamespace(dof_vel=torch.zeros(n, 29), dof_pos=torch.zeros(n, 29)),
        feet_indices=[0, 1],
    )


def _all_terrains(n_per: int = 1) -> torch.Tensor:
    return torch.arange(len(TERRAINS)).repeat_interleave(n_per)


def test_falcon_preset_groups_and_terms():
    terms = g1_29dof_falcon_hm_reward_falcon.terms
    assert "penalty_hip_roll" not in terms and "stand_still" not in terms
    assert terms["tracking_walk_base_height"].weight == 1.0
    assert all(t.ma_reward_group in ("lower_body", "upper_body") for t in terms.values())
    upper = {k for k, t in terms.items() if t.ma_reward_group == "upper_body"}
    assert upper == {k for k, t in g1_29dof_falcon_hm_reward.terms.items() if t.ma_reward_group == "upper_body"}
    assert terms["penalty_lower_body_action_rate"].weight == -0.5
    assert terms["penalty_upper_body_action_rate"].weight == -0.5
    assert terms["penalty_feet_contact_forces"].weight == -0.01
    # Every mask that covers FALCON's slope indices also covers the directional slope terrains.
    for t in terms.values():
        for key in ("terrain_type_mask", "reduce_terrain_type_mask"):
            mask = t.params.get(key)
            if mask and "slopes" in mask:
                assert {"slope_up", "slope_down"} <= set(mask)
    # The base preset is unchanged.
    assert "penalty_hip_roll" in g1_29dof_falcon_hm_reward.terms
    assert g1_29dof_falcon_hm_reward.terms["termination"].ma_reward_group == "shared"


def test_tracking_walk_base_height_reduces_to_walk_flag_off_mask():
    env = _make_env(_all_terrains(), base_heights=torch.tensor([0.5, 0.6, 0.7, 0.8]))
    env.command_manager.commands[1, 4] = 0.0
    rew = lm.tracking_walk_base_height(env, terrain_type_mask=["flat"])
    assert torch.allclose(rew, torch.tensor([1.0, 0.0, 1.0, 1.0]))
    # On a masked-in terrain the error is tracked and attenuated by the hand force.
    env.apply_force_tensor = torch.zeros(4, 3, 3)
    env.left_hand_link_index, env.right_hand_link_index = 1, 2
    env.apply_force_tensor[:, 1, 2] = -25.0  # |F| = 25 -> error x 0.5
    rew = lm.tracking_walk_base_height(env, terrain_type_mask=["slopes"])
    expected0 = torch.exp(torch.tensor(-(0.25 * 0.5) / 0.05))
    assert torch.isclose(rew[0], expected0)
    assert torch.allclose(rew[2:], torch.ones(2))


def test_falcon_dof_acc_uses_previous_control_step():
    env = _make_env(_all_terrains())
    env.simulator.dof_acc = torch.full((4, 29), 1e6)  # must be ignored
    v0 = torch.randn(4, 29)
    env.simulator.dof_vel = v0.clone()
    lm.penalty_lower_body_dof_acc(env, use_all_lower_dofs=True, falcon_control_step=True)
    env.common_step_counter += 1
    v1 = torch.randn(4, 29)
    env.simulator.dof_vel = v1.clone()
    env.episode_length_buf[3] = 1  # first step after a reset: previous velocity is zero
    lower = lm.penalty_lower_body_dof_acc(env, use_all_lower_dofs=True, falcon_control_step=True)
    upper = lm.penalty_upper_body_dof_acc(env, falcon_control_step=True)
    prev = v0.clone()
    prev[3] = 0.0
    acc = (prev - v1) / env.dt
    assert torch.allclose(lower, torch.sum(acc[:, :15] ** 2, dim=1))
    # The upper-body term in the same step shares the snapshot (not advanced twice).
    assert torch.allclose(upper, torch.sum(acc[:, 15:] ** 2, dim=1))


def test_lower_body_penalties_all_dofs():
    env = _make_env(_all_terrains())
    env.simulator.dof_vel = torch.ones(4, 29)
    assert torch.allclose(lm.penalty_lower_body_dof_vel(env), torch.full((4,), 2.0))
    assert torch.allclose(lm.penalty_lower_body_dof_vel(env, use_all_lower_dofs=True), torch.full((4,), 15.0))


def test_terrain_progress_global_mean():
    env = _make_env(_all_terrains(), levels=torch.tensor([0, 2, 4, 6]))
    assert torch.allclose(lm.terrain_progress(env), torch.tensor([0.0, 0.2, 0.4, 0.6]))
    assert torch.allclose(lm.terrain_progress(env, global_mean=True), torch.full((4,), 3.0 / 10 * 2))


def test_negative_knee_threshold_override():
    env = _make_env(_all_terrains())
    env.simulator.dof_pos[:, 3:5] = 0.1
    assert torch.allclose(lm.penalty_negative_knee_joint(env), torch.full((4,), 2.0))
    assert torch.allclose(lm.penalty_negative_knee_joint(env, knee_joint_min_threshold=0.02), torch.zeros(4))


def test_base_height_stairs_reduction():
    heights = torch.tensor([[0.0, 0.3, 0.0, 0.0]] * 4)
    env = _make_env(_all_terrains(), base_heights=torch.full((4,), 0.65), heights=heights)
    plain = lm.base_height(env, desired_base_height=0.75)
    reduced = lm.base_height(env, reduce_terrain_type_mask=["stairs_up", "stairs_down"])
    slope_factor = min(0.65 / 0.6, 1.0)  # max |h - base_h| = 0.65
    scale = torch.tensor([1.0, 1 - 0.75 * slope_factor, 1 - 0.75 * slope_factor, 1.0])
    assert torch.allclose(reduced, plain * scale)


def test_hip_pos_stones_reduction_and_lin_vel_z_mask(monkeypatch):
    env = _make_env(_all_terrains())
    monkeypatch.setattr(lm, "_hips_roll_yaw_dof_indices", lambda _env: torch.tensor([0, 1]))
    env.simulator.dof_pos[:, :2] = 1.0
    mask = ["flat", "curb", "slopes", "slope_up", "slope_down", "stairs_up", "stairs_down"]
    rew = lm.penalty_hip_pos(env, terrain_type_mask=mask, reduction_scale=0.25)
    assert torch.allclose(rew, torch.tensor([2.0, 2.0, 2.0, 0.5]))
    monkeypatch.setattr(lm, "get_base_lin_vel", lambda _env: torch.tensor([[0.0, 0.0, 1.0]] * 4))
    assert torch.allclose(lm.penalty_lin_vel_z(env, terrain_type_mask=mask), torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(lm.penalty_lin_vel_z(env), torch.ones(4))


def test_feet_contact_forces_hand_force_allowance():
    env = _make_env(_all_terrains())
    env.simulator.contact_forces = torch.zeros(4, 2, 3)
    env.simulator.contact_forces[:, :, 2] = 400.0
    env.left_ee_apply_force = torch.zeros(4, 3)
    env.right_ee_apply_force = torch.zeros(4, 3)
    env.left_ee_apply_force[:, 2] = -30.0
    base = lm.penalty_feet_contact_forces(env)
    relaxed = lm.penalty_feet_contact_forces(env, add_ee_force_z=True)
    assert torch.allclose(base, torch.full((4,), 100.0))
    assert torch.allclose(relaxed, torch.full((4,), 40.0))


def test_filtered_contact_and_air_time():
    env = _make_env(_all_terrains(1)[:1])
    env.simulator.contact_forces = torch.zeros(1, 2, 3)
    env.simulator.contact_forces[0, 0, 2] = 10.0
    assert lm._feet_contact_filt(env).tolist() == [[True, False]]
    env.common_step_counter += 1
    env.simulator.contact_forces[0, 0, 2] = 0.0
    # Previous-step contact is kept for one step; repeated calls in a step agree.
    assert lm._feet_contact_filt(env).tolist() == [[True, False]]
    assert lm._feet_contact_filt(env).tolist() == [[True, False]]
    env.common_step_counter += 1
    assert lm._feet_contact_filt(env).tolist() == [[False, False]]


def test_orientation_walk_only_and_close_feet_norm(monkeypatch):
    env = _make_env(_all_terrains())
    env.command_manager.commands[0, 4] = 0.0
    monkeypatch.setattr(loco_terms, "get_projected_gravity", lambda _env: torch.tensor([[0.1, 0.0, -1.0]] * 4))
    assert torch.allclose(loco_terms.penalty_orientation(env, walk_only=True), torch.tensor([0.0, 0.01, 0.01, 0.01]))
    env.simulator._rigid_body_pos = torch.zeros(4, 2, 3)
    env.simulator._rigid_body_pos[:, 1, 0] = 0.15  # 0.15 m apart fore-aft, no lateral gap
    assert torch.allclose(loco_terms.penalty_close_feet_xy(env, 0.16, mode="norm"), torch.ones(4))
    with pytest.raises(ValueError, match="mode"):
        loco_terms.penalty_close_feet_xy(env, 0.16, mode="bad")


def test_feet_edge_and_foothold_falcon_params():
    terms = g1_29dof_falcon_hm_reward_falcon.terms
    assert terms["penalty_feet_edge"].params["use_filtered_contact"] is True
    fh = terms["penalty_foothold"]
    assert fh.func == "rpl_training.terms.reward:penalty_foothold"
    assert fh.params["falcon_heightfield"] is True and fh.params["foothold_num_samples_long"] == 10
    assert "flat" not in fh.params["terrain_type_mask"] and "stepping_stones_columns" in fh.params["terrain_type_mask"]


def test_foothold_falcon_heightfield_lookup(monkeypatch):
    # ``holosoma.utils.draw`` imports a simulator-specific adapter at import time; none is needed here.
    monkeypatch.setitem(
        sys.modules, "holosoma.utils.draw", sys.modules.get("holosoma.utils.draw") or types.ModuleType("draw")
    )
    from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion

    # 1 env, one sole sample at the foot origin; heightfield cells are 0.1 m, vertical scale 0.01.
    env = _make_env(torch.tensor([3]))
    hs = torch.zeros(10, 10, dtype=torch.int16)
    hs[2, 3] = 10  # 0.10 m under the left foot (x=0.25 -> cell 2 by truncation, y=0.39 -> cell 3)
    state = env.terrain_manager.get_state("locomotion_terrain")
    state._height_samples = hs
    state._terrain = SimpleNamespace(border_size=0.0, horizontal_scale=0.1, vertical_scale=0.01)
    state._falcon_heightfield_heights = TerrainLocomotion._falcon_heightfield_heights.__get__(state)
    env.simulator.contact_forces = torch.zeros(1, 2, 3)
    env.simulator.contact_forces[:, :, 2] = 10.0
    env.simulator._rigid_body_pos = torch.tensor([[[0.25, 0.39, 0.12], [0.55, 0.55, 0.12]]])
    env.simulator._rigid_body_rot = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])
    env.foot_samples_local = torch.zeros(2, 1, 3)
    env.num_foot_samples = 1
    # Left: 0.12 - 0.10 = 0.02 <= eps; right: 0.12 - 0 > eps -> one bad sample.
    rew = lm.penalty_foothold(env, foothold_epsilon=0.05, falcon_heightfield=True)
    assert torch.allclose(rew, torch.tensor([1.0]))
    # Masked-out terrain type zeroes it.
    rew = lm.penalty_foothold(env, terrain_type_mask=["slopes"], falcon_heightfield=True)
    assert torch.allclose(rew, torch.tensor([0.0]))
