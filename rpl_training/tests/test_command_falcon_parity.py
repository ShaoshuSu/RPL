"""FAR-FALCON-Terrains parity of the loco-manip command / force-config logic (no simulator)."""

from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
from holosoma.config_types.command import CommandTermCfg
from holosoma.utils.safe_torch_import import torch

from rpl_training.config_values.command import (
    g1_29dof_falcon_hm_command,
    g1_29dof_rpl_command_distill,
    g1_29dof_rpl_command_expert,
)
from rpl_training.config_values.curriculum import (
    g1_29dof_falcon_hm_force_curriculum,
    g1_29dof_rpl_curriculum_distill,
    g1_29dof_rpl_curriculum_expert,
)
from rpl_training.envs.loco_manip_force_manager import (
    _DEFAULT_FORCE_CONFIG,
    LeggedRobotLocoManipForceManager,
)
from rpl_training.terms.command import HeadingLocomotionCommand, UpperBodyRefPoseCommand

TERRAINS = ["slopes", "stairs_up", "stairs_down", "stepping_stones_columns"]
STAND_PROB = {"slopes": 0.4, "stairs_up": 0.2, "stairs_down": 0.2, "stepping_stones_columns": 0.2}
N_PER_TERRAIN = 20000


class _TerrainState:
    def __init__(self, n_per: int, names: list[str]):
        self.terrain_type_names = names
        self.terrain_type_idx = torch.arange(len(names)).repeat_interleave(n_per)


class _TerrainManager:
    def __init__(self, n_per: int, names: list[str]):
        self._state = _TerrainState(n_per, names)

    def get_state(self, name):
        assert name == "locomotion_terrain"
        return self._state


def _make_env(n_per: int = N_PER_TERRAIN, names: list[str] = TERRAINS):
    n = n_per * len(names)
    quat = torch.zeros(n, 4)
    quat[:, 3] = 1.0  # identity, w last
    return SimpleNamespace(
        num_envs=n,
        device="cpu",
        is_evaluating=False,
        base_quat=quat,
        dt=0.02,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        max_episode_length_s=10.0,
        terrain_manager=_TerrainManager(n_per, names),
        robot_config=SimpleNamespace(init_state=SimpleNamespace(pos=[0.0, 0.0, 0.75])),
    )


def _make_command(cmd_cfg, env):
    term_cfg = deepcopy(cmd_cfg.setup_terms["locomotion_command"])
    term = HeadingLocomotionCommand(term_cfg, env)
    term.setup()
    return term


def _per_terrain(env, values: torch.Tensor) -> dict[str, float]:
    state = env.terrain_manager.get_state("locomotion_terrain")
    idx = state.terrain_type_idx
    return {name: float(values[idx == i].float().mean()) for i, name in enumerate(state.terrain_type_names)}


def test_falcon_stand_and_tapping_distribution_per_terrain():
    torch.manual_seed(0)
    env = _make_env()
    term = _make_command(g1_29dof_rpl_command_distill, env)
    term._resample(torch.arange(env.num_envs))
    cmds = term.commands

    walk_flag = cmds[:, 4] == 1
    stand = _per_terrain(env, ~walk_flag)
    for name in TERRAINS:
        p = STAND_PROB[name]
        assert stand[name] == pytest.approx(p, abs=0.015), name

    # Tapping = walk flag set but linear velocity zeroed by the first stance draw.
    # Disable the deadband so it is the only source of zero velocity.
    term.lin_vel_deadband = 0.0
    torch.manual_seed(1)
    term._resample(torch.arange(env.num_envs))
    walk_flag = term.commands[:, 4] == 1
    zero_lin = torch.all(term.commands[:, :2] == 0, dim=1)
    # lin_vel_y is masked everywhere, so zero_lin <=> vx == 0 (measure-zero otherwise).
    tapping = _per_terrain(env, walk_flag & zero_lin)
    for name in TERRAINS:
        p = STAND_PROB[name]
        assert tapping[name] == pytest.approx((1 - p) * p, abs=0.015), name


def test_falcon_yaw_zero_on_stones_and_tapping_follows_heading():
    torch.manual_seed(0)
    env = _make_env(n_per=2000)
    term = _make_command(g1_29dof_rpl_command_distill, env)
    term.lin_vel_deadband = 0.0
    term._resample(torch.arange(env.num_envs))
    cmds = term.commands
    idx = env.terrain_manager.get_state("locomotion_terrain").terrain_type_idx
    stones = idx == TERRAINS.index("stepping_stones_columns")

    term.step()  # per-step heading -> yaw update
    assert torch.all(cmds[stones, 2] == 0)
    assert torch.all(cmds[stones, 3] == 0)  # heading masked on stones
    assert torch.all(cmds[:, 1] == 0)  # no lateral velocity on any Holosoma terrain

    # Stance envs never turn; tapping envs (walk flag, zero lin vel) track the heading.
    assert torch.all(cmds[cmds[:, 4] == 0, :3] == 0)
    tapping = (cmds[:, 4] == 1) & torch.all(cmds[:, :2] == 0, dim=1) & ~stones
    assert tapping.any()
    expected = torch.clamp(0.5 * cmds[tapping, 3], -1.0, 1.0)  # identity base yaw
    assert torch.allclose(cmds[tapping, 2], expected)


def test_falcon_waist_zero_fix_rates_and_stance_gating():
    torch.manual_seed(0)
    env = _make_env()
    term = _make_command(g1_29dof_rpl_command_distill, env)
    term._resample(torch.arange(env.num_envs))
    cmds = term.commands
    stance = cmds[:, 4] == 0
    assert torch.all(cmds[~stance, 5:9] == 0)
    active = (cmds[stance, 5:8] != 0).float().mean(dim=0)
    assert active[0] == pytest.approx(0.9, abs=0.01)
    assert active[1] == 0.0  # roll prob 1.0 (and range 0)
    assert active[2] == pytest.approx(0.75, abs=0.01)
    assert torch.all(cmds[stance, 7] >= -0.5)


def test_default_preset_unchanged_behaviour():
    term_cfg = g1_29dof_falcon_hm_command.setup_terms["locomotion_command"]
    assert "falcon_stance_resample" not in term_cfg.params
    assert "ee_force_config" not in g1_29dof_falcon_hm_command.params
    assert g1_29dof_falcon_hm_command.setup_terms["upper_body_ref_pose"].params == {}
    assert term_cfg.params["ang_vel_yaw_terrain_mask"][-1] == "stepping_stones_columns"

    torch.manual_seed(0)
    env = _make_env(n_per=5000)
    term = _make_command(g1_29dof_falcon_hm_command, env)
    term._resample(torch.arange(env.num_envs))
    stand = (term.commands[:, 4] == 0).float().mean()
    assert float(stand) == pytest.approx(0.3, abs=0.015)
    # No zero-fix draw by default: every stance env keeps a waist yaw command.
    assert torch.all(term.commands[term.commands[:, 4] == 0, 5] != 0)


def test_falcon_mode_rejects_velocity_stand_flag():
    cfg = deepcopy(g1_29dof_rpl_command_distill.setup_terms["locomotion_command"])
    cfg.params["stand_flag_from_velocity"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        HeadingLocomotionCommand(cfg, _make_env(n_per=1))


class _FakeMotionLib:
    """Linear-in-time upper-body motion: dof_pos = t, dof_vel = 1."""

    def __init__(self, num_dof: int):
        self.num_dof = num_dof

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        n = motion_times.shape[0]
        return {
            "dof_pos": motion_times.unsqueeze(1).expand(n, self.num_dof).clone(),
            "dof_vel": torch.ones(n, self.num_dof),
        }


def _make_upper_term(params: dict, n: int = 4000):
    env = _make_env(n_per=n // len(TERRAINS))
    n = env.num_envs
    num_dof = 3
    env.default_dof_pos = torch.zeros(n, num_dof)
    env.motion_ids = torch.zeros(n, dtype=torch.long)
    env.motion_start_times = torch.full((n,), 1.0)
    env.upper_body_motion_scale = torch.full((n, 1), 0.5)
    env._pending_episode_lengths = torch.full((n,), 100, dtype=torch.long)
    env.simulator = SimpleNamespace(dof_pos=torch.zeros(n, num_dof), dof_vel=torch.zeros(n, num_dof))
    term = UpperBodyRefPoseCommand(CommandTermCfg(func="x", params=params), env)
    term.upper_dof_indices = torch.tensor([1, 2])
    term.ref_upper_dof_pos = torch.zeros(n, 2)
    term.fix_upper_body = torch.zeros(n, dtype=torch.bool)
    term.fix_upper_body_motion_times = torch.zeros(n)
    term.motion_lib = _FakeMotionLib(num_dof)
    return term, env


def test_upper_body_lookahead_fix_and_reset_pose():
    term, env = _make_upper_term(dict(g1_29dof_rpl_command_distill.setup_terms["upper_body_ref_pose"].params))
    idx = torch.arange(env.num_envs)
    env.episode_length_buf[:] = 9
    torch.manual_seed(0)
    term._resample_fix_upper_body(idx)
    assert float(term.fix_upper_body.float().mean()) == pytest.approx(0.3, abs=0.03)
    assert float(term.fix_upper_body_motion_times.max()) <= 10.0
    term._set_ref_from_motion(None)
    fix = term.fix_upper_body
    # Lookahead: ((9 + 1) * 0.02 + 1.0) * 0.5 around a zero default.
    assert torch.allclose(term.ref_upper_dof_pos[~fix], torch.full_like(term.ref_upper_dof_pos[~fix], 0.6))
    assert torch.allclose(term.ref_upper_dof_pos[fix, 0], term.fix_upper_body_motion_times[fix] * 0.5)

    term._reset_upper_dofs_to_motion(idx)
    # Previous-episode time quirk: (100 * 0.02 + 1.0) * 0.5; untouched lower dof 0.
    assert torch.allclose(env.simulator.dof_pos[:, 1:], torch.full((env.num_envs, 2), 1.5))
    assert torch.all(env.simulator.dof_pos[:, 0] == 0)
    assert torch.allclose(env.simulator.dof_vel[:, 1:], torch.full((env.num_envs, 2), 0.5))

    env.is_evaluating = True
    term._resample_fix_upper_body(idx)
    assert not term.fix_upper_body.any()


def test_upper_body_defaults_preserve_legacy_time():
    term, env = _make_upper_term({})
    env.episode_length_buf[:] = 9
    term._set_ref_from_motion(None)
    assert torch.allclose(term.ref_upper_dof_pos, torch.full_like(term.ref_upper_dof_pos, (9 * 0.02 + 1.0) * 0.5))


def test_force_config_plumbing():
    tyro_cfg = SimpleNamespace(command=g1_29dof_rpl_command_distill)
    cfg = LeggedRobotLocoManipForceManager._resolve_force_config(tyro_cfg)
    assert tuple(cfg["apply_force_z_range"]) == (-100.0, 5.0)
    assert tuple(cfg["zero_force_prob"]) == (1.0, 1.0, 0.25)
    assert cfg["dof_effort_limit_scale"] == 1.0
    assert cfg["apply_force_pos_mode"] == "hand_to_extend"
    assert cfg["resample_force_with_commands"] is True
    assert cfg["left_hand_link"] == "left_rubber_hand"

    legacy = LeggedRobotLocoManipForceManager._resolve_force_config(SimpleNamespace(command=g1_29dof_falcon_hm_command))
    assert legacy == _DEFAULT_FORCE_CONFIG
    assert legacy["apply_force_pos_mode"] == "hand" and legacy["resample_force_with_commands"] is False

    bad = SimpleNamespace(command=SimpleNamespace(params={"ee_force_config": {"typo": 1}}))
    with pytest.raises(ValueError, match="Unknown ee_force_config keys"):
        LeggedRobotLocoManipForceManager._resolve_force_config(bad)


def test_force_extend_point_interpolation():
    mgr = object.__new__(LeggedRobotLocoManipForceManager)
    mgr.extend_offset = torch.tensor([0.18, -0.025, 0.0])
    hand = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    parent = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Identity and +90 deg yaw (w last).
    s = math.sqrt(0.5)
    rot = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, s, s]])
    ratio = torch.tensor([[1.0], [0.5]])
    out = mgr._interp_to_extend(hand, parent, rot, ratio)
    assert torch.allclose(out[0], torch.tensor([1.18, -0.025, 0.0]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([(1.0 + 0.025) * 0.5, 0.18 * 0.5, 0.0]), atol=1e-6)


def test_falcon_distill_curriculum_preset():
    terms = g1_29dof_rpl_curriculum_distill.setup_terms
    assert g1_29dof_rpl_curriculum_distill.params["num_compute_average_epl"] == 10000
    cmd = terms["command_scale_curriculum"].params
    assert cmd["command_height_scale_initial_scale"] == 0.1
    assert (cmd["command_height_scale_up"], cmd["command_height_scale_down"]) == (0.05, 0.025)
    assert (cmd["waist_command_scale_up"], cmd["waist_command_scale_down"]) == (0.05, 0.05)
    assert cmd["upper_body_motion_scale_initial_scale"] == 0.0
    assert terms["force_scale_curriculum"].params["scale_up"] == 0.05
    assert terms["penalty_curriculum"].params["enabled"] is False
    assert terms["domain_rand_scale_curriculum"].params["enabled"] is False
    assert list(terms) == list(g1_29dof_falcon_hm_force_curriculum.setup_terms)
    # Base preset untouched by the deepcopy.
    base = g1_29dof_falcon_hm_force_curriculum.setup_terms
    assert base["penalty_curriculum"].params["enabled"] is True
    assert base["command_scale_curriculum"].params["command_height_scale_up"] == 0.02


EXPERT_TERRAINS = ["slope_up", "slope_down", "slopes", "stairs_up", "stairs_down", "stepping_stones_columns"]
EXPERT_STAND_PROB = {
    "slope_up": 0.4,
    "slope_down": 0.4,
    "slopes": 0.4,
    "stairs_up": 0.3,
    "stairs_down": 0.3,
    "stepping_stones_columns": 0.3,
}


def test_falcon_expert_stand_tapping_yaw_and_waist():
    torch.manual_seed(0)
    env = _make_env(names=EXPERT_TERRAINS)
    term = _make_command(g1_29dof_rpl_command_expert, env)
    term.lin_vel_deadband = 0.0
    term._resample(torch.arange(env.num_envs))
    cmds = term.commands
    walk_flag = cmds[:, 4] == 1
    zero_lin = torch.all(cmds[:, :2] == 0, dim=1)
    stand = _per_terrain(env, ~walk_flag)
    tapping = _per_terrain(env, walk_flag & zero_lin)
    for name, p in EXPERT_STAND_PROB.items():
        assert stand[name] == pytest.approx(p, abs=0.015), name
        assert tapping[name] == pytest.approx((1 - p) * p, abs=0.015), name

    term.step()
    idx = env.terrain_manager.get_state("locomotion_terrain").terrain_type_idx
    stones = idx == EXPERT_TERRAINS.index("stepping_stones_columns")
    assert torch.all(cmds[stones, 2] == 0) and torch.all(cmds[stones, 3] == 0)
    # Directional slopes keep yaw / heading like FALCON's slope_up / slope_down.
    slopes = (idx == 0) | (idx == 1)
    assert (cmds[slopes & walk_flag, 2] != 0).float().mean() > 0.9

    stance = cmds[:, 4] == 0
    active = (cmds[stance, 5:8] != 0).float().mean(dim=0)
    assert active[0] == pytest.approx(0.75, abs=0.01)
    assert active[1] == 0.0
    assert active[2] == pytest.approx(0.75, abs=0.01)


def test_falcon_expert_force_config_and_upper_body():
    cfg = LeggedRobotLocoManipForceManager._resolve_force_config(SimpleNamespace(command=g1_29dof_rpl_command_expert))
    assert tuple(cfg["apply_force_z_range"]) == (-60.0, 5.0)
    assert cfg["dof_effort_limit_scale"] == 1.0  # experts set robot.dof_effort_limit_scale=1.0
    assert cfg["apply_force_pos_mode"] == "hand_to_extend"
    expert = g1_29dof_rpl_command_expert.setup_terms
    distill = g1_29dof_rpl_command_distill.setup_terms
    assert expert["upper_body_ref_pose"].params == distill["upper_body_ref_pose"].params
    # Shared builder kept the distill preset's values.
    loco = distill["locomotion_command"].params
    assert loco["zero_fix_waist_yaw_prob"] == 0.1
    assert loco["heading_terrain_mask"] == ["slopes", "stairs_up", "stairs_down"]
    assert "stand_prob_terrain" not in g1_29dof_falcon_hm_command.setup_terms["locomotion_command"].params


def test_falcon_expert_curriculum_preset():
    terms = g1_29dof_rpl_curriculum_expert.setup_terms
    assert g1_29dof_rpl_curriculum_expert.params["num_compute_average_epl"] == 10000
    cmd = terms["command_scale_curriculum"].params
    for key in ("command_height_scale", "waist_command_scale", "upper_body_motion_scale"):
        assert (cmd[f"{key}_up"], cmd[f"{key}_down"]) == (0.02, 0.02)
    assert cmd["upper_body_motion_scale_initial_scale"] == 0.0
    assert terms["force_scale_curriculum"].params["scale_up"] == 0.02
    for name in ("penalty_curriculum", "domain_rand_scale_curriculum"):
        assert terms[name].params["enabled"] is True
        assert terms[name].params["degree"] == 1e-5
    assert list(terms) == list(g1_29dof_falcon_hm_force_curriculum.setup_terms)
