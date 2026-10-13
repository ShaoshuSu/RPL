"""CPU tests for the FAR-FALCON-Terrains terrain / heightmap / DR parity options."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

# ``holosoma.utils.draw`` imports a simulator-specific adapter at import time; none is needed here.
sys.modules.setdefault("holosoma.utils.draw", types.ModuleType("holosoma.utils.draw"))

from holosoma.managers.action.terms.joint_control import JointPositionActionTerm  # noqa: E402
from holosoma.managers.randomization.terms.locomotion import HeightMeasurementXYOffsetRandomizer  # noqa: E402
from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion  # noqa: E402
from holosoma.simulator.shared.terrain import Terrain  # noqa: E402
from holosoma.utils import terrain_utils  # noqa: E402
from holosoma.utils.rotations import quat_apply_yaw, quat_from_euler_xyz  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402
from holosoma.utils.sampler import STAGE_RESET, TermSampler  # noqa: E402

from rpl_training.config_values import simulator as _sim_presets  # noqa: E402
from rpl_training.config_values import terrain as _terrain_presets  # noqa: E402
from rpl_training.config_values.randomization import (  # noqa: E402
    g1_29dof_rpl_randomization_distill,
    g1_29dof_rpl_randomization_expert,
)

SIM_DEFAULTS = vars(_sim_presets)
TERRAIN_DEFAULTS = vars(_terrain_presets)
FALCON = TERRAIN_DEFAULTS["terrain_locomotion_multi_falcon"].terrain_term
FALCON_EXPERT = TERRAIN_DEFAULTS["terrain_locomotion_multi_falcon_expert"].terrain_term


def test_falcon_terrain_preset_params():
    # Expert order of the FALCON MoE teacher: stones, rough slope, stairs up, stairs down.
    assert list(FALCON.terrain_config) == ["stepping_stones_columns", "slopes", "stairs_up", "stairs_down"]
    assert list(FALCON.terrain_config.values()) == [0.35, 0.15, 0.25, 0.25]
    assert FALCON.max_slope == 0.75
    assert FALCON.step_width_range == [0.25, 0.30]
    assert FALCON.slopes_roughness
    assert FALCON.height_scan_mode == "falcon"
    assert FALCON.body_height_mode == "falcon"
    # Defaults stay off.
    base = TERRAIN_DEFAULTS["terrain_locomotion_multi"].terrain_term
    assert not base.slopes_roughness
    assert base.height_scan_mode == "raycast"
    assert base.body_height_mode == "raycast"


def test_falcon_expert_terrain_preset_params():
    base = TERRAIN_DEFAULTS["terrain_locomotion_multi"].terrain_term
    cfg = FALCON_EXPERT.terrain_config
    # Every type is listed so the per-terrain experts can pick their mix on the CLI.
    assert set(cfg) == {"slopes", "slope_up", "slope_down", "stairs_up", "stairs_down", "stepping_stones_columns"}
    assert cfg["slopes"] == 0.0 and cfg["slope_up"] == cfg["slope_down"]
    assert (FALCON_EXPERT.height_scan_mode, FALCON_EXPERT.body_height_mode) == ("falcon", "falcon")
    assert not FALCON_EXPERT.slopes_roughness  # FALCON exp sets ``rough_terrain: False``
    assert {"slope_up", "slope_down"} <= set(FALCON_EXPERT.spawn.xy_offset_terrain_mask)
    # FALCON ``falcon_hm_multi_terrain`` geometry is the multi-terrain base.
    for name in ("max_slope", "step_width_range", "stairs_ratio", "stone_size_range", "stone_depth_range"):
        assert getattr(FALCON_EXPERT, name) == getattr(base, name)


def _slope_tile(cfg, difficulty: float, seed: int = 0) -> np.ndarray:
    gen = Terrain.__new__(Terrain)
    gen._cfg = cfg
    gen._max_slope = cfg.max_slope
    tile = terrain_utils.SubTerrain(
        "terrain", width=80, length=80, vertical_scale=cfg.vertical_scale, horizontal_scale=cfg.horizontal_scale
    )
    np.random.seed(seed)
    gen._slopes_terrain_func(tile, difficulty)
    return tile.height_field_raw.astype(np.int64)


def test_rough_slope_adds_noise_only_when_enabled():
    smooth = _slope_tile(replace(FALCON, slopes_roughness=False), 0.5)
    rough = _slope_tile(FALCON, 0.5)

    # A pyramid slope is piecewise linear, so its second differences are zero except on the ridges.
    def curvature(h: np.ndarray) -> float:
        return float(np.mean(np.abs(np.diff(h, n=2, axis=1)) > 1))

    assert curvature(smooth) < 0.05
    assert curvature(rough) > 0.1
    # The noise is a 2-D field (FALCON ``interp2d``), not one row broadcast over the tile.
    noise = rough.astype(np.int64) - smooth.astype(np.int64)
    assert len(np.unique(noise, axis=0)) > noise.shape[0] // 2
    assert len(np.unique(noise, axis=1)) > noise.shape[1] // 2
    # Noise amplitude is bounded by amplitude_range[1] on top of a slope of the same magnitude.
    amp_cells = FALCON.amplitude_range[1] / FALCON.vertical_scale
    assert np.abs(np.abs(rough) - np.abs(smooth)).max() <= amp_cells + 1


def _falcon_reference(height_samples, points, border, hs_scale, vs_scale, offsets):
    """Verbatim port of FAR-FALCON-Terrains ``IsaacGym._get_heights`` index math."""
    num_envs = points.shape[0]
    points = ((points + border) / hs_scale).long()
    px = torch.clip(points[:, :, 0].view(-1), 0, height_samples.shape[0] - 2)
    py = torch.clip(points[:, :, 1].view(-1), 0, height_samples.shape[1] - 2)
    if offsets is None:
        heights = height_samples[px, py]
    else:
        n = offsets.unsqueeze(1)
        px_1 = (px.view(num_envs, -1) + n).view(-1)
        px_2 = (px.view(num_envs, -1) - n).view(-1)
        py_1 = (py.view(num_envs, -1) + n).view(-1)
        py_2 = (py.view(num_envs, -1) - n).view(-1)
        heights = torch.stack(
            [
                height_samples[px, py],
                height_samples[px, py_1],
                height_samples[px, py_2],
                height_samples[px_1, py],
                height_samples[px_2, py],
            ]
        ).min(dim=0)[0]
    return heights.view(num_envs, -1) * vs_scale


@pytest.mark.parametrize("use_offsets", [False, True])
def test_falcon_heightfield_lookup_matches_reference(use_offsets):
    torch.manual_seed(0)
    border, hs_scale, vs_scale = 2.0, 0.05, 0.005
    samples = torch.randint(-100, 100, (200, 240), dtype=torch.int16)
    term = TerrainLocomotion.__new__(TerrainLocomotion)
    term._height_samples = samples
    term._terrain = SimpleNamespace(border_size=border, horizontal_scale=hs_scale, vertical_scale=vs_scale)
    # Interior points so FALCON's unclipped neighbour indices stay in range.
    pts = torch.rand(6, 15, 3) * torch.tensor([5.0, 6.0, 1.0]) + 0.5
    offsets = torch.tensor([0, 1, 2, 3, 0, 1]) if use_offsets else None
    ours = term._falcon_heightfield_heights(pts, offsets)
    ref = _falcon_reference(samples, pts, border, hs_scale, vs_scale, offsets)
    torch.testing.assert_close(ours, ref.float())


def test_falcon_heightfield_lookup_hand_computed():
    samples = torch.arange(25, dtype=torch.int16).view(5, 5)  # value = 5 * ix + iy
    term = TerrainLocomotion.__new__(TerrainLocomotion)
    term._height_samples = samples
    term._terrain = SimpleNamespace(border_size=1.0, horizontal_scale=1.0, vertical_scale=0.5)
    # (0.9, 1.2) -> floor((1.9, 2.2)) = (1, 2) -> 7; (5, 5) clipped to (3, 3) -> 18; (-3, 0) -> (0, 1) -> 1
    pts = torch.tensor([[[0.9, 1.2, 0.0], [5.0, 5.0, 0.0], [-3.0, 0.0, 0.0]]])
    out = term._falcon_heightfield_heights(pts, None)
    torch.testing.assert_close(out, torch.tensor([[7.0, 18.0, 1.0]]) * 0.5)
    # Offset 1 at (1, 2): min of 7, 8, 6, 12, 2 -> 2
    out = term._falcon_heightfield_heights(pts[:, :1], torch.tensor([1]))
    torch.testing.assert_close(out, torch.tensor([[1.0]]))


def test_ctrl_delay_mask_only_delays_selected_joints():
    term = JointPositionActionTerm.__new__(JointPositionActionTerm)
    num_envs, num_dof, depth = 2, 4, 3
    term.env = SimpleNamespace(num_envs=num_envs, action_delay_idx=torch.tensor([1, 0]))
    term.action_queue = torch.zeros(num_envs, depth, num_dof)
    term.action_queue[:, 0] = 7.0  # previous step's action
    term._processed_actions = torch.ones(num_envs, num_dof)
    term._actions_after_delay = torch.zeros(num_envs, num_dof)
    term._ctrl_delay_dof_mask = torch.tensor([False, False, True, True])
    term._apply_action_delay()
    # env 0 delayed by 1 step on dofs 2,3 only; env 1 has zero delay.
    torch.testing.assert_close(term._actions_after_delay[0], torch.tensor([1.0, 1.0, 7.0, 7.0]))
    torch.testing.assert_close(term._actions_after_delay[1], torch.ones(num_dof))
    term._ctrl_delay_dof_mask = None
    term._apply_action_delay()
    torch.testing.assert_close(term._actions_after_delay[0], torch.ones(num_dof))  # queue[1] = last step's 1.0


def test_falcon_distill_randomization_preset():
    setup = g1_29dof_rpl_randomization_distill.setup_terms
    delay = setup["setup_action_delay_buffers"].params
    assert delay["enabled"] and delay["ctrl_delay_step_range"] == [0, 1]
    assert delay["upper_body_ctrl_delay"] and not delay.get("lower_body_ctrl_delay", False)
    assert setup["mass_randomizer"].params["base_mass_body_name"] == "pelvis"
    assert setup["randomize_friction_startup"].params["friction_range"] == [-0.1, 1.0]
    com = setup["randomize_base_com_startup"].params
    assert com["enabled"] and com["base_com_range"]["x"] == [-0.025, 0.025]


def test_isaacgym_falcon_preset():
    physx = SIM_DEFAULTS["isaacgym_falcon"].config.sim.physx
    assert (physx.num_position_iterations, physx.num_velocity_iterations) == (4, 0)
    assert (physx.contact_offset, physx.rest_offset, physx.max_depenetration_velocity) == (0.01, 0.0, 1.0)
    assert (physx.default_buffer_size_multiplier, physx.contact_collection) == (5, 2)
    base = SIM_DEFAULTS["isaacgym"].config.sim.physx
    assert base.contact_offset is None and base.contact_collection is None
    assert SIM_DEFAULTS["isaacgym_falcon"].config.name == "isaacgym"


def test_falcon_terrain_generates_with_raw_heightfield():
    np.random.seed(0)
    cfg = replace(FALCON, num_rows=2, num_cols=4, border_size=2)
    terrain = Terrain(cfg, num_robots=8)
    hf = terrain.height_field_raw
    assert hf is not None
    assert hf.shape == (
        int(2 * cfg.terrain_length / cfg.horizontal_scale) + 2 * int(cfg.border_size / cfg.horizontal_scale),
        int(4 * cfg.terrain_width / cfg.horizontal_scale) + 2 * int(cfg.border_size / cfg.horizontal_scale),
    )
    assert terrain.vertical_scale == cfg.vertical_scale
    assert list(terrain.terrain_type_names) == list(cfg.terrain_config)


def _falcon_body_height_term(samples, border, hs_scale, vs_scale, root_states, base_quat, feet_pos):
    term = TerrainLocomotion.__new__(TerrainLocomotion)
    term._heightfield = samples
    term._terrain = SimpleNamespace(border_size=border, horizontal_scale=hs_scale, vertical_scale=vs_scale)
    term.device = "cpu"
    term.num_envs = root_states.shape[0]
    term._body_height_mode = "falcon"
    bx = torch.tensor([-0.05, 0.0, 0.05])
    by = torch.tensor([-0.1, -0.05, 0.0, 0.05, 0.1])
    gx, gy = torch.meshgrid(bx, by, indexing="ij")
    term._base_height_points = torch.zeros(term.num_envs, gx.numel(), 3)
    term._base_height_points[:, :, 0] = gx.flatten()
    term._base_height_points[:, :, 1] = gy.flatten()
    term.env = SimpleNamespace(
        simulator=SimpleNamespace(robot_root_states=root_states, _rigid_body_pos=feet_pos),
        base_quat=base_quat,
        feet_height_indices=[1, 3],
    )
    return term


def _falcon_base_heights_reference(hs, root_states, base_quat, border, hs_scale, vs_scale):
    """Verbatim port of FAR-FALCON-Terrains ``IsaacGym._get_base_heights``."""
    y = torch.tensor([-0.1, -0.05, 0.0, 0.05, 0.1])
    x = torch.tensor([-0.05, 0.0, 0.05])
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    n = root_states.shape[0]
    pts = torch.zeros(n, grid_x.numel(), 3)
    pts[:, :, 0] = grid_x.flatten()
    pts[:, :, 1] = grid_y.flatten()
    points = quat_apply_yaw(base_quat.repeat(1, grid_x.numel()), pts, True) + root_states[:, :3].unsqueeze(1)
    points += border
    points = (points / hs_scale).long()
    px = torch.clip(points[:, :, 0].view(-1), 0, hs.shape[0] - 2)
    py = torch.clip(points[:, :, 1].view(-1), 0, hs.shape[1] - 2)
    heights = torch.min(torch.min(hs[px, py], hs[px + 1, py]), hs[px, py + 1])
    base_height = heights.view(n, -1) * vs_scale
    return root_states[:, 2] - torch.mean(base_height, dim=1)


def _falcon_feet_heights_reference(hs, feet_pos, border, hs_scale, vs_scale):
    """Verbatim port of FAR-FALCON-Terrains ``IsaacGym._get_feet_heights``."""
    points = feet_pos.clone() + border
    points = (points / hs_scale).long()
    px = torch.clip(points[:, :, 0].view(-1), 0, hs.shape[0] - 2)
    py = torch.clip(points[:, :, 1].view(-1), 0, hs.shape[1] - 2)
    heights = torch.max(torch.max(hs[px, py], hs[px + 1, py]), hs[px, py + 1])
    heights = heights.view(feet_pos.shape[0], -1) * vs_scale
    return feet_pos[:, :, 2] - heights


def test_falcon_base_and_feet_heights_match_reference():
    torch.manual_seed(0)
    border, hs_scale, vs_scale = 2.0, 0.05, 0.005
    samples = torch.randint(-100, 100, (200, 240), dtype=torch.int16)
    n = 7
    root = torch.zeros(n, 13)
    root[:, :3] = torch.rand(n, 3) * torch.tensor([5.0, 6.0, 1.0]) + torch.tensor([0.5, 0.5, 0.6])
    yaw = (torch.rand(n) - 0.5) * 6.0
    quat = quat_from_euler_xyz(torch.zeros(n), torch.zeros(n), yaw)
    bodies = torch.rand(n, 5, 3) * torch.tensor([5.0, 6.0, 0.3]) + 0.5
    term = _falcon_body_height_term(samples, border, hs_scale, vs_scale, root, quat, bodies)

    base_h, base_hits = term._get_base_heights()
    ref = _falcon_base_heights_reference(samples, root, quat, border, hs_scale, vs_scale)
    torch.testing.assert_close(base_h, ref.float())
    torch.testing.assert_close(root[:, 2] - base_hits[:, 0, 2], base_h)

    feet_h, _ = term._get_feet_heights()
    ref = _falcon_feet_heights_reference(samples, bodies[:, [1, 3]], border, hs_scale, vs_scale)
    torch.testing.assert_close(feet_h, ref.float())

    # Env subset (reset path) matches the full computation.
    ids = torch.tensor([1, 4])
    torch.testing.assert_close(term._get_base_heights(ids)[0], base_h[ids])
    torch.testing.assert_close(term._get_feet_heights(ids)[0], feet_h[ids])


def test_falcon_body_heights_bias_at_stair_edge():
    # One cell column of a 0.1 m step: base takes the min (lower step), feet the max (upper step).
    samples = torch.zeros(40, 40, dtype=torch.int16)
    samples[21:, :] = 20  # 0.1 m at vertical_scale 0.005
    root = torch.zeros(1, 13)
    root[0, :3] = torch.tensor([0.9999, 1.0, 0.8])  # cell 19 / 20 boundary with border 0, scale 0.05
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    feet = torch.zeros(1, 4, 3)
    feet[0, :, :] = torch.tensor([1.0001, 1.0, 0.1])  # cell 20, +x neighbour is the upper step
    term = _falcon_body_height_term(samples, 0.0, 0.05, 0.005, root, quat, feet)
    base_h, _ = term._get_base_heights()
    feet_h, _ = term._get_feet_heights()
    # Base grid spans cells 18..20 in x; the min with the +x neighbour keeps every sample on the lower step.
    torch.testing.assert_close(base_h, torch.tensor([0.8]))
    torch.testing.assert_close(feet_h, torch.tensor([[0.0, 0.0]]))


@pytest.mark.parametrize(("truncate", "scale", "expected_hi"), [(False, 0.6, 1), (True, 0.6, 0), (True, 1.0, 1)])
def test_xy_offset_scaled_range_rounding(truncate, scale, expected_hi):
    params = {"enabled": True, "height_measurements_xy_offset_points_range": [0, 1]}
    if truncate:
        params["truncate_scaled_range"] = True
    env = SimpleNamespace(num_envs=4096, device="cpu")
    term = HeightMeasurementXYOffsetRandomizer(SimpleNamespace(params=params), env)
    term._offsets = torch.zeros(env.num_envs, dtype=torch.long)
    term.set_curriculum_scale(scale)
    sampler = TermSampler.bind(
        0, "height_measurements_xy_offset", STAGE_RESET, torch.zeros(env.num_envs, dtype=torch.long)
    )
    term._resample(torch.arange(env.num_envs), sampler)
    assert int(term._offsets.max()) == expected_hi


def test_falcon_expert_randomization_preset():
    setup = g1_29dof_rpl_randomization_expert.setup_terms
    distill = g1_29dof_rpl_randomization_distill.setup_terms
    assert setup["randomize_friction_startup"].params["friction_range"] == [0.25, 1.25]
    assert not setup["randomize_base_com_startup"].params["enabled"]
    assert setup["randomize_base_com_startup"].params["base_com_range"]["x"] == [-0.025, 0.025]
    assert setup["height_measurements_xy_offset"].params["truncate_scaled_range"]
    assert not setup["height_measurements_xy_offset"].params["enabled"]
    for name in ("setup_action_delay_buffers", "mass_randomizer"):
        assert setup[name] == distill[name]


def test_terrain_spacing_layout():
    # FAR-FALCON ``spacing_size``: (n - 1) flat gaps per axis, origins shifted by the gap per tile.
    cfg = replace(FALCON, num_rows=2, num_cols=3, border_size=2, terrain_config={"stairs_up": 1.0}, spacing_size=1.0)
    np.random.seed(0)
    terrain = Terrain(cfg, num_robots=6)
    hs = cfg.horizontal_scale
    L, W = int(cfg.terrain_length / hs), int(cfg.terrain_width / hs)
    sp, border = int(1.0 / hs), int(cfg.border_size / hs)
    hf = terrain.height_field_raw
    assert hf.shape == (2 * L + sp + 2 * border, 3 * W + 2 * sp + 2 * border)
    # Gaps are flat; every tile has stairs.
    assert not hf[border + L : border + L + sp].any()
    assert (
        not hf[:, border + W : border + W + sp].any() and not hf[:, border + 2 * W + sp : border + 2 * (W + sp)].any()
    )
    assert hf[border + L + sp : border + 2 * L + sp, border + 2 * (W + sp) : border + 3 * W + 2 * sp].any()
    origins = terrain._env_origins
    np.testing.assert_allclose(origins[1, 2, :2], [1.5 * cfg.terrain_length + 1.0, 2.5 * cfg.terrain_width + 2.0])
    assert terrain._total_length == 2 * cfg.terrain_length + 1.0 + 2 * cfg.border_size
    # No spacing keeps the packed layout.
    np.random.seed(0)
    packed = Terrain(replace(cfg, spacing_size=0.0), num_robots=6)
    assert packed.height_field_raw.shape == (2 * L + 2 * border, 3 * W + 2 * border)


def test_directional_slope_roughness():
    def tile(rough: bool, func: str) -> np.ndarray:
        gen = Terrain.__new__(Terrain)
        gen._cfg = replace(FALCON, slopes_roughness=rough)
        gen._max_slope = gen._cfg.max_slope
        t = terrain_utils.SubTerrain("terrain", width=80, length=80, vertical_scale=0.005, horizontal_scale=0.05)
        np.random.seed(1)
        getattr(gen, func)(t, 0.5)
        return t.height_field_raw.astype(np.int64)

    for func in ("_slope_up_terrain_func", "_slope_down_terrain_func"):
        smooth, rough = tile(False, func), tile(True, func)
        noise = rough - smooth
        assert noise.any()
        assert np.abs(noise).max() <= FALCON.amplitude_range[1] / FALCON.vertical_scale + 1


def test_falcon_per_terrain_expert_presets():
    from rpl_training.config_values import experiment

    stones = experiment.g1_29dof_rpl_expert_stones
    stairs = experiment.g1_29dof_rpl_expert_stairs
    slopes = experiment.g1_29dof_rpl_expert_slopes
    for exp, mix in (
        (stones, {"stepping_stones_columns"}),
        (stairs, {"stairs_up", "stairs_down"}),
        (slopes, {"slope_up", "slope_down"}),
    ):
        t = exp.terrain.terrain_term
        assert {k for k, v in t.terrain_config.items() if v > 0} == mix
        assert t.spacing_size == 1.0
        rs = exp.randomization.setup_terms
        assert rs["randomize_base_com_startup"].params["enabled"]
        assert not rs["setup_action_delay_buffers"].params["lower_body_ctrl_delay"]
        assert exp.curriculum == experiment.g1_29dof_rpl_expert.curriculum
    assert stairs.terrain.terrain_term.step_width_range == [0.25, 0.30]
    assert slopes.terrain.terrain_term.slopes_roughness and slopes.terrain.terrain_term.max_slope == 0.75
    assert not stairs.terrain.terrain_term.slopes_roughness
    fr = [
        e.randomization.setup_terms["randomize_friction_startup"].params["friction_range"]
        for e in (stones, stairs, slopes)
    ]
    assert fr == [[-0.6, 1.0], [-0.6, 1.0], [-0.1, 1.0]]
    assert stairs.randomization.setup_terms["height_measurements_xy_offset"].params["enabled"]
    assert not stones.randomization.setup_terms["height_measurements_xy_offset"].params["enabled"]
    assert [e.training.num_envs for e in (stones, stairs, slopes)] == [16384, 16384, 4096]
    r_stones, r_stairs, r_slopes = (e.reward.terms for e in (stones, stairs, slopes))
    assert r_stones["penalty_feet_ori"].weight == -0.25
    assert "stepping_stones_columns" in r_stones["penalty_feet_ori"].params["terrain_type_mask"]
    assert "stepping_stones_columns" not in r_stairs["penalty_feet_ori"].params["terrain_type_mask"]
    assert (r_stones["penalty_feet_edge"].weight, r_stones["penalty_feet_contact_forces"].weight) == (-5.0, -0.0025)
    assert (r_stairs["penalty_feet_ori"].weight, r_stairs["penalty_stumble"].weight) == (-1.0, -50.0)
    assert r_slopes["penalty_feet_ori"].weight == -2.0
    for r, vx in ((r_stones, 5.0), (r_stairs, 6.0), (r_slopes, 6.0)):
        assert r["tracking_lin_vel_x"].weight == vx
        assert r["penalty_contact_no_vel"].weight == -1.0
        assert r["penalty_lower_body_torques"].weight == -4e-5
    hm = [e.observation.groups["perception_obs"].terms["height_measurements"].noise for e in (stones, stairs, slopes)]
    assert hm == [0.1, 0.0, 0.1]
