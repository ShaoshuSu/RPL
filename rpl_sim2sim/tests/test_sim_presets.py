"""Sim-side checks (need holosoma + mujoco): the sim2sim environment is FALCON's."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("holosoma")
mujoco = pytest.importorskip("mujoco")

from holosoma.config_types.run_sim import RunSimConfig  # noqa: E402
from holosoma.utils.config_registry import parse_config  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402

from rpl_sim2sim import sim_presets  # noqa: E402
from rpl_sim2sim.run_sim import DEFAULT_ARGS  # noqa: E402


def _falcon_model():
    path = resolve_data_file_path(f"@holosoma/data/robots/{sim_presets.g1_29dof_falcon_sim2sim.asset.xml_file}")
    return mujoco.MjModel.from_xml_path(path)


def test_launcher_defaults_parse() -> None:
    cfg = parse_config(RunSimConfig, args=list(DEFAULT_ARGS))
    assert cfg.simulator.config.sim.fps == 200
    assert cfg.robot.asset.robot_type == "g1_29dof"
    assert set(cfg.sensor) == set(sim_presets.FALCON_DEPTH_CAMERAS)
    assert cfg.plugin["depth"].cameras == list(sim_presets.FALCON_DEPTH_CAMERAS)


def test_scene_is_falcons_course_and_robot() -> None:
    model = _falcon_model()
    assert model.nu == 29  # FALCON's zero-force floating-base motors removed
    for i in range(model.nu):
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == joint
    ctrl = np.abs(model.actuator_ctrlrange[:, 1])
    np.testing.assert_allclose(ctrl, sim_presets.g1_29dof_falcon_sim2sim.dof_effort_limit_list)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    assert {"cam_front_depth", "cam_back_depth"} <= names


@pytest.mark.parametrize(
    ("preset", "falcon_camera"),
    [(sim_presets.falcon_front_depth, "cam_front_depth"), (sim_presets.falcon_back_depth, "cam_back_depth")],
)
def test_depth_cameras_use_falcons_mjcf_poses(preset, falcon_camera) -> None:
    model = _falcon_model()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, falcon_camera)
    np.testing.assert_allclose(preset.mount.position, model.cam_pos[cam], atol=1e-6)
    np.testing.assert_allclose(preset.mount.orientation, model.cam_quat[cam], atol=1e-6)
    assert preset.vertical_fov == pytest.approx(model.cam_fovy[cam])
    assert (preset.width, preset.height) == (240, 135)


def test_depth_block_follows_falcons_image_server() -> None:
    cfg = sim_presets.depth_shm_falcon
    assert (cfg.resized_height, cfg.resized_width, cfg.near_clip, cfg.far_clip) == (27, 48, 0.1, 2.0)
    assert cfg.resize_mode == "cv2_cubic" and not cfg.clip_before_resize
    assert cfg.latency_frames == 1 and cfg.render_hz == 10.0
