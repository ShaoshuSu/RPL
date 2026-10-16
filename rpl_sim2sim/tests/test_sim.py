"""Sim side (``rplmujoco`` env): the vendored FAR-FALCON-Terrains sim2sim scene and config."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("yaml")

from rpl_sim2sim.run_sim import load_config  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def model(config):
    return mujoco.MjModel.from_xml_path(config["ROBOT_SCENE"])


def test_scene_is_falcons(model, config):
    # FALCON's G1 has six zero-force floating-base motors ahead of the 29 joint motors.
    robot = config["ROBOT_CFG"]
    assert robot["FREE_BASE"] and model.nu == 6 + robot["NUM_MOTORS"] == 35
    assert model.opt.iterations == 50
    for name in config["ROBOT_CFG"]["cameras"]["poses"]:
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert cam >= 0 and model.cam_fovy[cam] == pytest.approx(69.0)
    # MJCF actuator limits are the effort limits the bridge clips to.
    np.testing.assert_allclose(model.actuator_ctrlrange[6:, 1], robot["motor_effort_limit_list"])


def test_depth_settings_follow_falcon(config):
    props = config["ROBOT_CFG"]["cameras"]["props"]
    assert (props["width"], props["height"], props["resized_width"], props["resized_height"]) == (240, 135, 48, 27)
    assert (props["near_clip"], props["far_clip"], props["frame_rate"], props["depth_delay"]) == (0.1, 2.0, 10, 1)
    assert config["SIMULATE_DT"] == 0.005 and config["VIEWER_DT"] == 0.02


def test_image_server_post_process():
    cv2 = pytest.importorskip("cv2")
    from rpl_sim2sim.image_server import ImageServer

    server = ImageServer.__new__(ImageServer)
    server.if_resize, server.expected_shape, server.near_clip, server.far_clip = True, [27, 48], 0.1, 2.0
    depth = np.random.default_rng(0).uniform(0.0, 4.0, size=(135, 240)).astype(np.float32)
    ref = cv2.resize(depth, (48, 27), interpolation=cv2.INTER_CUBIC)
    np.testing.assert_allclose(server._post_process_frame(depth), (np.clip(ref, 0.1, 2.0) - 0.1) / 1.9 - 0.5, atol=1e-6)
