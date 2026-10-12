"""Unit tests for the FALCON-parity features of ``WarpDepthCamera``.

These exercise the post-processing / delay / randomization paths without a
simulator: ``depth_buffer`` is filled by hand, except for one end-to-end
ray-cast against a flat plane.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from holosoma.config_types.camera import CameraManagerCfg, CameraPose, CameraProps, CameraTermCfg
from holosoma.config_values.camera import CAMERA_MANAGER_REGISTRY
from holosoma.managers.camera.manager import CameraManager
from holosoma.managers.camera.terms.warp_depth import WarpDepthCamera, resize_depth
from holosoma.utils.rotations import quat_mul

from rpl_training.config_values import registry  # noqa: F401  (registers the presets)
from rpl_training.config_values.camera import warp_dual_depth_cameras_falcon
from rpl_training.terms.observation import cam_depth

# Plain warp dual-depth rig (no FALCON post-processing), as in FAR-Holosoma's
# ``warp_dual_depth_cameras`` preset: the camera manager's default path.
warp_dual_depth_cameras = CameraManagerCfg(
    terms={
        "warp_depth": CameraTermCfg(
            func="holosoma.managers.camera.terms.warp_depth:WarpDepthCamera",
            params={
                "poses": {
                    "front_depth": CameraPose(
                        camera_body_link="torso_link", camera_offset=[0.1, 0.0, 0.1], camera_rotation=[0.0, 75.0, 0.0]
                    ),
                    "back_depth": CameraPose(
                        camera_body_link="torso_link",
                        camera_offset=[-0.1, 0.0, 0.1],
                        camera_rotation=[0.0, 75.0, 180.0],
                    ),
                },
                "props": CameraProps(
                    image_type="depth",
                    width=240,
                    height=135,
                    resized_width=48,
                    resized_height=27,
                    horizontal_fov=101.41,
                    vertical_fov=69.00,
                    near_clip=0.1,
                    far_clip=2.0,
                    image_show=False,
                    frame_rate=10,
                    depth_delay=1,
                ),
                "dynamic_meshes": {"enable": False},
            },
        ),
    }
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_ENVS = 6


def _falcon_params(**overrides: Any) -> dict[str, Any]:
    params = dict(warp_dual_depth_cameras_falcon.terms["warp_depth"].params)
    params.pop("dynamic_meshes")  # needs mesh files + simulator; irrelevant here
    params.update(overrides)
    return params


def _make_term(params: dict[str, Any]) -> WarpDepthCamera:
    env = SimpleNamespace(num_envs=NUM_ENVS, device=DEVICE)
    return WarpDepthCamera(CameraTermCfg(func="unused", params=params), env)


def _make_manager(params: dict[str, Any]) -> tuple[CameraManager, WarpDepthCamera, SimpleNamespace]:
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device=DEVICE,
        reset_buf=torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE),
    )
    cfg = CameraManagerCfg(
        terms={
            "warp_depth": CameraTermCfg(func="holosoma.managers.camera.terms.warp_depth:WarpDepthCamera", params=params)
        }
    )
    manager = CameraManager(cfg, env, DEVICE)
    env.camera_manager = manager
    term = manager.get_state("warp_depth")
    assert isinstance(term, WarpDepthCamera)
    return manager, term, env


def _install_fake_capture(term: WarpDepthCamera) -> list[float]:
    """Replace ray casting by filling depth_buffer with a constant per capture."""
    values: list[float] = []

    def fake_capture() -> torch.Tensor:
        term.depth_buffer.fill_(values.pop(0))
        term._update_processed()
        return term.depth_buffer

    term.step = lambda: None  # type: ignore[method-assign]
    term.capture = fake_capture  # type: ignore[method-assign]
    return values


def test_preset_registered() -> None:
    assert CAMERA_MANAGER_REGISTRY["warp_dual_depth_cameras_falcon"] is warp_dual_depth_cameras_falcon


def test_default_preset_unchanged() -> None:
    params = dict(warp_dual_depth_cameras.terms["warp_depth"].params)
    params.pop("dynamic_meshes")
    term = _make_term(params)
    assert not term._postprocess and not term._use_delay and not term.needs_reset
    assert not term.output_preclipped
    assert term.output_depth is term.depth_buffer
    assert term.depth_buffer.shape == (NUM_ENVS, 2, 27, 48)
    assert term.far_plane == pytest.approx(2.0)
    # Nominal pose, identical across envs.
    assert torch.equal(term._local_pos_list[0], term._local_pos_list[0][:1].expand(NUM_ENVS, -1))


def test_shapes_and_delay_config() -> None:
    term = _make_term(_falcon_params())
    assert term.depth_buffer.shape == (NUM_ENVS, 2, 135, 240)
    assert term.processed_buffer.shape == (NUM_ENVS, 2, 27, 48)
    assert term.delay_buffer.shape == (NUM_ENVS, 4, 2, 27, 48)
    assert term.output_depth.shape == (NUM_ENVS, 2, 27, 48)
    assert torch.all(term.delay_idx == 2)
    assert term.needs_reset and term.output_preclipped
    assert term.far_plane == pytest.approx(2.5)


def test_delay_buffer_len_grows_to_cover_max_delay() -> None:
    props = replace(warp_dual_depth_cameras_falcon.terms["warp_depth"].params["props"], depth_buffer_len=2)
    term = _make_term(_falcon_params(props=props))
    assert term.delay_buffer.shape[1] == 3


def test_delay_buffer_shift_and_delayed_read() -> None:
    manager, term, env = _make_manager(_falcon_params())
    values = _install_fake_capture(term)
    fill = 0.5 * (0.1 + 2.0)

    # Initially the delayed frame is the "zero" (normalized) frame.
    assert torch.allclose(term.output_depth, torch.full_like(term.output_depth, fill))
    obs0 = cam_depth(env)
    assert obs0.shape == (NUM_ENVS, 2 * 27 * 48)
    assert obs0.abs().max() < 1e-6

    for k, v in enumerate([0.5, 1.0, 1.5, 1.8]):
        values.append(v)
        manager.render_cameras()  # interval 1 -> captures every call
        assert torch.allclose(term.delay_buffer[:, 0], torch.full_like(term.delay_buffer[:, 0], v))
        expected = [fill, fill, 0.5, 1.0][k]
        assert torch.allclose(term.output_depth, torch.full_like(term.output_depth, expected), atol=1e-6)

    # Buffer holds the last 4 captures, newest first.
    for i, v in enumerate([1.8, 1.5, 1.0, 0.5]):
        assert torch.allclose(term.delay_buffer[:, i], torch.full_like(term.delay_buffer[:, i], v))


def test_reset_zeroes_only_reset_envs_and_resamples_delay() -> None:
    props = replace(warp_dual_depth_cameras_falcon.terms["warp_depth"].params["props"], depth_delay_range=[0, 3])
    manager, term, env = _make_manager(_falcon_params(props=props))
    values = _install_fake_capture(term)
    fill = term._delay_fill_value
    for v in [0.5, 1.0, 1.5, 1.8]:
        values.append(v)
        manager.render_cameras()

    reset_ids = torch.tensor([1, 4], device=DEVICE)
    env.reset_buf[reset_ids] = 1
    torch.manual_seed(0)
    values.append(0.3)
    manager.render_cameras()

    keep = torch.tensor([0, 2, 3, 5], device=DEVICE)
    # Reset envs: cleared, then the new frame is inserted at index 0.
    assert torch.allclose(term.delay_buffer[reset_ids, 0], torch.full_like(term.delay_buffer[reset_ids, 0], 0.3))
    assert torch.allclose(term.delay_buffer[reset_ids, 1:], torch.full_like(term.delay_buffer[reset_ids, 1:], fill))
    # Other envs keep their history.
    for i, v in enumerate([0.3, 1.8, 1.5, 1.0]):
        assert torch.allclose(term.delay_buffer[keep, i], torch.full_like(term.delay_buffer[keep, i], v))
    assert term.delay_idx.min() >= 0 and term.delay_idx.max() <= 3

    # No resets flagged -> nothing cleared.
    env.reset_buf.zero_()
    values.append(0.7)
    manager.render_cameras()
    assert torch.allclose(term.delay_buffer[reset_ids, 1], torch.full_like(term.delay_buffer[reset_ids, 1], 0.3))


def test_resample_delay_respects_range() -> None:
    props = replace(warp_dual_depth_cameras_falcon.terms["warp_depth"].params["props"], depth_delay_range=[1, 3])
    term = _make_term(_falcon_params(props=props))
    term.reset(torch.arange(NUM_ENVS, device=DEVICE))
    assert term.delay_idx.min() >= 1 and term.delay_idx.max() <= 3


def test_resize_matches_torchvision() -> None:
    tv = pytest.importorskip("torchvision")
    x = torch.rand(NUM_ENVS, 2, 135, 240, device=DEVICE) * 1.9 + 0.1
    ref = tv.transforms.Resize((27, 48), interpolation=tv.transforms.InterpolationMode.BICUBIC)(x)
    out = resize_depth(x, (27, 48), mode="bicubic", antialias=True)
    assert out.shape == (NUM_ENVS, 2, 27, 48)
    assert torch.allclose(out, ref, atol=1e-6)


def test_processing_clips_before_resize_and_obs_keeps_overshoot() -> None:
    manager, term, env = _make_manager(
        _falcon_params(
            props=replace(warp_dual_depth_cameras_falcon.terms["warp_depth"].params["props"], depth_delay_range=[0, 0])
        )
    )
    term.depth_buffer.fill_(2.0)
    term.depth_buffer[..., :, :120] = 1000.0  # no-hit -> clipped to far
    term.depth_buffer[..., :, 120:] = -2.5  # dropout -> clipped to near
    term._update_processed()
    expected = resize_depth(term.depth_buffer.clamp(0.1, 2.0), (27, 48))
    assert torch.allclose(term.output_depth, expected)
    # Bicubic overshoot at the step edge survives (FALCON does not re-clip).
    assert term.output_depth.max() > 2.0 or term.output_depth.min() < 0.1
    obs = cam_depth(env)
    ref = ((expected - 0.1) / 1.9 - 0.5).reshape(NUM_ENVS, -1)
    assert torch.allclose(obs, ref, atol=1e-6)
    assert torch.allclose(cam_depth(env, clip=True), ref.clamp(-0.5, 0.5), atol=1e-6)


def test_noise_falcon_values() -> None:
    term = _make_term(_falcon_params())
    torch.manual_seed(0)
    term.depth_buffer.fill_(1.0)
    term._apply_noise()
    buf = term.depth_buffer
    assert buf.shape == (NUM_ENVS, 2, 135, 240)
    dropped = buf == -2.5
    assert dropped.float().mean().item() == pytest.approx(0.05, abs=0.005)
    kept = buf[~dropped]
    assert kept.mean().item() == pytest.approx(1.0, abs=0.005)
    assert kept.std().item() == pytest.approx(0.1, abs=0.005)


def test_pose_randomization_ranges() -> None:
    term = _make_term(_falcon_params())
    nominal = [torch.tensor([0.135, 0.06, 0.02]), torch.tensor([-0.125, -0.06, 0.02])]
    for s in range(2):
        delta = term._local_pos_list[s].cpu() - nominal[s]
        assert delta.abs().max() <= 0.025 + 1e-6
        assert term._local_pos_list[s].shape == (NUM_ENVS, 3)
        assert term._local_quat_list[s].shape == (NUM_ENVS, 4)
        assert torch.allclose(term._local_quat_list[s].norm(dim=-1), torch.ones(NUM_ENVS, device=DEVICE))
    # Envs differ from each other.
    assert not torch.allclose(term._local_pos_list[0][0], term._local_pos_list[0][1])


def test_pose_randomization_zero_range_matches_nominal() -> None:
    zero = {
        "enable": True,
        "translation_min": [0.0] * 3,
        "translation_max": [0.0] * 3,
        "rotation_min_deg": [0.0] * 3,
        "rotation_max_deg": [0.0] * 3,
    }
    rand_term = _make_term(_falcon_params(pose_randomization=zero))
    nominal_term = _make_term(_falcon_params(pose_randomization={"enable": False}))
    for s in range(2):
        assert torch.allclose(rand_term._local_pos_list[s], nominal_term._local_pos_list[s])
        assert torch.allclose(rand_term._local_quat_list[s], nominal_term._local_quat_list[s], atol=1e-6)


def test_fov_randomization_range() -> None:
    term = _make_term(_falcon_params())
    fov_deg = torch.rad2deg(term._fov_rad)
    assert fov_deg.shape == (NUM_ENVS, 2)
    assert fov_deg.min() >= 96.41 - 1e-3 and fov_deg.max() <= 106.41 + 1e-3
    # K_inv built at the 240x135 ray-cast resolution; vfov derived from hfov + aspect.
    k_inv = term._K_inv_array.reshape(NUM_ENVS, 2, 4, 4)
    k = torch.linalg.inv(k_inv)
    hfov = term._fov_rad
    f = 120.0 / torch.tan(hfov / 2)
    assert torch.allclose(k[..., 0, 0], f, rtol=1e-4)
    vfov = 2 * torch.atan(135.0 / (2 * f))
    assert torch.allclose(k[..., 1, 1], 67.5 / torch.tan(vfov / 2), rtol=1e-4)
    assert torch.allclose(k[..., 0, 0], k[..., 1, 1], rtol=1e-4)  # square pixels
    assert math.isclose(term._c_x, 120) and math.isclose(term._c_y, 67)


def test_occlusion_shared_across_cameras() -> None:
    term = _make_term(_falcon_params())
    # Minimal stand-in for setup(): terrain state + occlusion geometry.
    term._terrain_state = SimpleNamespace(
        terrain_type_names=["a", "b", "c", "d"],
        terrain_type_idx=torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE),
    )
    term._setup_occlusion()
    assert (term._occl_x1, term._occl_x2, term._occl_diff_x) == (90, 50, 30)  # native pixels, no scaling
    term.depth_buffer.fill_(5.0)
    torch.manual_seed(1)
    term._apply_random_depth_occlusion()
    left = term.depth_buffer[..., :, :60]  # covered by both small (60) and large (90) masks
    occluded = (left != 5.0).flatten(1).any(dim=1)
    assert occluded.any()
    assert torch.equal(term.depth_buffer[occluded, 0, :, :60], term.depth_buffer[occluded, 1, :, :60])


def test_capture_pipeline_on_flat_plane() -> None:
    """Ray-cast a flat ground plane end to end: kernel -> noise -> clip -> resize -> delay."""
    import numpy as np
    import warp as wp
    from holosoma.managers.camera.terms.warp_depth import _convert_to_warp_mesh

    term = _make_term(_falcon_params(random_depth_occlusion={"enable": False}))
    verts = np.array([[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = _convert_to_warp_mesh(verts, faces, DEVICE)
    term._terrain_wp_mesh = mesh
    term._terrain_mesh_id = mesh.id

    # Cameras 1 m above the ground on an identity-pose body, using the (randomized) local orientation.
    for s in range(2):
        term._global_pos[:, s] = torch.tensor([0.0, 0.0, 1.0], device=DEVICE)
        term._global_quat[:, s] = quat_mul(term._local_quat_list[s], term._frame_quat, w_last=True)

    term.capture()
    wp.synchronize()
    raw = term.depth_buffer
    assert raw.shape == (NUM_ENVS, 2, 135, 240)
    assert (raw == -2.5).float().mean().item() == pytest.approx(0.05, abs=0.01)
    out = term.processed_buffer
    assert out.shape == (NUM_ENVS, 2, 27, 48)
    assert torch.isfinite(out).all()
    # Pitched ~75 deg down from 1 m: the centre pixels see the ground at ~1/sin(75 deg) ~ 1.04 m.
    centre = term.depth_buffer.clamp(0.1, 2.0)[..., 60:75, 110:130].median()
    assert 0.8 < centre.item() < 1.3
    assert torch.allclose(term.delay_buffer[:, 0], out)
