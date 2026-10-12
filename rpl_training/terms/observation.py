"""Loco-manipulation observation terms (RPL HM / decoupled layout)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def command_stand(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Stand / locomotion gate from command index 4 (same semantics as RPL). Shape ``[N, 1]``."""
    return env.command_manager.commands[:, 4:5]


def command_waist_dofs(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Commanded waist yaw / roll / pitch (indices 5:8). Shape ``[N, 3]``."""
    return env.command_manager.commands[:, 5:8]


def command_base_height(env: LeggedRobotLocomotionManager, absolute: bool = False) -> torch.Tensor:
    """Base-height delta (index 8, meters) relative to ``env.command_base_height_nominal``. Zero while walking.

    With ``absolute=True`` the nominal height is added back, matching FAR-FALCON's absolute
    ``commands[:, 8]`` (``desired_base_height`` while walking); needed to feed FALCON-trained policies.
    """
    cmd = env.command_manager.commands[:, 8:9]
    if absolute:
        cmd = cmd + env.command_base_height_nominal
    return cmd


def ref_upper_dof_pos(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Reference upper-body joint positions (absolute), from ``UpperBodyRefPoseCommand``."""
    if not hasattr(env, "ref_upper_dof_pos"):
        raise RuntimeError("ref_upper_dof_pos observation requires UpperBodyRefPoseCommand (env.ref_upper_dof_pos).")
    return env.ref_upper_dof_pos


def height_measurements(
    env: LeggedRobotLocomotionManager,
    min_height: float = -1.0,
    max_height: float = 1.0,
) -> torch.Tensor:
    """RPL-style relative height grid under the base (flattened per env).

    Terrain heights are queried from the terrain term and converted to
    root-relative depths: ``root_z - lidar_offset - terrain_z``, clamped to
    ``[min_height, max_height]``. Mirrors FAR-FALCON's
    ``min_measured_height`` / ``max_measured_height`` terrain knobs (default
    there is ``[-0.5, 0.5]``); Holosoma's legacy default is the wider
    ``[-1.0, 1.0]`` for backward compatibility.
    """
    terrain = env.terrain_manager.get_state("locomotion_terrain")
    if terrain is None or not hasattr(terrain, "height_measurements"):
        raise RuntimeError(
            "height_measurements requires terrain term locomotion_terrain with height_measurements "
            "(e.g. TerrainLocomotion)."
        )
    terrain_z = cast("torch.Tensor", terrain.height_measurements())
    root_z = env.simulator.robot_root_states[:, 2:3]
    lidar = float(getattr(env, "lidar_height_offset", 0.5))
    rel = root_z - lidar - terrain_z
    rel = torch.nan_to_num(rel, nan=0.0, posinf=max_height, neginf=min_height)
    return rel.clamp(min_height, max_height)


def cam_depth(
    env: LeggedRobotLocomotionManager,
    camera_term: str = "warp_depth",
    near_clip: float | None = None,
    far_clip: float | None = None,
    clip: bool | None = None,
) -> torch.Tensor:
    """Flattened depth images from the camera manager. Shape ``[N, S * H * W]``.

    Mirrors FAR-FALCON-Terrains' ``_pre_process_depth``: the raw metric depth
    buffer is clipped to ``[near_clip, far_clip]`` then mapped to a
    **zero-centered** ``[-0.5, 0.5]`` range via
    ``(d - near_clip) / (far_clip - near_clip) - 0.5``.

    ``near_clip``/``far_clip`` default to the camera term's
    ``props.near_clip``/``props.far_clip`` (falling back to ``far_plane`` for
    the far bound), so a depth policy trained under FALCON sees identical
    observation values here.

    The term's ``output_depth`` (delayed / resized frame) is read when it
    exists, else ``depth_buffer``.  ``clip=None`` (default) clamps unless the
    term reports ``output_preclipped`` -- FALCON clips *before* the bicubic
    resize and normalizes the resized image without clipping again, so bicubic
    overshoot slightly outside ``[-0.5, 0.5]`` is kept.
    """
    if env.camera_manager is None or not env.camera_manager.has_cameras:
        raise RuntimeError("cam_depth observation requires a camera manager with at least one term.")
    term = env.camera_manager.get_state(camera_term)
    buf = getattr(term, "output_depth", None)
    if buf is None:
        buf = getattr(term, "depth_buffer", None)
    if buf is None:
        raise RuntimeError(f"camera term '{camera_term}' has no depth_buffer attribute.")
    if clip is None:
        clip = not getattr(term, "output_preclipped", False)

    props = getattr(term, "props", None)
    if near_clip is None:
        near_clip = getattr(props, "near_clip", 0.0)
    if far_clip is None:
        far_clip = getattr(props, "far_clip", None)
        if far_clip is None:
            far_clip = getattr(term, "far_plane", 1.0)
    near = float(near_clip)
    far = float(far_clip)
    span = (far - near) or 1.0

    depth = buf.clamp(near, far) if clip else buf
    normalized = (depth - near) / span - 0.5
    return normalized.reshape(buf.shape[0], -1)


def terrain_type_idx(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Per-env terrain-type index, used by the MoE teacher's routing layer.

    Mirrors FAR-FALCON-Terrains' ``terrain_idx_obs``. Returns a ``[N, 1]`` long
    tensor cast to float so it can flow through the standard observation pipe.
    """
    terrain = env.terrain_manager.get_state("locomotion_terrain")
    if terrain is None or not hasattr(terrain, "terrain_type_idx"):
        raise RuntimeError(
            "terrain_type_idx observation requires terrain term locomotion_terrain (e.g. TerrainLocomotion)."
        )
    return terrain.terrain_type_idx.view(-1, 1).to(dtype=torch.float32)


def ee_apply_force(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """External EE forces in base frame, ``[N, 6]`` (left xyz, right xyz).

    RPL WBC uses sampled hand forces; single-policy HM stacks zeros when not simulated.
    ``LeggedRobotLocoManipForceManager`` already caches ``{left,right}_ee_apply_force`` in the
    base frame, so they are returned as-is (rotating them again doubled the yaw).
    """
    if hasattr(env, "left_ee_apply_force") and hasattr(env, "right_ee_apply_force"):
        return torch.cat([env.left_ee_apply_force, env.right_ee_apply_force], dim=-1)
    return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)


__all__ = [
    "cam_depth",
    "command_base_height",
    "command_stand",
    "command_waist_dofs",
    "ee_apply_force",
    "height_measurements",
    "ref_upper_dof_pos",
    "terrain_type_idx",
]
