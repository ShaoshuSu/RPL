"""FAR-FALCON-Terrains sim2sim environment for Holosoma's MuJoCo ``run_sim``.

Mirrors FALCON's ``sim2real/run_sim.py task=loco_manip/loco_manip_cam robot=g1/g1_29dof``
with ``ROBOT_SCENE=humanoidverse/data/robots/g1/scene_g1_29dof_terrains_freebase_real.xml``:

* scene: FALCON's course (stairs, two-column stepping stones, platforms, 10-degree slope) and
  FALCON's G1 model, spawned where FALCON spawns it (on the stair platform);
* physics + PD at 200 Hz (FALCON's ``SIMULATE_DT=0.005``), torques clipped to FALCON's
  ``MOTOR_EFFORT_LIMIT_LIST``;
* front + back depth cameras at FALCON's sim poses (``cam_front_depth`` / ``cam_back_depth``),
  240x135, vertical FOV 69 deg, rendered at 10 Hz and published as one ``[2, 1, 27, 48]`` block
  with FALCON's image-server processing (cv2 bicubic resize, clip to [0.1, 2.0], normalize to
  [-0.5, 0.5], 1-frame delay, FALCON's ``depth_delay``).
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.plugin import DepthShmPluginConfig
from holosoma.config_types.sensor import CameraSensorConfig, SensorMountConfig
from holosoma.config_types.terrain import MeshType
from holosoma.config_values.plugin import PLUGIN_REGISTRY
from holosoma.config_values.robot import ROBOT_REGISTRY, g1_29dof
from holosoma.config_values.run_sim import RUN_SIM_REGISTRY, mujoco
from holosoma.config_values.sensor import CAMERA_REGISTRY
from holosoma.config_values.terrain import TERRAIN_REGISTRY, terrain_locomotion_plane

# FALCON ``g1_29dof.yaml`` ``MOTOR_EFFORT_LIMIT_LIST`` (MJCF actuator ctrlrange), motor order.
_FALCON_EFFORT_LIMITS = (
    [88.0, 88.0, 88.0, 139.0, 50.0, 50.0] * 2 + [88.0, 50.0, 50.0] + ([25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * 2)
)

# Robot: FALCON's scene MJCF (it includes FALCON's G1 model). ``robot_type`` stays ``g1_29dof``
# for the Unitree bridge. FALCON spawns the pelvis on the stair platform (``pos="2.5 -2 1.893"``);
# override ``--robot.init-state.pos`` to start elsewhere (e.g. ``[0, 0, 0.793]`` in front of it).
# FALCON's MJCF has no keyframe, so its joints spawn at zero.
g1_29dof_falcon_sim2sim = replace(
    g1_29dof,
    dof_effort_limit_list=_FALCON_EFFORT_LIMITS,
    init_state=replace(
        g1_29dof.init_state,
        pos=[2.5, -2.0, 1.893],
        default_joint_angles=dict.fromkeys(g1_29dof.init_state.default_joint_angles, 0.0),
    ),
    asset=replace(g1_29dof.asset, xml_file="g1/scene_g1_29dof_falcon_terrains_real.xml"),
)

# The scene MJCF brings its own floor, course, and lights. ``fake`` keeps a flat terrain state
# for the terrain manager while the MuJoCo scene manager adds no terrain geometry.
terrain_falcon_sim2sim = replace(
    terrain_locomotion_plane,
    terrain_term=replace(
        terrain_locomotion_plane.terrain_term,
        mesh_type=MeshType.FAKE,
        terrain_config={"flat": 1.0},
        num_rows=1,
        num_cols=1,
    ),
)

# Physics and the bridge's PD loop at 200 Hz with the scene's 50 solver iterations, as in FALCON.
# Xbox pad on by default (FALCON's ``task.USE_JOYSTICK=true``): pygame reads it here and the bridge
# publishes it as the Unitree wireless controller. The gantry anchor sits at FALCON's
# ``ElasticBand.point`` height (3.5 m).
mujoco_falcon_sim2sim = replace(
    mujoco,
    config=replace(
        mujoco.config,
        sim=replace(mujoco.config.sim, fps=200),
        mujoco_solver_iterations=50,
        mujoco_depth_gymnasium=True,  # depth from gymnasium's MujocoRenderer, as FALCON's image server
        mujoco_offscreen_samples=4,  # no MSAA: edge depth = pixel-centre surface, like the ray-cast training camera
        debug_viz=False,  # no camera frusta / depth-range overlay in the viewer (FALCON draws none)
        bridge=replace(mujoco.config.bridge, use_joystick=True, joystick_type="xbox"),
        virtual_gantry=replace(mujoco.config.virtual_gantry, height=3.5),
    ),
)


def _falcon_depth_camera(position: list[float], orientation: list[float]) -> CameraSensorConfig:
    # FALCON renders with MuJoCo's default clip planes scaled by the scene extent (2.0).
    return CameraSensorConfig(
        mount=SensorMountConfig(
            target_kind="robot_link", target="torso_link", position=position, orientation=orientation
        ),
        width=240,
        height=135,
        vertical_fov=69.0,
        near=0.02,
        far=100.0,
        data_types=["depth"],
    )


# Poses compiled from FALCON's MJCF (``euler="0 -0.2618 -1.57"`` / ``"0 0.2618 1.57"``): about
# 75 deg below horizontal, looking forward and backward. Training used x = 0.135 / -0.125.
falcon_front_depth = _falcon_depth_camera([0.15, 0.06, 0.02], [0.70133644, 0.09225941, -0.09233291, -0.70077817])
falcon_back_depth = _falcon_depth_camera([-0.13, -0.06, 0.02], [0.70133644, 0.09225941, 0.09233291, 0.70077817])

FALCON_DEPTH_CAMERAS = ("front_depth", "back_depth")
"""Sensor keys the launcher binds the two cameras to; the plugin publishes them in this order."""

depth_shm_falcon = DepthShmPluginConfig(
    cameras=list(FALCON_DEPTH_CAMERAS),
    resized_width=48,
    resized_height=27,
    near_clip=0.1,
    far_clip=2.0,
    empty_threshold=0.0,
    resize_mode="cv2_cubic",
    clip_before_resize=False,
    render_hz=10.0,
    latency_frames=1,
)


def register() -> None:
    """Add the presets to Holosoma's registries (idempotent). Dict-field presets use CLI names."""
    RUN_SIM_REGISTRY.add("mujoco_falcon_sim2sim", mujoco_falcon_sim2sim)
    ROBOT_REGISTRY.add("g1_29dof_falcon_sim2sim", g1_29dof_falcon_sim2sim)
    TERRAIN_REGISTRY.add("terrain_falcon_sim2sim", terrain_falcon_sim2sim)
    CAMERA_REGISTRY.add("falcon-front-depth", falcon_front_depth)
    CAMERA_REGISTRY.add("falcon-back-depth", falcon_back_depth)
    PLUGIN_REGISTRY.add("depth-shm-falcon", depth_shm_falcon)


register()

__all__ = [
    "FALCON_DEPTH_CAMERAS",
    "depth_shm_falcon",
    "falcon_back_depth",
    "falcon_front_depth",
    "g1_29dof_falcon_sim2sim",
    "mujoco_falcon_sim2sim",
    "terrain_falcon_sim2sim",
]
