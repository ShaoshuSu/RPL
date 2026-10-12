"""Command presets for G1 loco-manipulation (upper-body reference + locomotion)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

g1_29dof_falcon_hm_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 10.0,
    },
    setup_terms={
        "locomotion_command": CommandTermCfg(
            func="rpl_training.terms.command:HeadingLocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-1.0, 1.0],
                    "lin_vel_y": [-1.0, 1.0],
                    "ang_vel_yaw": [-1.0, 1.0],
                    "heading": [-3.14, 3.14],
                    "base_height": [-0.3, 0.0],
                    "waist_yaw": [-1.0, 1.0],
                    "waist_roll": [-0.0, 0.0],
                    "waist_pitch": [-0.52, 1.57],
                },
                "default_base_height": 0.75,
                "stand_prob": 0.3,
                "heading_p_gain": 0.5,
                "lin_vel_deadband": 0.2,
                # FALCON-style per-terrain command masking. Terrain-type names not
                # present in the active ``terrain_config`` are silently skipped.
                # Empty list zeroes the command on every terrain; ``None`` disables masking.
                "lin_vel_y_terrain_mask": [],
                "heading_terrain_mask": [
                    "slopes",
                    "stairs_up",
                    "stairs_down",
                ],
                "ang_vel_yaw_terrain_mask": [
                    "slopes",
                    "stairs_up",
                    "stairs_down",
                    "stepping_stones_columns",
                ],
            },
        ),
        "upper_body_ref_pose": CommandTermCfg(
            func="rpl_training.terms.command:UpperBodyRefPoseCommand",
        ),
    },
    reset_terms={
        "locomotion_command": CommandTermCfg(func="rpl_training.terms.command:HeadingLocomotionCommand"),
        "upper_body_ref_pose": CommandTermCfg(func="rpl_training.terms.command:UpperBodyRefPoseCommand"),
    },
    step_terms={
        "locomotion_command": CommandTermCfg(func="rpl_training.terms.command:HeadingLocomotionCommand"),
        "upper_body_ref_pose": CommandTermCfg(func="rpl_training.terms.command:UpperBodyRefPoseCommand"),
    },
)


# FAR-FALCON-Terrains task parity (resolved ``env`` config of FALCON's exps):
# FALCON's double stance draw (``(1 - p) * p`` tapping envs), per-terrain
# ``stand_prob`` (FALCON order [flat, curb, stones, slope_up, slope_down, up, down, gap]),
# per-terrain yaw / heading masks, zero-fix waist probabilities, frozen upper-body
# reference (``fix_upper_body_prob``), upper body reset onto the motion, and the FALCON
# EE-force config (read by ``LeggedRobotLocoManipForceManager`` from
# ``params["ee_force_config"]``). Holosoma has no flat terrain, so FALCON's
# ``lin_vel_y_commands_mask=[flat]`` is the base ``[]``.
def _with_falcon_task(
    cfg: CommandManagerCfg,
    *,
    stand_prob_terrain: dict[str, float],
    ang_vel_yaw_terrain_mask: list[str],
    heading_terrain_mask: list[str],
    zero_fix_waist_yaw_prob: float,
    apply_force_z_range: tuple[float, float],
    dof_effort_limit_scale: float,
) -> CommandManagerCfg:
    setup_terms = deepcopy(cfg.setup_terms)
    loco = setup_terms["locomotion_command"]
    loco.params["falcon_stance_resample"] = True
    loco.params["stand_flag_from_velocity"] = False
    loco.params["stand_prob_terrain"] = dict(stand_prob_terrain)
    loco.params["ang_vel_yaw_terrain_mask"] = list(ang_vel_yaw_terrain_mask)
    loco.params["heading_terrain_mask"] = list(heading_terrain_mask)
    loco.params["command_ranges"]["waist_pitch"] = [-0.5, 1.57]
    loco.params["zero_fix_waist_yaw_prob"] = zero_fix_waist_yaw_prob
    loco.params["zero_fix_waist_roll_prob"] = 1.0
    loco.params["zero_fix_waist_pitch_prob"] = 0.25
    setup_terms["upper_body_ref_pose"] = replace(
        setup_terms["upper_body_ref_pose"],
        params={
            "fix_upper_body_prob": 0.3,
            "motion_time_lookahead_steps": 1,
            "reset_dof_to_motion": True,
            "reset_pose_use_prev_episode_time": True,
        },
    )
    params = {
        **cfg.params,
        "ee_force_config": {
            "apply_force_z_range": apply_force_z_range,
            "zero_force_prob": (1.0, 1.0, 0.25),
            "dof_effort_limit_scale": dof_effort_limit_scale,
            "apply_force_pos_mode": "hand_to_extend",
            "apply_force_pos_ratio_range": (0.0, 1.0),
            "resample_force_with_commands": True,
        },
    }
    return replace(cfg, params=params, setup_terms=setup_terms)


# ``+exp=falcon_hm_dagger_moe`` (depth distillation): stand_prob .4 / .2 / .2 / .2 on
# slopes / stairs_up / stairs_down / stones, no yaw on stones, force z in [-100, 5].
g1_29dof_falcon_hm_command_falcon_distill = _with_falcon_task(
    g1_29dof_falcon_hm_command,
    stand_prob_terrain={
        "slopes": 0.4,
        "stairs_up": 0.2,
        "stairs_down": 0.2,
        "stepping_stones_columns": 0.2,
    },
    ang_vel_yaw_terrain_mask=["slopes", "stairs_up", "stairs_down"],
    heading_terrain_mask=["slopes", "stairs_up", "stairs_down"],
    zero_fix_waist_yaw_prob=0.1,
    apply_force_z_range=(-100.0, 5.0),
    dof_effort_limit_scale=1.0,
)

# ``+exp=falcon_hm_multi_terrain`` (RL experts): stand_prob .4 on slopes, .3 on stairs /
# stones, no yaw on stones, waist yaw zero-fixed with prob .25, force z in [-60, 5] with a
# 1.0 effort-limit budget. Lists both Holosoma's 50/50 ``slopes`` bucket and the
# directional ``slope_up`` / ``slope_down`` types FALCON trains its slopes expert on.
_FALCON_SLOPES = ["slopes", "slope_up", "slope_down"]
g1_29dof_falcon_hm_command_falcon = _with_falcon_task(
    g1_29dof_falcon_hm_command,
    stand_prob_terrain={
        **dict.fromkeys(_FALCON_SLOPES, 0.4),
        "stairs_up": 0.3,
        "stairs_down": 0.3,
        "stepping_stones_columns": 0.3,
    },
    ang_vel_yaw_terrain_mask=[*_FALCON_SLOPES, "stairs_up", "stairs_down"],
    heading_terrain_mask=[*_FALCON_SLOPES, "stairs_up", "stairs_down"],
    zero_fix_waist_yaw_prob=0.25,
    apply_force_z_range=(-60.0, 5.0),
    # The experts' final stages set ``robot.dof_effort_limit_scale=1.0`` (the yaml has 0.8),
    # which FALCON only uses for this hand-force budget.
    dof_effort_limit_scale=1.0,
)

__all__ = [
    "g1_29dof_falcon_hm_command",
    "g1_29dof_falcon_hm_command_falcon",
    "g1_29dof_falcon_hm_command_falcon_distill",
]
