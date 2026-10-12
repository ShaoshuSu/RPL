"""Domain randomization presets for G1 loco-manipulation (RPL HM-style)."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

g1_29dof_falcon_hm_randomization = RandomizationManagerCfg(
    setup_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
            params={
                "push_interval_s": [5, 10],
                "max_push_vel": [1.0, 1.0],
                "enabled": False,
            },
        ),
        "setup_action_delay_buffers": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
            params={
                "ctrl_delay_step_range": [0, 1],
                "enabled": False,
            },
        ),
        "setup_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
            params={
                "enabled": False,
                "rfi_lim": 0.1,
            },
        ),
        "setup_dof_pos_bias": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
            params={
                "dof_pos_bias_range": [-0.01, 0.01],
                "enabled": False,
            },
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
            params={
                "kp_range": [0.9, 1.1],
                "kd_range": [0.9, 1.1],
                "rfi_lim_range": [0.5, 1.5],
                "enable_pd_gain": False,
                "enable_rfi_lim": False,
            },
        ),
        "height_measurements_xy_offset": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:HeightMeasurementXYOffsetRandomizer",
            params={
                "enabled": False,
                "height_measurements_xy_offset_points_range": [0, 1],
            },
        ),
        "mass_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
            params={
                "enable_link_mass": True,
                "link_mass_range": [0.9, 1.2],
                "enable_base_mass": True,
                "added_mass_range": [-1.0, 3.0],
            },
        ),
        "randomize_friction_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_friction_startup",
            params={
                "friction_range": [0.1, 1.0],
                "enabled": True,
            },
        ),
        "randomize_base_com_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
            params={
                "base_com_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
                "enabled": False,
            },
        ),
    },
    reset_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
        ),
        "height_measurements_xy_offset": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:HeightMeasurementXYOffsetRandomizer"
        ),
        "randomize_push_schedule": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
        ),
        "randomize_action_delay": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
        ),
        "randomize_dof_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
            params={
                "joint_pos_scale_range": [0.5, 1.5],
                "joint_pos_bias_range": [0.0, 0.0],
                "joint_vel_range": [0.0, 0.0],
                "randomize_dof_pos_bias": False,
            },
        ),
        "configure_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:configure_torque_rfi",
        ),
    },
    step_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "apply_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
        ),
    },
)


def _with_params(term: RandomizationTermCfg, **params) -> RandomizationTermCfg:
    return replace(term, params={**term.params, **params})


_setup = g1_29dof_falcon_hm_randomization.setup_terms

# FAR-FALCON-Terrains ``domain_rand`` as resolved by ``+exp=falcon_hm_dagger_moe``
# (the setting the MoE-CNN teachers and the depth student were trained with):
#   - ctrl delay 0-1 step on the upper body (arms) only, resampled per reset
#   - friction U[-0.1, 1.0] (64 buckets, FALCON passes negatives through as-is)
#   - base COM offset x +/-0.025, y/z +/-0.05 on torso_link
#   - added base mass U[-1, 3] kg on pelvis; link mass scale U[0.9, 1.2]
g1_29dof_falcon_hm_randomization_falcon_distill = replace(
    g1_29dof_falcon_hm_randomization,
    setup_terms={
        **_setup,
        "setup_action_delay_buffers": _with_params(
            _setup["setup_action_delay_buffers"],
            ctrl_delay_step_range=[0, 1],
            enabled=True,
            upper_body_ctrl_delay=True,
            lower_body_ctrl_delay=False,
        ),
        "mass_randomizer": _with_params(_setup["mass_randomizer"], base_mass_body_name="pelvis"),
        "randomize_friction_startup": _with_params(
            _setup["randomize_friction_startup"], friction_range=[-0.1, 1.0], enabled=True
        ),
        "randomize_base_com_startup": _with_params(
            _setup["randomize_base_com_startup"],
            base_com_range={"x": [-0.025, 0.025], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
            enabled=True,
        ),
    },
)

# FAR-FALCON-Terrains ``domain_rand_unitree_rl_gym_g1_falcon_hm`` as resolved by
# ``+exp=falcon_hm_multi_terrain`` (RL experts). Same as the distill preset except
# friction U[0.25, 1.25], base COM off by default (the slopes / stones experts enable it
# on the CLI), and the height-scan XY offset scaled with FALCON's ``int()`` truncation by
# the DR-scale curriculum (the stairs expert enables it on the CLI).
g1_29dof_falcon_hm_randomization_falcon = replace(
    g1_29dof_falcon_hm_randomization_falcon_distill,
    setup_terms={
        **g1_29dof_falcon_hm_randomization_falcon_distill.setup_terms,
        "height_measurements_xy_offset": _with_params(
            _setup["height_measurements_xy_offset"], truncate_scaled_range=True
        ),
        "randomize_friction_startup": _with_params(
            _setup["randomize_friction_startup"], friction_range=[0.25, 1.25], enabled=True
        ),
        "randomize_base_com_startup": _with_params(
            g1_29dof_falcon_hm_randomization_falcon_distill.setup_terms["randomize_base_com_startup"],
            enabled=False,
        ),
    },
)


def _falcon_expert_randomization(friction_range: list[float], height_xy_offset: bool = False):
    """FALCON's final expert stages (wandb run args): ``domain_rand.friction_range``,
    ``randomize_base_com=True`` and, for stairs, ``randomize_height_measurements_xy_offset_points``."""
    terms = g1_29dof_falcon_hm_randomization_falcon.setup_terms
    return replace(
        g1_29dof_falcon_hm_randomization_falcon,
        setup_terms={
            **terms,
            "randomize_friction_startup": _with_params(
                terms["randomize_friction_startup"], friction_range=friction_range
            ),
            "randomize_base_com_startup": _with_params(terms["randomize_base_com_startup"], enabled=True),
            "height_measurements_xy_offset": _with_params(
                terms["height_measurements_xy_offset"], enabled=height_xy_offset
            ),
        },
    )


g1_29dof_falcon_hm_randomization_falcon_stones = _falcon_expert_randomization([-0.6, 1.0])
g1_29dof_falcon_hm_randomization_falcon_stairs = _falcon_expert_randomization([-0.6, 1.0], height_xy_offset=True)
g1_29dof_falcon_hm_randomization_falcon_slopes = _falcon_expert_randomization([-0.1, 1.0])

__all__ = [
    "g1_29dof_falcon_hm_randomization",
    "g1_29dof_falcon_hm_randomization_falcon",
    "g1_29dof_falcon_hm_randomization_falcon_distill",
    "g1_29dof_falcon_hm_randomization_falcon_slopes",
    "g1_29dof_falcon_hm_randomization_falcon_stairs",
    "g1_29dof_falcon_hm_randomization_falcon_stones",
]
