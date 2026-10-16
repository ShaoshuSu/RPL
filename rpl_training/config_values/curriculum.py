"""Curriculum presets for G1 loco-manipulation."""

from copy import deepcopy

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

g1_29dof_falcon_hm_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "terrain_curriculum": CurriculumTermCfg(
            func="rpl_training.terms.curriculum:MotionTimeTerrainCurriculum",
            params={
                "enabled": True,
                "terrain_move_up_ratio": 0.3125,
                "terrain_move_down_ratio": 0.5,
            },
        ),
        "command_scale_curriculum": CurriculumTermCfg(
            func="rpl_training.terms.curriculum:CommandScaleCurriculum",
            params={
                "enabled": True,
                "command_height_scale_curriculum": True,
                "command_height_scale_initial_scale": 0.1,
                "command_height_scale_up_threshold": 210.0,
                "command_height_scale_down_threshold": 200.0,
                "command_height_scale_up": 0.02,
                "command_height_scale_down": 0.02,
                "command_height_scale_min": 0.0,
                "command_height_scale_max": 1.0,
                "waist_command_scale_curriculum": True,
                "waist_command_scale_initial_scale": 0.1,
                "waist_command_scale_up_threshold": 210.0,
                "waist_command_scale_down_threshold": 200.0,
                "waist_command_scale_up": 0.02,
                "waist_command_scale_down": 0.02,
                "waist_command_scale_min": 0.0,
                "waist_command_scale_max": 1.0,
                "upper_body_motion_scale_curriculum": True,
                "upper_body_motion_scale_initial_scale": 0.1,
                "upper_body_motion_scale_up_threshold": 210.0,
                "upper_body_motion_scale_down_threshold": 200.0,
                "upper_body_motion_scale_up": 0.02,
                "upper_body_motion_scale_down": 0.02,
                "upper_body_motion_scale_min": 0.0,
                "upper_body_motion_scale_max": 1.0,
                "resample_after_update": True,
            },
        ),
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.1,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 40.0,
                "level_up_threshold": 210.0,
                "degree": 0.00002,
            },
        ),
        "domain_rand_scale_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:DomainRandScaleCurriculum",
            params={
                "enabled": True,
                "initial_scale": 0.1,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 40.0,
                "level_up_threshold": 210.0,
                "degree": 0.00002,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)


# Force-enabled variant: same as ``g1_29dof_falcon_hm_curriculum`` plus the
# ``ForceScaleCurriculum`` term that drives ``env.apply_force_scale``.
def _build_force_curriculum_setup_terms() -> dict:
    """Insert ``force_scale_curriculum`` before ``average_episode_tracker``.

    Keep episode-length-driven scale curricula grouped before the tracker so
    their logged values reflect the same reset batch that updates the average.
    """
    base = g1_29dof_falcon_hm_curriculum.setup_terms
    out: dict = {}
    for name, term in base.items():
        if name == "average_episode_tracker":
            out["force_scale_curriculum"] = CurriculumTermCfg(
                func="rpl_training.terms.curriculum:ForceScaleCurriculum",
                params={
                    "enabled": True,
                    "initial_scale": 0.1,
                    "up_threshold": 210.0,
                    "down_threshold": 200.0,
                    "scale_up": 0.02,
                    "scale_down": 0.02,
                    "min_scale": 0.0,
                    "max_scale": 1.0,
                },
            )
        out[name] = term
    return out


g1_29dof_falcon_hm_force_curriculum = CurriculumManagerCfg(
    params=g1_29dof_falcon_hm_curriculum.params,
    setup_terms=_build_force_curriculum_setup_terms(),
    reset_terms={},
    step_terms={},
)


# FAR-FALCON-Terrains ``+exp=falcon_hm_dagger_moe`` parity (resolved env config):
# 0.05 scale steps (command height steps down by 0.025), upper-body motion scale
# starting at 0, a 10000-episode length average, and no penalty / domain-rand
# scale curricula (both off in FALCON, i.e. full penalty weights and DR scale 1).
def _build_falcon_distill_setup_terms() -> dict:
    out = deepcopy(_build_force_curriculum_setup_terms())
    cmd = out["command_scale_curriculum"].params
    cmd["command_height_scale_up"] = 0.05
    cmd["command_height_scale_down"] = 0.025
    cmd["waist_command_scale_up"] = 0.05
    cmd["waist_command_scale_down"] = 0.05
    cmd["upper_body_motion_scale_initial_scale"] = 0.0
    cmd["upper_body_motion_scale_up"] = 0.05
    cmd["upper_body_motion_scale_down"] = 0.05
    # The command reset that follows the curriculum update already resamples.
    cmd["resample_after_update"] = False
    force = out["force_scale_curriculum"].params
    force["scale_up"] = 0.05
    force["scale_down"] = 0.05
    out["penalty_curriculum"].params["enabled"] = False
    out["domain_rand_scale_curriculum"].params["enabled"] = False
    return out


g1_29dof_rpl_curriculum_distill = CurriculumManagerCfg(
    params={"num_compute_average_epl": 10000},
    setup_terms=_build_falcon_distill_setup_terms(),
    reset_terms={},
    step_terms={},
)


# FAR-FALCON-Terrains ``+exp=falcon_hm_multi_terrain`` parity (RL experts): 0.02 scale
# steps (base), upper-body motion scale starting at 0, a 10000-episode length average,
# and the penalty / domain-rand scale curricula on with a 1e-5 degree (FALCON's exp
# default is 2.5e-5; its stones expert overrides both to 1e-5 on the CLI).
def _build_falcon_setup_terms() -> dict:
    out = deepcopy(_build_force_curriculum_setup_terms())
    cmd = out["command_scale_curriculum"].params
    cmd["upper_body_motion_scale_initial_scale"] = 0.0
    cmd["resample_after_update"] = False
    out["penalty_curriculum"].params["degree"] = 1e-5
    out["domain_rand_scale_curriculum"].params["degree"] = 1e-5
    return out


g1_29dof_rpl_curriculum_expert = CurriculumManagerCfg(
    params={"num_compute_average_epl": 10000},
    setup_terms=_build_falcon_setup_terms(),
    reset_terms={},
    step_terms={},
)

__all__ = [
    "g1_29dof_falcon_hm_curriculum",
    "g1_29dof_falcon_hm_force_curriculum",
    "g1_29dof_rpl_curriculum_expert",
    "g1_29dof_rpl_curriculum_distill",
]
