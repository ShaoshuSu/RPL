"""Loco-manipulation termination presets for the G1 robot (RPL HM-style).

Mirrors ``falcon_hm_moe.yaml`` termination flags and scales for gravity tilt and
base height (``terminate_by_gravity``, ``terminate_by_low_height``, and
``termination_scales`` thresholds).
"""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_falcon_hm_termination = TerminationManagerCfg(
    terms={
        "gravity": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:gravity_tilt_exceeded",
            params={
                "threshold_x": 0.8,
                "threshold_y": 0.8,
            },
        ),
        "low_height": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:base_height_below_threshold",
            params={
                "min_height": 0.3,
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "motion_end": TerminationTermCfg(
            func="rpl_training.terms.termination:motion_ends",
            is_timeout=True,
        ),
    }
)

__all__ = [
    "g1_29dof_falcon_hm_termination",
]
