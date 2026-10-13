"""Reward presets for G1 loco-manipulation (RPL HM-style)."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.reward import MultiAgentRewardManagerCfg, MultiAgentRewardTermCfg

g1_29dof_falcon_hm_reward = MultiAgentRewardManagerCfg(
    only_positive_rewards=False,
    multi_agent_body_keys=("lower_body", "upper_body"),
    terms={
        # Lower body rewards
        "tracking_lin_vel_x": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_lin_vel_x",
            weight=4.0,
            params={"tracking_sigma": 0.25},
            ma_reward_group="lower_body",
        ),
        "tracking_lin_vel_y": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_lin_vel_y",
            weight=1.5,
            params={"tracking_sigma": 0.25},
            ma_reward_group="lower_body",
        ),
        "tracking_ang_vel": MultiAgentRewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=4.0,
            params={"tracking_sigma": 0.25},
            ma_reward_group="lower_body",
        ),
        "tracking_heading": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_heading",
            weight=2.0,
            params={
                "tracking_sigma": 0.25,
                "terrain_type_mask": ["stepping_stones_columns"],
            },
            ma_reward_group="lower_body",
        ),
        "tracking_stance_base_height": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_stance_base_height",
            weight=4.0,
            params={"tracking_sigma": 0.05},
            ma_reward_group="lower_body",
        ),
        "tracking_torso_orientation_tapping": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_torso_orientation_tapping",
            weight=1.0,
            params={"tracking_sigma": 0.05},
            ma_reward_group="shared",
        ),
        "tracking_torso_orientation_stance": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_torso_orientation_stance",
            weight=4.0,
            params={"tracking_sigma": 0.05},
            ma_reward_group="shared",
        ),
        "terrain_progress": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:terrain_progress",
            weight=2.0,
            params={},
            ma_reward_group="shared",
        ),
        "penalty_lin_vel_z": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_lin_vel_z",
            weight=-2.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_ang_vel_xy": MultiAgentRewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.05,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_orientation": MultiAgentRewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-1.5,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_base_roll_orientation": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_base_roll_orientation",
            weight=-1.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_base_pitch_orientation": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_base_pitch_orientation",
            weight=-1.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_lower_body_torques": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_lower_body_torques",
            weight=-1e-5,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_lower_body_dof_vel": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_lower_body_dof_vel",
            weight=-1.0e-4,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_lower_body_dof_acc": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_lower_body_dof_acc",
            weight=-2.5e-7,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_lower_body_action_rate": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_lower_body_action_rate",
            weight=-0.5,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "limits_lower_body_dof_pos": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:limits_lower_body_dof_pos",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        # FALCON's ``reward_g1_29dof_falcon_multi_terrain.yaml`` comments these
        # two out (weight 0) but still tags them for ``reward_penalty_curriculum``.
        # Registered here at weight 0 so users can enable via the CLI.
        "limits_lower_body_dof_vel": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:limits_lower_body_dof_vel",
            weight=0.0,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "limits_lower_body_torque": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:limits_lower_body_torque",
            weight=0.0,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_feet_contact_forces": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_feet_contact_forces",
            weight=-0.01,
            params={
                "locomotion_max_contact_force": 350.0,
                "terrain_type_mask": ["slopes", "stairs_up", "stairs_down"],
            },
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_stumble": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_stumble",
            weight=-10.0,
            params={
                "terrain_type_mask": ["stepping_stones_columns", "stairs_up", "stairs_down"],
                "stumble_ratio": 0.5,
            },
            ma_reward_group="lower_body",
        ),
        "penalty_contact_no_vel": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_contact_no_vel",
            weight=-0.2,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_feet_ori": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_feet_ori",
            weight=-2.0,
            params={
                "terrain_type_mask": ["stairs_up"],
                "feet_ori_multiply_scale": 0.1,
            },
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_ankle_roll": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_ankle_roll",
            weight=-2.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "feet_air_time": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:feet_air_time",
            weight=1.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "base_height": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:base_height",
            weight=-25.0,
            # ``desired_base_height`` is only used if the experiment's command
            # doesn't set ``command_base_height_nominal`` (i.e. no 9-D command);
            # value matches :class:`HeadingLocomotionCommand` default (0.75).
            params={
                "desired_base_height": 0.75,
            },
            ma_reward_group="lower_body",
        ),
        "termination": MultiAgentRewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-250.0,
            params={},
            ma_reward_group="shared",
        ),
        "feet_heading_alignment": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:feet_heading_alignment",
            weight=-0.25,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_close_feet_xy": MultiAgentRewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            ma_reward_group="lower_body",
        ),
        "penalty_close_knees_xy": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_close_knees_xy",
            weight=-10.0,
            params={"close_knees_threshold": 0.16},
            ma_reward_group="lower_body",
        ),
        "penalty_hip_pos": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_hip_pos",
            weight=-2.5,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_hip_roll": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_hip_roll",
            weight=-2.5,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_ang_vel_xy_torso": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_ang_vel_xy_torso",
            weight=-1.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_contact": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_contact",
            weight=-4.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_negative_knee_joint": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_negative_knee_joint",
            weight=-1.0,
            params={},
            ma_reward_group="lower_body",
        ),
        "penalty_diff_feet_air_time": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_diff_feet_air_time",
            weight=-2.5,
            params={"terrain_type_mask": ["stairs_up", "stairs_down"]},
            ma_reward_group="lower_body",
        ),
        # Foothold: same three params as RPL reward YAML (e.g. reward_g1_29dof_falcon_multi_terrain.yaml).
        "penalty_foothold": MultiAgentRewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_foothold",
            weight=-10.0,
            params={
                "foothold_epsilon": 0.05,
                "foothold_num_samples_long": 10,
                "foothold_num_samples_wide": 5,
            },
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        "penalty_feet_edge": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_feet_edge",
            weight=0.0,
            params={"terrain_type_mask": ["stepping_stones_columns"]},
            tags=["penalty_curriculum"],
            ma_reward_group="lower_body",
        ),
        # Upper body rewards
        "tracking_upper_body_dofs": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_upper_body_dofs",
            weight=4.0,
            params={"tracking_sigma": 0.1},
            ma_reward_group="upper_body",
        ),
        "penalty_upper_body_action_rate": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_upper_body_action_rate",
            weight=-0.5,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "penalty_upper_body_torques": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_upper_body_torques",
            weight=-1e-5,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "penalty_upper_body_dof_vel": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_upper_body_dof_vel",
            weight=-1.0e-3,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "penalty_upper_body_dof_acc": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_upper_body_dof_acc",
            weight=-2.5e-7,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "limits_upper_body_dof_pos": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:limits_upper_body_dof_pos",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "penalty_ee_lin_acc": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_ee_lin_acc",
            weight=-1e-1,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "penalty_ee_ang_acc": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:penalty_ee_ang_acc",
            weight=-1e-2,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "limits_upper_body_dof_vel": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:limits_upper_body_dof_vel",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
        "limits_upper_body_torque": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:limits_upper_body_torque",
            weight=-0.1,
            params={},
            tags=["penalty_curriculum"],
            ma_reward_group="upper_body",
        ),
    },
)


# FAR-FALCON-Terrains rewards as resolved by ``+exp=falcon_hm_multi_terrain``
# (``reward_g1_29dof_falcon_multi_terrain.yaml`` + the exp's action-rate -0.5 /
# feet-contact-force -0.01 overrides), used by the FALCON-parity RL experts.
# FALCON's terrain-index masks map to Holosoma terrain names as
# 0 flat, 1 curb, 2 stepping_stones_columns, 3 slope_up, 4 slope_down,
# 5 stairs_up, 6 stairs_down, 7 gap; Holosoma's bidirectional ``slopes`` is
# listed wherever FALCON lists 3/4. Names absent from the terrain are skipped.
# Differences from :data:`g1_29dof_falcon_hm_reward`:
#   * every term is in FALCON's ``lower_body`` / ``upper_body`` group (no shared terms:
#     torso tapping / stance, terrain progress and termination go to the lower body);
#   * no ``penalty_hip_roll`` / ``stand_still``; adds ``tracking_walk_base_height``;
#   * lower-body torque / dof-vel / dof-acc penalties over all 15 lower dofs, dof-acc
#     differenced over one control step (upper body too);
#   * walk-only orientation penalty, global-mean terrain progress, knee threshold 0.02;
#   * stones: no z-velocity penalty, hip-pos penalty x0.25; stairs: base-height penalty
#     reduced by the local height spread;
#   * close feet = world xy distance < 0.16, close knees < 0.18; foot-force cap
#     relaxed by the hand-force z; air time / feet orientation / feet edge use filtered contact;
#   * foothold reads the raw heightfield (FALCON ``_get_terrain_heights``), not a mesh ray-cast;
#   * FALCON's per-terrain masks for heading, stumble, feet orientation, feet contact
#     forces, diff air time, feet edge and foothold.
_SLOPES = ["slopes", "slope_up", "slope_down"]
_FALCON_LIN_VEL_Z_MASK = ["flat", "curb", *_SLOPES, "stairs_up", "stairs_down"]  # [0, 1, 3, 4, 5, 6]
_FALCON_HIP_POS_MASK = ["flat", "curb", *_SLOPES, "stairs_up", "stairs_down"]  # [0, 1, 3, 4, 5, 6]
_FALCON_CONTACT_FORCES_MASK = ["flat", "curb", *_SLOPES, "stairs_up", "stairs_down"]  # [0, 1, 3, 4, 5, 6]
_FALCON_STUMBLE_MASK = ["flat", "curb", "stepping_stones_columns", "stairs_up", "stairs_down", "gap"]  # [0,1,2,5,6,7]
_FALCON_FEET_ORI_MASK = ["flat", "curb", "stairs_up", "gap"]  # [0, 1, 5, 7]
_FALCON_DIFF_AIR_TIME_MASK = ["flat", "curb", "stairs_up", "stairs_down"]  # [0, 1, 5, 6]
_FALCON_FEET_EDGE_MASK = ["curb", "stepping_stones_columns"]  # [1, 2]
_FALCON_FOOTHOLD_MASK = ["curb", "stepping_stones_columns", *_SLOPES, "stairs_up", "stairs_down", "gap"]  # [1..7]
_FALCON_REDUCE_BASE_HEIGHT_MASK = ["stairs_up", "stairs_down"]  # [5, 6]


def _falcon_term(name: str, **overrides) -> MultiAgentRewardTermCfg:
    base = g1_29dof_falcon_hm_reward.terms[name]
    params = {**base.params, **overrides.pop("params", {})}
    return replace(base, params=params, **overrides)


_falcon_terms = {
    name: replace(term, ma_reward_group="lower_body") if term.ma_reward_group == "shared" else term
    for name, term in g1_29dof_falcon_hm_reward.terms.items()
    if name != "penalty_hip_roll"
}
_falcon_terms.update(
    {
        "tracking_walk_base_height": MultiAgentRewardTermCfg(
            func="rpl_training.terms.reward:tracking_walk_base_height",
            weight=1.0,
            params={"tracking_sigma": 0.05, "terrain_type_mask": ["flat"]},
            ma_reward_group="lower_body",
        ),
        "terrain_progress": _falcon_term(
            "terrain_progress", params={"global_mean": True}, ma_reward_group="lower_body"
        ),
        "termination": _falcon_term("termination", ma_reward_group="lower_body"),
        "tracking_torso_orientation_tapping": _falcon_term(
            "tracking_torso_orientation_tapping", ma_reward_group="lower_body"
        ),
        "tracking_torso_orientation_stance": _falcon_term(
            "tracking_torso_orientation_stance", ma_reward_group="lower_body"
        ),
        "penalty_lin_vel_z": _falcon_term("penalty_lin_vel_z", params={"terrain_type_mask": _FALCON_LIN_VEL_Z_MASK}),
        "penalty_orientation": _falcon_term("penalty_orientation", params={"walk_only": True}),
        # The experts' final stages all set ``penalty_lower_body_torques=-4e-5`` on the CLI.
        "penalty_lower_body_torques": _falcon_term(
            "penalty_lower_body_torques", weight=-4e-5, params={"use_all_lower_dofs": True}
        ),
        "penalty_lower_body_dof_vel": _falcon_term("penalty_lower_body_dof_vel", params={"use_all_lower_dofs": True}),
        "penalty_lower_body_dof_acc": _falcon_term(
            "penalty_lower_body_dof_acc", params={"use_all_lower_dofs": True, "falcon_control_step": True}
        ),
        "penalty_upper_body_dof_acc": _falcon_term("penalty_upper_body_dof_acc", params={"falcon_control_step": True}),
        "penalty_feet_contact_forces": _falcon_term(
            "penalty_feet_contact_forces",
            params={"terrain_type_mask": _FALCON_CONTACT_FORCES_MASK, "add_ee_force_z": True},
        ),
        "penalty_stumble": _falcon_term("penalty_stumble", params={"terrain_type_mask": _FALCON_STUMBLE_MASK}),
        "penalty_feet_ori": _falcon_term(
            "penalty_feet_ori", params={"terrain_type_mask": _FALCON_FEET_ORI_MASK, "use_filtered_contact": True}
        ),
        "feet_air_time": _falcon_term("feet_air_time", params={"use_filtered_contact": True}),
        "base_height": _falcon_term(
            "base_height",
            params={
                "reduce_terrain_type_mask": _FALCON_REDUCE_BASE_HEIGHT_MASK,
                "reduction_factor": 0.75,
                "max_delta_height_range": 0.6,
            },
        ),
        "penalty_close_feet_xy": _falcon_term(
            "penalty_close_feet_xy", params={"close_feet_threshold": 0.16, "mode": "norm"}
        ),
        "penalty_close_knees_xy": _falcon_term("penalty_close_knees_xy", params={"close_knees_threshold": 0.18}),
        "penalty_hip_pos": _falcon_term(
            "penalty_hip_pos", params={"terrain_type_mask": _FALCON_HIP_POS_MASK, "reduction_scale": 0.25}
        ),
        "penalty_negative_knee_joint": _falcon_term(
            "penalty_negative_knee_joint", params={"knee_joint_min_threshold": 0.02}
        ),
        "penalty_diff_feet_air_time": _falcon_term(
            "penalty_diff_feet_air_time", params={"terrain_type_mask": _FALCON_DIFF_AIR_TIME_MASK}
        ),
        "penalty_feet_edge": _falcon_term(
            "penalty_feet_edge", params={"terrain_type_mask": _FALCON_FEET_EDGE_MASK, "use_filtered_contact": True}
        ),
        "penalty_foothold": _falcon_term(
            "penalty_foothold",
            func="rpl_training.terms.reward:penalty_foothold",
            params={"terrain_type_mask": _FALCON_FOOTHOLD_MASK, "falcon_heightfield": True},
        ),
    }
)

g1_29dof_rpl_reward_expert = MultiAgentRewardManagerCfg(
    only_positive_rewards=g1_29dof_falcon_hm_reward.only_positive_rewards,
    multi_agent_body_keys=g1_29dof_falcon_hm_reward.multi_agent_body_keys,
    terms=_falcon_terms,
)


def _falcon_expert_reward(
    weights: dict[str, float], params: dict[str, dict] | None = None
) -> MultiAgentRewardManagerCfg:
    terms = dict(g1_29dof_rpl_reward_expert.terms)
    for name, weight in weights.items():
        terms[name] = replace(terms[name], weight=weight)
    for name, extra in (params or {}).items():
        terms[name] = replace(terms[name], params={**terms[name].params, **extra})
    return replace(g1_29dof_rpl_reward_expert, terms=terms)


# Per-terrain reward overrides of FALCON's final expert stages (wandb run args):
# stones far-wandb/falcon-29-terrains-experts-stones/lrlo8qpt.
g1_29dof_rpl_reward_expert_stones = _falcon_expert_reward(
    {
        "penalty_feet_ori": -0.25,
        "penalty_contact_no_vel": -1.0,
        "penalty_feet_edge": -5.0,
        "penalty_feet_contact_forces": -0.0025,
        "tracking_lin_vel_x": 5.0,
    },
    # ``rewards.penalty_feet_ori_mask=[0,1,2,5,7]``: both feet penalized on stones too.
    params={"penalty_feet_ori": {"terrain_type_mask": [*_FALCON_FEET_ORI_MASK, "stepping_stones_columns"]}},
)
# stairs far-wandb/falcon-29-terrains-experts/h5d52hm0.
g1_29dof_rpl_reward_expert_stairs = _falcon_expert_reward(
    {
        "penalty_feet_ori": -1.0,
        "penalty_stumble": -50.0,
        "penalty_contact_no_vel": -1.0,
        "tracking_lin_vel_x": 6.0,
    }
)
# slopes far-wandb/falcon-29-terrains-experts/b04ael3p.
g1_29dof_rpl_reward_expert_slopes = _falcon_expert_reward(
    {
        "penalty_contact_no_vel": -1.0,
        "tracking_lin_vel_x": 6.0,
    }
)

__all__ = [
    "g1_29dof_falcon_hm_reward",
    "g1_29dof_rpl_reward_expert",
    "g1_29dof_rpl_reward_expert_slopes",
    "g1_29dof_rpl_reward_expert_stairs",
    "g1_29dof_rpl_reward_expert_stones",
]
