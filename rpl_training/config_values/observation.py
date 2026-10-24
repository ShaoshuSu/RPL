"""Locomotion observation presets for the G1 robot."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg


def _group(**kwargs) -> ObsGroupCfg:
    """Observation group with FALCON's time-major history layout (``[t-N: all terms, ..., t]``)."""
    return ObsGroupCfg(history_time_major=True, **kwargs)


g1_29dof_falcon_hm_history_wolinvel = ObservationManagerCfg(
    groups={
        "actor_state_obs": _group(
            concatenate=True,
            enable_noise=True,
            history_length=5,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_stand": ObsTermCfg(
                    func="rpl_training.terms.observation:command_stand",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_waist_dofs": ObsTermCfg(
                    func="rpl_training.terms.observation:command_waist_dofs",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_base_height": ObsTermCfg(
                    func="rpl_training.terms.observation:command_base_height",
                    scale=2.0,
                    noise=0.0,
                ),
                "ref_upper_dof_pos": ObsTermCfg(
                    func="rpl_training.terms.observation:ref_upper_dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.1,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "perception_obs": _group(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                "height_measurements": ObsTermCfg(
                    func="rpl_training.terms.observation:height_measurements",
                    scale=1.0,
                    noise=0.0,
                    # Match FAR-FALCON's ``terrain.{min,max}_measured_height``.
                    params={"min_height": -0.5, "max_height": 0.5},
                ),
            },
        ),
        "perception_obs_wo_noise": _group(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "height_measurements": ObsTermCfg(
                    func="rpl_training.terms.observation:height_measurements",
                    scale=1.0,
                    noise=0.0,
                    params={"min_height": -0.5, "max_height": 0.5},
                ),
            },
        ),
        "critic_state_obs": _group(
            concatenate=True,
            enable_noise=False,
            history_length=5,
            terms={
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=2.0,
                    noise=0.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_stand": ObsTermCfg(
                    func="rpl_training.terms.observation:command_stand",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_waist_dofs": ObsTermCfg(
                    func="rpl_training.terms.observation:command_waist_dofs",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_base_height": ObsTermCfg(
                    func="rpl_training.terms.observation:command_base_height",
                    scale=2.0,
                    noise=0.0,
                ),
                "ref_upper_dof_pos": ObsTermCfg(
                    func="rpl_training.terms.observation:ref_upper_dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "ee_apply_force": ObsTermCfg(
                    func="rpl_training.terms.observation:ee_apply_force",
                    scale=0.1,
                    noise=0.0,
                ),
            },
        ),
    }
)


# Depth-distillation variant for ``dagger_ma_moe_cnn_depth_vtr``. Mirrors
# FAR-FALCON's ``g1_29dof_obs_falcon_history_wolinvel_hm_depth_moe_bc.yaml``:
#   * ``actor_state_obs`` history bumped to 16 (FALCON parity).
#   * ``critic_state_obs`` history bumped to 16.
#   * separate ``actor_teacher_state_obs`` group (history=5) so the MoE teacher
#     can run on a shorter window than the student.
#   * ``perception_obs`` swapped from height map to ``cam_depth`` (student input).
#   * ``perception_obs_wo_noise`` keeps ``height_measurements`` (teacher input).
#   * standalone ``command_lin_vel_obs`` group consumed by the VTR vision-mask.
#   * ``terrain_idx_obs`` group consumed by the MoE routing layer.
_base_groups = g1_29dof_falcon_hm_history_wolinvel.groups
_actor_state_terms = _base_groups["actor_state_obs"].terms

g1_29dof_falcon_hm_depth_distill = ObservationManagerCfg(
    groups={
        "actor_state_obs": replace(_base_groups["actor_state_obs"], history_length=16),
        "actor_teacher_state_obs": _group(
            concatenate=True,
            enable_noise=False,
            history_length=5,
            terms=_actor_state_terms,
        ),
        "perception_obs": _group(
            concatenate=True,
            enable_noise=True,
            history_length=2,
            terms={
                "cam_depth": ObsTermCfg(
                    func="rpl_training.terms.observation:cam_depth",
                    scale=1.0,
                    noise=0.0,
                    # Pulls from ``warp_dual_depth_cameras`` (single ``warp_depth`` term).
                    params={"camera_term": "warp_depth"},
                ),
            },
        ),
        "perception_obs_wo_noise": _base_groups["perception_obs_wo_noise"],
        "command_lin_vel_obs": _group(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "terrain_idx_obs": _group(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "terrain_type_idx": ObsTermCfg(
                    func="rpl_training.terms.observation:terrain_type_idx",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_state_obs": replace(_base_groups["critic_state_obs"], history_length=16),
    }
)


# Exact mirror of FAR-FALCON-Terrains' resolved ``obs`` for ``+exp=falcon_hm_dagger_moe``
# (``g1_29dof_obs_falcon_history_wolinvel_hm_depth_moe_bc.yaml``), used with
# ``dagger_ma_moe_cnn_depth_vtr_falcon`` so FALCON MoE teacher checkpoints load as-is.
# Differences from :data:`g1_29dof_falcon_hm_depth_distill`:
#   * student ``actor_state_obs`` history 5 (not 16) -> 575-dim upper-body / VTR state input;
#   * ``perception_obs`` (2-frame depth stack) only pushes a new frame every 5 env steps
#     (FALCON ``obs_intervals.perception_obs: 5``);
#   * FALCON never clears observation history on env reset, only the interval counters;
#   * teacher height map gets noise 0.1 * noise_level 0.1 = 0.01;
#   * critic has no history and noise_level 0.1 (dof_pos 0.001, dof_vel 0.01);
#   * ``command_base_height`` is absolute (FALCON semantics), see ``_falcon_abs_height``.
# Holosoma applies the absolute noise before the scale, like FALCON's
# ``(obs + U(-1, 1) * noise_scale * noise_level) * obs_scale``.
_falcon_noisy_hm_terms = {
    "height_measurements": replace(_base_groups["perception_obs_wo_noise"].terms["height_measurements"], noise=0.01),
}


# FALCON's ``commands[:, 8]`` is the absolute base height (``desired_base_height`` = 0.75 while
# walking); Holosoma stores a delta from the nominal, so add it back for FALCON-trained teachers.
def _falcon_abs_height(terms: dict[str, ObsTermCfg]) -> dict[str, ObsTermCfg]:
    terms = dict(terms)
    terms["command_base_height"] = replace(terms["command_base_height"], params={"absolute": True})
    return terms


_falcon_actor_terms = _falcon_abs_height(_actor_state_terms)
_falcon_critic_terms = _falcon_abs_height(_base_groups["critic_state_obs"].terms)
_falcon_critic_terms["dof_pos"] = replace(_falcon_critic_terms["dof_pos"], noise=0.001)
_falcon_critic_terms["dof_vel"] = replace(_falcon_critic_terms["dof_vel"], noise=0.01)

g1_29dof_rpl_depth_distill = replace(
    g1_29dof_falcon_hm_depth_distill,
    groups={
        **g1_29dof_falcon_hm_depth_distill.groups,
        "actor_state_obs": replace(
            _base_groups["actor_state_obs"],
            history_length=5,
            reset_history_on_env_reset=False,
            terms=_falcon_actor_terms,
        ),
        "actor_teacher_state_obs": replace(
            g1_29dof_falcon_hm_depth_distill.groups["actor_teacher_state_obs"],
            reset_history_on_env_reset=False,
            terms=_falcon_actor_terms,
        ),
        "perception_obs": replace(
            g1_29dof_falcon_hm_depth_distill.groups["perception_obs"],
            interval=5,
            reset_history_on_env_reset=False,
        ),
        "perception_obs_wo_noise": _group(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=_falcon_noisy_hm_terms,
        ),
        "critic_state_obs": replace(
            _base_groups["critic_state_obs"],
            enable_noise=True,
            history_length=1,
            terms=_falcon_critic_terms,
        ),
    },
)


# Height-map student: g1_29dof_rpl_depth_distill with the depth stack replaced by the teachers' 16 x 10 height map
# carrying LiDAR elevation-map errors (rpl_training.terms.lidar_height_map; parameters in the randomization term
# "lidar_height_map_noise"). One frame, every policy step: the map's lower update rate and latency are modelled
# inside the term. Teacher, critic and state groups are unchanged.
g1_29dof_rpl_heightmap_distill = replace(
    g1_29dof_rpl_depth_distill,
    groups={
        **g1_29dof_rpl_depth_distill.groups,
        "perception_obs": _group(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            reset_history_on_env_reset=False,
            terms={
                "lidar_height_map": ObsTermCfg(
                    func="rpl_training.terms.lidar_height_map:lidar_height_measurements",
                    scale=1.0,
                    noise=0.0,
                    params={"noise_term": "lidar_height_map_noise", "min_height": -0.5, "max_height": 0.5},
                ),
            },
        ),
    },
)

# FALCON's RL experts (final fine-tune stages of falcon-29-terrains-experts{,-stones}) add
# IMU noise ``obs.noise_scales.base_ang_vel=0.2 projected_gravity=0.05`` on the CLI; the
# critic sees it at noise_level 0.1.
def _with_imu_noise(terms: dict[str, ObsTermCfg], level: float) -> dict[str, ObsTermCfg]:
    terms = dict(terms)
    terms["base_ang_vel"] = replace(terms["base_ang_vel"], noise=0.2 * level)
    terms["projected_gravity"] = replace(terms["projected_gravity"], noise=0.05 * level)
    return terms


# FAR-FALCON-Terrains' resolved ``obs`` for the RL experts (``+exp=falcon_hm_multi_terrain``,
# ``g1_29dof_obs_falcon_history_wolinvel_hm.yaml`` with ``add_noise: true`` plus the experts'
# IMU-noise overrides). Differences from :data:`g1_29dof_falcon_hm_history_wolinvel`:
#   * ``command_base_height`` is absolute (FALCON semantics), see ``_falcon_abs_height``;
#   * FALCON never clears observation history on env reset;
#   * IMU noise base_ang_vel 0.2 / projected_gravity 0.05 (critic x0.1: 0.02 / 0.005);
#   * critic noise_level 0.1 (dof_pos 0.001, dof_vel 0.01). The heightmaps stay noise-free
#     (the stairs expert); :data:`g1_29dof_rpl_history_wolinvel_hm_noise` adds
#     the stones / slopes experts' heightmap noise.
g1_29dof_rpl_history_wolinvel = replace(
    g1_29dof_falcon_hm_history_wolinvel,
    groups={
        "actor_state_obs": replace(
            _base_groups["actor_state_obs"],
            reset_history_on_env_reset=False,
            terms=_with_imu_noise(_falcon_actor_terms, 1.0),
        ),
        "perception_obs": replace(_base_groups["perception_obs"], reset_history_on_env_reset=False),
        "perception_obs_wo_noise": replace(_base_groups["perception_obs_wo_noise"], reset_history_on_env_reset=False),
        "critic_state_obs": replace(
            _base_groups["critic_state_obs"],
            enable_noise=True,
            reset_history_on_env_reset=False,
            terms=_with_imu_noise(_falcon_critic_terms, 0.1),
        ),
    },
)


def _with_hm_noise(groups: dict, noise: float) -> dict:
    """FALCON ``obs.noise_scales.height_measurements``: actor heightmap at noise_level 1,
    the critic's ``perception_obs_wo_noise`` at noise_level 0.1 (FALCON noises every group)."""
    groups = dict(groups)
    for name, level in (("perception_obs", 1.0), ("perception_obs_wo_noise", 0.1)):
        terms = {"height_measurements": replace(groups[name].terms["height_measurements"], noise=noise * level)}
        groups[name] = replace(groups[name], enable_noise=True, terms=terms)
    return groups


# Stones / slopes experts: ``obs.noise_scales.height_measurements=0.1``.
g1_29dof_rpl_history_wolinvel_hm_noise = replace(
    g1_29dof_rpl_history_wolinvel,
    groups=_with_hm_noise(g1_29dof_rpl_history_wolinvel.groups, 0.1),
)


__all__ = [
    "g1_29dof_rpl_heightmap_distill",
    "g1_29dof_falcon_hm_depth_distill",
    "g1_29dof_rpl_depth_distill",
    "g1_29dof_falcon_hm_history_wolinvel",
    "g1_29dof_rpl_history_wolinvel",
    "g1_29dof_rpl_history_wolinvel_hm_noise",
]
