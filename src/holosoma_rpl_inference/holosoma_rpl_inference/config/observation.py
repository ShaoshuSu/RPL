"""Observation config for loco-manip stand-height-waist policy."""

from holosoma_inference.config.config_types.observation import ObservationConfig

loco_manip_stand_height_waist = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "command_stand",
            "command_base_height",
            "command_waist_dofs",
            "ref_upper_dof_pos",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
        "command_lin_vel_obs": [
            "command_lin_vel",
        ],
    },
    obs_dims={
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "command_stand": 1,
        "command_base_height": 1,
        "command_waist_dofs": 3,
        "ref_upper_dof_pos": 14,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 2592,  # 2 cameras * 48 width * 27 height
    },
    obs_scales={
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "command_stand": 1.0,
        "command_base_height": 2.0,
        "command_waist_dofs": 1.0,
        "ref_upper_dof_pos": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "history": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs": 5,
        "perception_obs": 2,
        "command_lin_vel_obs": 1,
    },
)
