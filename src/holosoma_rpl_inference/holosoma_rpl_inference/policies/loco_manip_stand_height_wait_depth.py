"""Loco-manip policy with depth cameras, ported from FAR-Holosoma.

Hardcodes camera parameters and obs_intervals to avoid modifying the
holosoma submodule's config types (CameraConfig, ObservationConfig).
"""

from collections import deque
from multiprocessing import shared_memory

import numpy as np

from holosoma_inference.policies.locomotion import LocomotionPolicy

# Camera layout — matches the ImageServer in holosoma_rpl.
NUM_CAMERAS = 2
CAMERA_WIDTH = 48
CAMERA_HEIGHT = 27

# Observation intervals in rl_policy_rate frames.
# actor_obs and command_lin_vel_obs update every step;
# perception_obs (depth) updates every 5 steps.
OBS_INTERVALS = {
    "actor_obs": 1,
    "perception_obs": 5,
    "command_lin_vel_obs": 1,
}


class LocoManipStandHeightWaitDepthPolicy(LocomotionPolicy):
    """Locomotion-manipulation policy with base height control, waist commands,
    and depth camera integration.

    Inherits from LocomotionPolicy to maintain compatibility with holosoma
    while adding upper body control, per-group obs history, and depth camera
    integration via shared memory.
    """

    def __init__(self, config):
        super().__init__(config)

        # Attributes that live on ObservationConfig in FAR-Holosoma but are
        # absent from the RPL submodule.  Set them here instead.
        self.obs_intervals = OBS_INTERVALS
        self.obs_interval_counters = 0

        self._init_upper_body_pos()
        self._init_depth_shm()
        self._init_fixed_forward_speed_command()

    # ------------------------------------------------------------------
    # History (overrides base to use per-group buffers with intervals)
    # ------------------------------------------------------------------

    def _initialize_history_state(self):
        super()._initialize_history_state()
        self.obs_history_buffers_per_group = {}
        for group in self.obs_terms_sorted:
            self.obs_history_buffers_per_group[group] = deque(
                maxlen=self.history_length_dict[group]
            )

    # ------------------------------------------------------------------
    # Upper body
    # ------------------------------------------------------------------

    def _init_upper_body_pos(self):
        robot_config = self.config.robot
        self.upper_dof_indices = [
            robot_config.dof_names.index(dof)
            for dof in robot_config.dof_names_upper_body
        ]
        self.ref_upper_dof_pos = np.array(
            [robot_config.default_dof_angles[i] for i in self.upper_dof_indices]
        ).reshape(1, -1)

    # ------------------------------------------------------------------
    # Depth camera (shared memory)
    # ------------------------------------------------------------------

    def _init_depth_shm(self):
        channels = 1  # depth is single-channel
        expected_shape = [NUM_CAMERAS, channels, CAMERA_HEIGHT, CAMERA_WIDTH]
        print(f"expected_shape: {expected_shape}")

        self.depth_img_shm = shared_memory.SharedMemory(name="depth_img_shm")
        self.depth_img_array = np.ndarray(
            expected_shape, dtype=np.float32, buffer=self.depth_img_shm.buf
        )
        print(
            f"[Policy] Depth image client initialized with shared memory: "
            f"{self.depth_img_shm.name}"
        )

    # ------------------------------------------------------------------
    # Fixed forward speed
    # ------------------------------------------------------------------

    def _init_fixed_forward_speed_command(self):
        self.fixed_forward_speed = 0.5
        self.fixed_forward_speed_enabled = False

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)

        current_obs_buffer_dict["command_base_height"] = self.base_height_command
        current_obs_buffer_dict["command_waist_dofs"] = self.waist_dofs_command
        current_obs_buffer_dict["ref_upper_dof_pos"] = self.ref_upper_dof_pos

        depth_images = self.depth_img_array.copy()  # (2, 1, 27, 48)
        current_obs_buffer_dict["cam_depth"] = depth_images.reshape(1, -1)
        current_obs_buffer_dict["command_lin_vel_obs"] = self.lin_vel_command

        return current_obs_buffer_dict

    def _update_obs_history(self, current_obs_dict):
        for group, term_dict in current_obs_dict.items():
            obs_list_per_group: list[np.ndarray] = []
            for term in self.obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                obs_list_per_group.append(obs.copy())
            obs_per_group = np.concatenate(obs_list_per_group, axis=1)

            if self.obs_interval_counters % self.obs_intervals[group] == 0:
                self.obs_history_buffers_per_group[group].append(obs_per_group)

            history_len = self.history_length_dict.get(group, 1)
            history = list(self.obs_history_buffers_per_group[group])
            if len(history) < history_len:
                missing = history_len - len(history)
                history = [history[0]] * missing + history
            self.obs_buf_dict[group] = np.stack(history, axis=1).reshape(1, -1)

        self.obs_interval_counters += 1

    def prepare_obs_for_rl(self, robot_state_data):
        self._prepare_group_observations(robot_state_data)

        actor_obs_lower_body = np.concatenate(
            [
                self.obs_buf_dict["actor_obs"],
                self.obs_buf_dict["perception_obs"],
                self.obs_buf_dict["command_lin_vel_obs"],
            ],
            axis=1,
        )
        actor_obs_upper_body = self.obs_buf_dict["actor_obs"]
        return {
            "actor_obs_lower_body": actor_obs_lower_body.astype(np.float32),
            "actor_obs_upper_body": actor_obs_upper_body.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Joystick overrides
    # ------------------------------------------------------------------

    def process_joystick_input(self):
        super().process_joystick_input()
        if self.fixed_forward_speed_enabled:
            self.lin_vel_command[0, 0] = self.fixed_forward_speed

    def handle_joystick_button(self, cur_key: str):
        super().handle_joystick_button(cur_key)
        if cur_key == "L2":
            self.fixed_forward_speed_enabled = True
            self.fixed_forward_speed = 0.5
        elif cur_key == "R2":
            self.fixed_forward_speed_enabled = True
            self.fixed_forward_speed = -0.6
        elif cur_key == "R2+up":
            self.fixed_forward_speed += 0.1
        elif cur_key == "R2+down":
            self.fixed_forward_speed -= 0.1

    def _handle_stand_command(self):
        super()._handle_stand_command()
        if self.stand_command[0, 0] == 0:
            self.fixed_forward_speed_enabled = False
            self.fixed_forward_speed = 0.0
