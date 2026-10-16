"""Loco-manipulation environment with EE force application + force-scale curriculum.

Ports the activated subset of FAR-FALCON-Terrains'
``LeggedRobotDecoupledLocomotionStanceHeightWBCForce`` pipeline. Only the
``force_estimation_type=0`` path is kept (jacobian-based per-axis effort
budget), with no LPF, no sphere-sampling of force-application points, no
``push_end_effectors`` impulses, and no ``randomize_force_pos`` /
``randomize_force_xyz_scale``.
"""

from __future__ import annotations

from typing import Any

from holosoma.utils.rotations import quat_rotate, quat_rotate_inverse
from holosoma.utils.safe_torch_import import torch
from loguru import logger

from rpl_training.envs.loco_manip_manager import LeggedRobotLocoManipManager

# Defaults mirror FAR-FALCON-Terrains ``humanoidverse/config/exp/falcon_g1_29dof.yaml`` activated values.
_DEFAULT_FORCE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "apply_force_x_range": (-100.0, 100.0),
    "apply_force_y_range": (-100.0, 100.0),
    "apply_force_z_range": (-60.0, 5.0),
    "zero_force_prob": (1.01, 1.01, 0.25),
    "force_xyz_scale_ratio": (0.01, 0.01, 1.0),
    "randomize_force_duration": (150, 250),
    "random_force_prob": 0.0,
    "only_apply_z_force_when_walking": True,
    "only_apply_resistance_when_walking": True,
    "dof_effort_limit_scale": 0.8,
    "left_hand_link": "left_rubber_hand",
    "right_hand_link": "right_rubber_hand",
    # Force application point. ``"hand"`` (legacy): the hand link origin.
    # ``"hand_to_extend"`` (FALCON ``extend_config``): a point on the segment from
    # the hand link to ``parent + R_parent @ extend_offset``, at a per-env ratio
    # drawn once from ``apply_force_pos_ratio_range``.
    "apply_force_pos_mode": "hand",
    "left_extend_parent_link": "left_wrist_yaw_link",
    "right_extend_parent_link": "right_wrist_yaw_link",
    "extend_offset": (0.18, -0.025, 0.0),
    "apply_force_pos_ratio_range": (0.0, 1.0),
    # When True, force settings are resampled together with the locomotion
    # commands (every ``locomotion_command_resampling_time`` and *after* the
    # command reset), as in FALCON. Legacy: resampled on reset only, before the
    # commands are reset.
    "resample_force_with_commands": False,
}

# Key in ``tyro_config.command.params`` holding per-experiment overrides of the
# force config above (merged over ``_DEFAULT_FORCE_CONFIG``).
FORCE_CONFIG_PARAM_KEY = "ee_force_config"

_LEFT_HAND_LINK_FALLBACKS = (
    "left_rubber_hand",
    "left_hand_link",
    "left_wrist_yaw_link",
)
_RIGHT_HAND_LINK_FALLBACKS = (
    "right_rubber_hand",
    "right_hand_link",
    "right_wrist_yaw_link",
)


class LeggedRobotLocoManipForceManager(LeggedRobotLocoManipManager):
    """LocoManip env that applies external forces to both end-effectors.

    The applied force magnitude is gated by ``env.apply_force_scale`` (a
    per-env tensor). The companion ``ForceScaleCurriculum`` curriculum term
    drives that scale up / down with episode length, replicating
    FALCON-Terrains' ``force_scale_curriculum`` behaviour.
    """

    force_config: dict[str, Any] = _DEFAULT_FORCE_CONFIG

    def __init__(self, tyro_config, *, device):
        # Resolve the force config before ``super().__init__`` because
        # ``_setup_robot_body_indices`` / ``_init_buffers`` read it.
        self.force_config = self._resolve_force_config(tyro_config)
        super().__init__(tyro_config, device=device)

    @staticmethod
    def _resolve_force_config(tyro_config) -> dict[str, Any]:
        command_cfg = getattr(tyro_config, "command", None)
        params = getattr(command_cfg, "params", None) or {}
        overrides = params.get(FORCE_CONFIG_PARAM_KEY) or {}
        unknown = set(overrides) - set(_DEFAULT_FORCE_CONFIG)
        if unknown:
            raise ValueError(f"Unknown {FORCE_CONFIG_PARAM_KEY} keys: {sorted(unknown)}")
        return {**_DEFAULT_FORCE_CONFIG, **dict(overrides)}

    def _init_buffers(self):
        super()._init_buffers()
        self._init_force_settings()

    def _resolve_hand_link_index(self, side: str, configured_name: str, fallbacks: tuple[str, ...]) -> tuple[str, int]:
        """Resolve an EE body name across robot variants (G1 rubber hand vs T1 hand link)."""
        candidates: list[str] = []
        for name in (configured_name, *fallbacks):
            if name and name not in candidates:
                candidates.append(name)

        for name in candidates:
            try:
                index = self.simulator.find_rigid_body_indice(name)
            except ValueError:
                continue
            if index is None or int(index) < 0:
                continue
            return name, int(index)

        raise RuntimeError(
            f"Could not resolve {side} hand link on {type(self.simulator).__name__}. Tried: {candidates}"
        )

    def _setup_robot_body_indices(self):
        super()._setup_robot_body_indices()
        rc = self.robot_config
        cfg = self.force_config
        left_name, self.left_hand_link_index = self._resolve_hand_link_index(
            "left",
            cfg.get("left_hand_link", "left_rubber_hand"),
            _LEFT_HAND_LINK_FALLBACKS,
        )
        right_name, self.right_hand_link_index = self._resolve_hand_link_index(
            "right",
            cfg.get("right_hand_link", "right_rubber_hand"),
            _RIGHT_HAND_LINK_FALLBACKS,
        )
        if rc.left_arm_dof_names is None or rc.right_arm_dof_names is None:
            raise RuntimeError(
                "LeggedRobotLocoManipForceManager requires robot_config.left_arm_dof_names and right_arm_dof_names."
            )
        self.left_arm_dof_indices = torch.tensor(
            [self.dof_names.index(n) for n in rc.left_arm_dof_names], device=self.device, dtype=torch.long
        )
        self.right_arm_dof_indices = torch.tensor(
            [self.dof_names.index(n) for n in rc.right_arm_dof_names], device=self.device, dtype=torch.long
        )
        logger.info(
            f"Force EE: left={left_name}({self.left_hand_link_index}), right={right_name}({self.right_hand_link_index})"
        )
        if cfg.get("apply_force_pos_mode", "hand") == "hand_to_extend":
            self.left_extend_parent_index = int(self.simulator.find_rigid_body_indice(cfg["left_extend_parent_link"]))
            self.right_extend_parent_index = int(self.simulator.find_rigid_body_indice(cfg["right_extend_parent_link"]))
            if min(self.left_extend_parent_index, self.right_extend_parent_index) < 0:
                raise RuntimeError("Could not resolve the force extend parent links on the simulator.")

    def _init_force_settings(self):
        cfg = self.force_config
        device = self.device
        n = self.num_envs

        # apply_force_scale is owned by the ForceScaleCurriculum term; create a
        # default here so the env still works without the curriculum.
        if not hasattr(self, "apply_force_scale"):
            self.apply_force_scale = torch.ones(n, 1, dtype=torch.float, device=device)

        effort_limit_scale = float(cfg.get("dof_effort_limit_scale", 1.0))
        self.joint_effort_limit = (
            torch.as_tensor(self.robot_config.dof_effort_limit_list, device=device, dtype=torch.float)
            * effort_limit_scale
        )

        # Per-env force-application buffers and exposed observation tensors.
        self.left_ee_apply_force = torch.zeros(n, 3, device=device, dtype=torch.float)
        self.right_ee_apply_force = torch.zeros(n, 3, device=device, dtype=torch.float)
        self.apply_force_tensor = torch.zeros(n, self.num_bodies, 3, device=device, dtype=torch.float)
        self.apply_force_pos_tensor = torch.zeros(n, self.num_bodies, 3, device=device, dtype=torch.float)
        pos_mode = cfg.get("apply_force_pos_mode", "hand")
        if pos_mode not in ("hand", "hand_to_extend"):
            raise ValueError(f"Unknown apply_force_pos_mode: {pos_mode!r}")
        self.apply_force_pos_mode = pos_mode
        ratio_lo, ratio_hi = cfg.get("apply_force_pos_ratio_range", (0.0, 1.0))
        # Sampled once, as in FALCON with ``randomize_force_pos=False``.
        self.left_apply_force_pos_ratio = ratio_lo + (ratio_hi - ratio_lo) * torch.rand(n, 1, device=device)
        self.right_apply_force_pos_ratio = ratio_lo + (ratio_hi - ratio_lo) * torch.rand(n, 1, device=device)
        self.extend_offset = torch.as_tensor(
            cfg.get("extend_offset", (0.0, 0.0, 0.0)), device=device, dtype=torch.float
        )

        self.only_apply_z_force_when_walking = bool(cfg.get("only_apply_z_force_when_walking", True))
        self.only_apply_resistance_when_walking = bool(cfg.get("only_apply_resistance_when_walking", True))

        # Per-axis force ranges.
        x_range = cfg["apply_force_x_range"]
        y_range = cfg["apply_force_y_range"]
        z_range = cfg["apply_force_z_range"]
        self.force_range_low = torch.zeros(n, 3, device=device, dtype=torch.float)
        self.force_range_high = torch.zeros(n, 3, device=device, dtype=torch.float)
        self.force_range_low[:, 0] = float(x_range[0])
        self.force_range_high[:, 0] = float(x_range[1])
        self.force_range_low[:, 1] = float(y_range[0])
        self.force_range_high[:, 1] = float(y_range[1])
        self.force_range_low[:, 2] = float(z_range[0])
        self.force_range_high[:, 2] = float(z_range[1])

        # Zero-force probability per axis (separately for left / right).
        zero_prob = cfg.get("zero_force_prob", (0.2, 0.2, 0.2))
        if isinstance(zero_prob, (int, float)):
            zero_prob = (float(zero_prob),) * 3
        zero_prob_t = torch.as_tensor(zero_prob, device=device, dtype=torch.float)
        self.left_zero_force_prob = zero_prob_t
        self.right_zero_force_prob = zero_prob_t
        self.left_zero_force = (torch.rand(n, 3, device=device) < zero_prob_t).float()
        self.right_zero_force = (torch.rand(n, 3, device=device) < zero_prob_t).float()

        # Per-env axis weighting via a Dirichlet prior on (x, y, z).
        ratio = cfg.get("force_xyz_scale_ratio", (1.0, 1.0, 1.0))
        self.force_xyz_scale_ratio = torch.as_tensor(ratio, device=device, dtype=torch.float)
        self.force_xyz_scale = torch.distributions.Dirichlet(self.force_xyz_scale_ratio).sample((n,))

        # Force application duration and per-env phase (in [0, 1]).
        dur_lo, dur_hi = cfg.get("randomize_force_duration", (150, 250))
        self.apply_force_duration = torch.randint(int(dur_lo), int(dur_hi) + 1, (n, 1), device=device)
        self.left_ee_apply_force_phase = torch.rand(n, 1, device=device)
        self.right_ee_apply_force_phase = torch.rand(n, 1, device=device)

        # Random-force gating probability (currently 0 in the activated config; left here for parity).
        self.random_force_prob = float(cfg.get("random_force_prob", 0.0))
        self.random_force = (torch.rand(n, 1, device=device) < self.random_force_prob).float()

        # Cache walking-env masks; refreshed each step from commands[:, 4].
        self.walking_env_mask = torch.zeros(n, dtype=torch.bool, device=device)
        # Tapping = walking but linear-velocity command is (0, 0).
        self.tapping_env_mask = torch.zeros(n, dtype=torch.bool, device=device)
        # Random tapping direction in body frame for tapping envs (used by FALCON's resistance projection).
        self.random_tapping_dir = torch.zeros(n, 2, device=device, dtype=torch.float)
        self._resample_random_tapping_dir(torch.arange(n, device=device, dtype=torch.long))

        if not self.ee_forces_enabled:
            logger.warning(
                f"EE forces disabled on {type(self.simulator).__name__} "
                f"(force_config.enabled={self.force_config.get('enabled', True)}, "
                f"jacobian available={hasattr(self.simulator, 'jacobian')})."
            )

    @property
    def ee_forces_enabled(self) -> bool:
        """Whether the Jacobian-based EE force pipeline should run."""
        if not self.force_config.get("enabled", True):
            return False
        return hasattr(self.simulator, "jacobian")

    def _clear_ee_forces(self) -> None:
        self.apply_force_tensor.zero_()
        self.left_ee_apply_force.zero_()
        self.right_ee_apply_force.zero_()

    # ------------------------------------------------------------------
    # Force pipeline

    def _refresh_command_masks(self) -> None:
        cmds = self.command_manager.commands
        self.walking_env_mask = cmds[:, 4] == 1
        zero_cmd_mask = torch.all(cmds[:, :2] == 0, dim=-1)
        self.tapping_env_mask = self.walking_env_mask & zero_cmd_mask

    def _resample_random_tapping_dir(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        choices = torch.randint(0, 2, (env_ids.shape[0], 1), device=self.device)
        pos = torch.tensor([1.0, 0.0], device=self.device)
        neg = torch.tensor([-1.0, 0.0], device=self.device)
        self.random_tapping_dir[env_ids] = torch.where(choices == 0, pos, neg)

    def _calculate_max_ee_forces(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-env Jacobian-based force budget in (x, y, z) for each hand.

        Mirrors FALCON-Terrains' ``force_estimation_type == 0`` branch:
        treat each axis independently, divide the per-arm joint torque
        margin by ``|J|``, then take the most conservative joint per axis.
        """
        jac = self.simulator.jacobian  # (num_envs, num_bodies, 6, 6+num_dofs)
        j_left = jac[:, self.left_hand_link_index, :3, 6:]  # (num_envs, 3, num_dofs)
        j_right = jac[:, self.right_hand_link_index, :3, 6:]
        j_left_arm = j_left[:, :, self.left_arm_dof_indices]  # (num_envs, 3, num_arm_dofs)
        j_right_arm = j_right[:, :, self.right_arm_dof_indices]

        # FALCON sets joint_effort_est to zero, so the budget reduces to ±joint_effort_limit.
        max_delta = self.joint_effort_limit  # (num_dofs,)
        min_delta = -self.joint_effort_limit

        eps = 1.0e-2
        left_max_per_joint = (1.0 / (torch.abs(j_left_arm) + eps)) * max_delta[self.left_arm_dof_indices].view(1, 1, -1)
        left_min_per_joint = (1.0 / (torch.abs(j_left_arm) + eps)) * min_delta[self.left_arm_dof_indices].view(1, 1, -1)
        right_max_per_joint = (1.0 / (torch.abs(j_right_arm) + eps)) * max_delta[self.right_arm_dof_indices].view(
            1, 1, -1
        )
        right_min_per_joint = (1.0 / (torch.abs(j_right_arm) + eps)) * min_delta[self.right_arm_dof_indices].view(
            1, 1, -1
        )

        left_min = torch.max(left_min_per_joint, dim=2)[0]
        left_max = torch.min(left_max_per_joint, dim=2)[0]
        right_min = torch.max(right_min_per_joint, dim=2)[0]
        right_max = torch.min(right_max_per_joint, dim=2)[0]

        # Per-env axis weighting from the Dirichlet sample.
        scale = self.force_xyz_scale  # (num_envs, 3)
        left_min = left_min * scale
        left_max = left_max * scale
        right_min = right_min * scale
        right_max = right_max * scale

        # Clip to absolute force range.
        left_min = torch.clip(left_min, self.force_range_low, self.force_range_high)
        left_max = torch.clip(left_max, self.force_range_low, self.force_range_high)
        right_min = torch.clip(right_min, self.force_range_low, self.force_range_high)
        right_max = torch.clip(right_max, self.force_range_low, self.force_range_high)

        if self.only_apply_z_force_when_walking:
            walking_idx = self.walking_env_mask.nonzero(as_tuple=False).flatten()
            if walking_idx.numel() > 0:
                left_min[walking_idx, :2] = 0.0
                left_max[walking_idx, :2] = 0.0
                right_min[walking_idx, :2] = 0.0
                right_max[walking_idx, :2] = 0.0

        return left_min, left_max, right_min, right_max

    def _scale_forces(
        self,
        left_min: torch.Tensor,
        left_max: torch.Tensor,
        right_min: torch.Tensor,
        right_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a force inside the per-env budget and apply the curriculum scale."""
        left_phased = left_min + (left_max - left_min) * self.left_ee_apply_force_phase
        right_phased = right_min + (right_max - right_min) * self.right_ee_apply_force_phase

        noise_l = torch.rand(self.num_envs, 3, device=self.device)
        noise_r = torch.rand(self.num_envs, 3, device=self.device)
        left_force = left_phased * self.apply_force_scale + noise_l
        right_force = right_phased * self.apply_force_scale + noise_r

        # Per-axis gating (zero out if zero_force probability hit).
        left_force = left_force * (1.0 - self.left_zero_force)
        right_force = right_force * (1.0 - self.right_zero_force)

        left_force = torch.clip(left_force, self.force_range_low, self.force_range_high)
        right_force = torch.clip(right_force, self.force_range_low, self.force_range_high)
        return left_force, right_force

    def _calculate_ee_forces(self) -> None:
        self._refresh_command_masks()
        left_min, left_max, right_min, right_max = self._calculate_max_ee_forces()
        left_force, right_force = self._scale_forces(left_min, left_max, right_min, right_max)

        # Optional resistance-only mode while walking: project XY force onto the negative walking direction.
        if self.only_apply_resistance_when_walking:
            cmds = self.command_manager.commands
            walking_dir = quat_rotate(
                self.base_quat,
                torch.cat(
                    [cmds[:, :2], torch.zeros(self.num_envs, 1, device=self.device, dtype=cmds.dtype)],
                    dim=-1,
                ),
                w_last=True,
            )[:, :2]
            walking_dir_norm = torch.norm(walking_dir, dim=-1, keepdim=True) + 1e-6

            num_tapping = int(self.tapping_env_mask.sum().item())
            if num_tapping > 0:
                tap_idx = self.tapping_env_mask.nonzero(as_tuple=False).flatten()
                walking_dir[tap_idx] = quat_rotate(
                    self.base_quat[tap_idx],
                    torch.cat(
                        [
                            self.random_tapping_dir[tap_idx],
                            torch.zeros(num_tapping, 1, device=self.device),
                        ],
                        dim=-1,
                    ),
                    w_last=True,
                )[:, :2]
                walking_dir_norm[tap_idx] = torch.norm(walking_dir[tap_idx], dim=-1, keepdim=True) + 1e-6

            walking_dir_unit = torch.zeros_like(walking_dir)
            walking_idx = self.walking_env_mask.nonzero(as_tuple=False).flatten()
            walking_dir_unit[walking_idx] = -walking_dir[walking_idx] / walking_dir_norm[walking_idx]

            left_xy = left_force[:, :2]
            right_xy = right_force[:, :2]
            left_proj = left_xy.clone()
            right_proj = right_xy.clone()
            if walking_idx.numel() > 0:
                left_proj[walking_idx] = (
                    torch.abs(torch.sum(left_xy[walking_idx] * walking_dir_unit[walking_idx], dim=-1, keepdim=True))
                    * walking_dir_unit[walking_idx]
                )
                right_proj[walking_idx] = (
                    torch.abs(torch.sum(right_xy[walking_idx] * walking_dir_unit[walking_idx], dim=-1, keepdim=True))
                    * walking_dir_unit[walking_idx]
                )
            left_force = torch.cat([left_proj, left_force[:, 2:3]], dim=-1)
            right_force = torch.cat([right_proj, right_force[:, 2:3]], dim=-1)

        # Cache base-frame forces for observation terms.
        self.left_ee_apply_force = quat_rotate_inverse(self.base_quat, left_force.clone(), w_last=True)
        self.right_ee_apply_force = quat_rotate_inverse(self.base_quat, right_force.clone(), w_last=True)

        # Update the force tensor that gets handed to IsaacGym.
        self.apply_force_tensor[:, self.left_hand_link_index, :] = left_force
        self.apply_force_tensor[:, self.right_hand_link_index, :] = right_force

    def _physics_step(self):
        # Compute the EE forces once per *control* step (FALCON parity), then
        # apply force + torques each physics sub-step inside the decimation loop.
        self.render()
        if self.ee_forces_enabled:
            self._calculate_ee_forces()
        else:
            self._clear_ee_forces()
        for _ in range(self.simulator.simulator_config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        if self.ee_forces_enabled:
            self.simulator.apply_rigid_body_force_at_pos_tensor(self.apply_force_tensor, self.apply_force_pos_tensor)
        if self.action_manager is not None:
            self.action_manager.apply_actions()

    # ------------------------------------------------------------------
    # Hooks

    def _pre_compute_observations_callback(self, env_ids=None):
        super()._pre_compute_observations_callback(env_ids)
        body_pos = self.simulator._rigid_body_pos
        left_pos = body_pos[:, self.left_hand_link_index, :]
        right_pos = body_pos[:, self.right_hand_link_index, :]
        if self.apply_force_pos_mode == "hand_to_extend":
            body_rot = self.simulator._rigid_body_rot
            left_pos = self._interp_to_extend(
                left_pos,
                body_pos[:, self.left_extend_parent_index, :],
                body_rot[:, self.left_extend_parent_index, :],
                self.left_apply_force_pos_ratio,
            )
            right_pos = self._interp_to_extend(
                right_pos,
                body_pos[:, self.right_extend_parent_index, :],
                body_rot[:, self.right_extend_parent_index, :],
                self.right_apply_force_pos_ratio,
            )
        self.apply_force_pos_tensor[:, self.left_hand_link_index, :] = left_pos
        self.apply_force_pos_tensor[:, self.right_hand_link_index, :] = right_pos

    def _interp_to_extend(
        self, hand_pos: torch.Tensor, parent_pos: torch.Tensor, parent_rot: torch.Tensor, ratio: torch.Tensor
    ) -> torch.Tensor:
        """FALCON: ``hand + (extend - hand) * ratio`` with ``extend = parent + R_parent @ offset``."""
        offset = self.extend_offset.unsqueeze(0).expand(hand_pos.shape[0], 3)
        extend_pos = parent_pos + quat_rotate(parent_rot, offset, w_last=True)
        return hand_pos + (extend_pos - hand_pos) * ratio

    @property
    def _resample_force_with_commands(self) -> bool:
        return bool(self.force_config.get("resample_force_with_commands", False))

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        if not self._resample_force_with_commands:
            self._resample_force_settings(env_ids)

    def _post_command_reset_callback(self, env_ids):
        super()._post_command_reset_callback(env_ids)
        if self._resample_force_with_commands:
            self._resample_force_settings(env_ids)

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if not self._resample_force_with_commands or self.is_evaluating:
            return
        resample_time = getattr(self.command_manager.command_cfg, "locomotion_command_resampling_time", None)
        interval = int(resample_time / self.dt) if resample_time else 0
        if interval <= 0:
            return
        env_ids = (self.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_force_settings(env_ids)

    def _resample_force_settings(self, env_ids: torch.Tensor) -> None:
        if env_ids is None:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        cfg = self.force_config
        n = env_ids.shape[0]
        device = self.device

        dur_lo, dur_hi = cfg.get("randomize_force_duration", (150, 250))
        self.apply_force_duration[env_ids] = torch.randint(int(dur_lo), int(dur_hi) + 1, (n, 1), device=device)
        self.left_ee_apply_force_phase[env_ids] = torch.rand(n, 1, device=device)
        self.right_ee_apply_force_phase[env_ids] = torch.rand(n, 1, device=device)
        self.left_zero_force[env_ids] = (torch.rand(n, 3, device=device) < self.left_zero_force_prob).float()
        self.right_zero_force[env_ids] = (torch.rand(n, 3, device=device) < self.right_zero_force_prob).float()
        self.random_force[env_ids] = (torch.rand(n, 1, device=device) < self.random_force_prob).float()

        # Refresh tapping direction for envs that just became tapping envs after reset.
        self._refresh_command_masks()
        tapping_to_resample = self.tapping_env_mask[env_ids]
        if tapping_to_resample.any():
            self._resample_random_tapping_dir(env_ids[tapping_to_resample])

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)
        # Clear stale per-env force outputs.
        self.apply_force_tensor[env_ids] = 0.0
        self.left_ee_apply_force[env_ids] = 0.0
        self.right_ee_apply_force[env_ids] = 0.0

    def _update_log_dict(self):
        super()._update_log_dict()
        if hasattr(self, "apply_force_scale"):
            self.log_dict["apply_force_scale"] = self.apply_force_scale.mean().detach().cpu()
