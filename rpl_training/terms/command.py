"""Loco-manipulation command terms (HM-style upper-body reference, etc.)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from holosoma.managers.command.base import CommandTermBase
from holosoma.managers.command.terms.locomotion import LocomotionCommand as _BaseLocomotionCommand
from holosoma.utils.motion_lib import MotionLibRobot
from holosoma.utils.rotations import quat_apply, wrap_to_pi
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


class LocomotionCommand(_BaseLocomotionCommand):
    """9-D command buffer matching RPL decoupled layout.

    Indices: ``0:3`` linear/angular velocity (masked by stand flag), ``4`` locomotion vs stand,
    ``5:8`` waist yaw / roll / pitch commands, ``8`` commanded base-height delta.
    """

    def __init__(self, cfg: Any, env: Any):
        merged = {**(cfg.params or {}), "command_dim": 9}
        super().__init__(replace(cfg, params=merged), env)
        p = cfg.params or {}
        # When True, ``commands[:, 4]`` is *derived* from the command velocity each
        # step (1 = moving, 0 = standing) instead of being sampled directly from
        # ``stand_prob``. This makes "velocity ~= 0" and "stance flag == 0" a true
        # invariant at every timestep — including walk envs whose heading-derived
        # yaw rate has decayed to ~0 — so reward terms can gate on ``commands[:, 4]``
        # without reintroducing tapping at zero velocity. Default off (legacy
        # behavior: flag sampled from ``stand_prob`` only).
        self.stand_flag_from_velocity: bool = bool(p.get("stand_flag_from_velocity", False))
        # Velocity magnitude (linear norm or |yaw|) below which an env counts as
        # standing when ``stand_flag_from_velocity`` is enabled.
        self.stand_vel_threshold: float = float(p.get("stand_vel_threshold", 0.1))
        # Persistent per-env "designated stand" mask sampled from ``stand_prob``.
        # Kept across the resample interval so a stand env's heading-derived yaw
        # rate stays suppressed (otherwise it would start turning mid-interval).
        self._stand_sampled: torch.Tensor | None = None
        # Optional per-terrain stand probability (FALCON's per-terrain-index
        # ``stand_prob`` list), mapping a terrain-type name to its probability.
        # Terrain types not listed keep the scalar ``stand_prob``. ``None`` disables.
        self._stand_prob_terrain: dict[str, float] | None = p.get("stand_prob_terrain")
        # FALCON ``zero_fix_waist_{yaw,roll,pitch}_prob``: per-axis probability that
        # a stand env gets *no* waist command for the interval. ``0`` keeps every
        # axis active (legacy behavior, no extra random draw).
        self.zero_fix_waist_probs: tuple[float, float, float] = (
            float(p.get("zero_fix_waist_yaw_prob", 0.0)),
            float(p.get("zero_fix_waist_roll_prob", 0.0)),
            float(p.get("zero_fix_waist_pitch_prob", 0.0)),
        )
        # Terrain-type names on which the waist / base-height commands stay active
        # (FALCON ``waist_commands_mask`` / ``base_height_commands_mask``). Same
        # semantics as the velocity terrain masks; ``None`` disables.
        self._waist_terrain_mask: list[str] | None = p.get("waist_terrain_mask")
        self._base_height_terrain_mask: list[str] | None = p.get("base_height_terrain_mask")

    @property
    def _stand_mask(self) -> torch.Tensor:
        """Per-env designated-stand decision (allocated in ``setup``)."""
        assert self._stand_sampled is not None, "setup() must run before the stand mask is read"
        return self._stand_sampled

    def setup(self) -> None:
        super().setup()
        env = self.env
        p = self.cfg.params or {}
        env.command_base_height_nominal = float(p.get("default_base_height", env.robot_config.init_state.pos[2]))

        if self._stand_sampled is None:
            self._stand_sampled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Per-env command scales (curriculum can overwrite during setup/reset).
        if not hasattr(env, "command_height_scale"):
            env.command_height_scale = torch.ones(env.num_envs, 1, dtype=torch.float32, device=env.device)
        if not hasattr(env, "waist_command_scale"):
            env.waist_command_scale = torch.ones(env.num_envs, 1, dtype=torch.float32, device=env.device)
        if not hasattr(env, "upper_body_motion_scale"):
            env.upper_body_motion_scale = torch.ones(env.num_envs, 1, dtype=torch.float32, device=env.device)

    def _resample(self, env_ids: torch.Tensor) -> None:
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return

        device = self.env.device
        ranges = self.command_ranges
        ne = env_ids.shape[0]

        commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0], ranges["lin_vel_x"][1], (ne, 1), device=device
        ).squeeze(1)
        commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0], ranges["lin_vel_y"][1], (ne, 1), device=device
        ).squeeze(1)
        commands[env_ids, 2] = torch_rand_float(
            ranges["ang_vel_yaw"][0], ranges["ang_vel_yaw"][1], (ne, 1), device=device
        ).squeeze(1)

        # Sample the "designated stand" decision from ``stand_prob`` and persist
        # it (so velocity stays suppressed for the whole interval). The actual
        # ``commands[:, 4]`` flag is set below.
        self._sample_stand(env_ids)
        commands[env_ids, :3] *= (~self._stand_mask[env_ids]).float().unsqueeze(1)
        self._update_stand_flag(env_ids)

        self._resample_waist_and_height(env_ids)

    def _sample_stand(self, env_ids: torch.Tensor) -> None:
        """Sample the persistent per-env ``stand_prob`` decision into ``_stand_sampled``."""
        if self._stand_sampled is None:
            self._stand_sampled = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self._stand_sampled[env_ids] = torch.rand(env_ids.shape[0], device=self.env.device) <= self._stand_prob_for(
            env_ids
        )

    def _stand_prob_for(self, env_ids: torch.Tensor) -> torch.Tensor | float:
        """Stand probability per env: scalar ``stand_prob`` unless ``stand_prob_terrain`` is set."""
        if not self._stand_prob_terrain:
            return self.stand_prob
        probs = torch.full((env_ids.shape[0],), self.stand_prob, device=self.env.device, dtype=torch.float32)
        for name, prob in self._stand_prob_terrain.items():
            mask = self._terrain_mask([name], env_ids)
            if mask is None:
                break
            probs = torch.where(mask.bool(), torch.full_like(probs, float(prob)), probs)
        return probs

    def _terrain_mask(self, names: list[str] | None, env_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        """Float 0/1 mask over envs whose terrain type is in ``names``.

        Semantics:
            - ``None``  → disable masking (return ``None``; caller leaves command as-is).
            - ``[]``    → empty allow-list → zero everywhere.
            - ``[...]`` → keep command on listed terrain types; zero elsewhere.

        Names not present in the active terrain are silently ignored.
        """
        if names is None:
            return None
        size = self.env.num_envs if env_ids is None else int(env_ids.numel())
        if len(names) == 0:
            return torch.zeros(size, device=self.env.device, dtype=torch.float32)
        state = getattr(self.env, "terrain_manager", None)
        if state is None:
            return None
        try:
            term = state.get_state("locomotion_terrain")
        except Exception:
            return None
        type_names = getattr(term, "terrain_type_names", None)
        if not type_names:
            return None
        idx_list = [type_names.index(n) for n in names if n in type_names]
        if not idx_list:
            return torch.zeros(size, device=self.env.device, dtype=torch.float32)
        allowed = torch.tensor(idx_list, device=self.env.device, dtype=torch.long)
        per_env = term.terrain_type_idx if env_ids is None else term.terrain_type_idx[env_ids]
        return torch.isin(per_env, allowed).float()

    def _update_stand_flag(self, env_ids: torch.Tensor | None = None) -> None:
        """Set ``commands[:, 4]`` (1 = locomotion, 0 = stand).

        With ``stand_flag_from_velocity`` the flag is derived from the current
        command velocity, so it tracks heading-aligned walk envs that have
        decayed to ~0 velocity. Otherwise it falls back to the persistent
        ``stand_prob`` decision (legacy behavior).
        """
        commands = self.commands
        if commands is None:
            return
        sel = slice(None) if env_ids is None else env_ids
        if self.stand_flag_from_velocity:
            moving = (torch.norm(commands[sel, :2], dim=1) > self.stand_vel_threshold) | (
                torch.abs(commands[sel, 2]) > self.stand_vel_threshold
            )
            commands[sel, 4] = moving.float()
        else:
            commands[sel, 4] = (~self._stand_mask[sel]).float()

    def _resample_waist_and_height(self, env_ids: torch.Tensor) -> None:
        """Sample waist and base-height commands (shared by both locomotion variants)."""
        commands = self.commands
        if commands is None:
            return
        device = self.env.device
        ranges = self.command_ranges
        ne = env_ids.shape[0]

        wy = ranges.get("waist_yaw", (-0.35, 0.35))
        wr = ranges.get("waist_roll", (-0.2, 0.2))
        wp = ranges.get("waist_pitch", (-0.35, 0.35))
        waist_scale = getattr(self.env, "waist_command_scale", None)
        if waist_scale is None:
            waist_scale_local = torch.ones(ne, dtype=torch.float32, device=device)
        else:
            waist_scale_local = waist_scale[env_ids].squeeze(-1)
        commands[env_ids, 5] = torch_rand_float(float(wy[0]), float(wy[1]), (ne, 1), device=device).squeeze(1)
        commands[env_ids, 6] = torch_rand_float(float(wr[0]), float(wr[1]), (ne, 1), device=device).squeeze(1)
        commands[env_ids, 7] = torch_rand_float(float(wp[0]), float(wp[1]), (ne, 1), device=device).squeeze(1)
        commands[env_ids, 5:8] *= waist_scale_local.unsqueeze(-1)
        if any(prob > 0.0 for prob in self.zero_fix_waist_probs):
            # FALCON: an axis stays active iff ``rand > zero_fix_waist_<axis>_prob``.
            probs = torch.tensor(self.zero_fix_waist_probs, device=device, dtype=torch.float32)
            commands[env_ids, 5:8] *= (torch.rand(ne, 3, device=device) > probs).float()
        waist_mask = self._terrain_mask(self._waist_terrain_mask, env_ids)
        if waist_mask is not None:
            commands[env_ids, 5:8] *= waist_mask.unsqueeze(-1)

        bh = ranges.get("base_height", (-0.3, 0.0))
        base_height_scale = getattr(self.env, "command_height_scale", None)
        if base_height_scale is None:
            base_height_scale_local = torch.ones(ne, dtype=torch.float32, device=device)
        else:
            base_height_scale_local = base_height_scale[env_ids].squeeze(-1)
        commands[env_ids, 8] = torch_rand_float(float(bh[0]), float(bh[1]), (ne, 1), device=device).squeeze(1)
        commands[env_ids, 8] *= base_height_scale_local
        base_height_mask = self._terrain_mask(self._base_height_terrain_mask, env_ids)
        if base_height_mask is not None:
            commands[env_ids, 8] *= base_height_mask

        # Waist and base-height commands are only active in stand mode. Gate on
        # the persistent stand decision so the activation matches the velocity
        # suppression (and doesn't flicker if the velocity-derived flag flips
        # mid-interval). Falls back to the flag when no stand buffer exists.
        if self._stand_sampled is not None:
            stance = self._stand_sampled[env_ids].float()
        else:
            stance = 1.0 - commands[env_ids, 4]
        commands[env_ids, 5:9] *= stance.unsqueeze(-1)


class HeadingLocomotionCommand(LocomotionCommand):
    """Heading-based locomotion command matching RPL behaviour.

    Instead of directly sampling ``ang_vel_yaw``, this term samples a **target
    heading** (stored in ``commands[:, 3]``) and derives the yaw angular velocity
    every step via a P-controller::

        ang_vel_yaw = clip(Kp * wrap_to_pi(target_heading - current_heading),
                           ang_vel_yaw_min, ang_vel_yaw_max)

    This produces straight-line walking toward the target heading, which
    maximises displacement from spawn and improves terrain curriculum
    progression.

    Extra params (on top of :class:`LocomotionCommand`):
        heading_p_gain (float): Proportional gain for the heading controller.
            Default ``0.5`` (matches RPL).
        lin_vel_y_terrain_mask (list[str] | None): Terrain-type names on which the
            lateral (y) velocity command remains active; on all others it is zeroed
            at resample time. ``None`` disables masking.
        heading_terrain_mask (list[str] | None): Terrain-type names on which the
            heading command remains active; zeroed elsewhere (which also zeroes the
            derived ``ang_vel_yaw``).
        ang_vel_yaw_terrain_mask (list[str] | None): Terrain-type names on which
            the derived yaw-rate command remains active; zeroed elsewhere each step.
        lin_vel_x_terrain_ranges (dict[str, list[float]] | None): Optional
            per-terrain forward-velocity ranges, mapping a terrain-type name to
            ``[lo, hi]``. Envs on a listed terrain sample ``commands[:, 0]`` from
            that range instead of the global ``command_ranges["lin_vel_x"]``;
            unlisted terrains keep the global range. ``None`` disables.
        falcon_stance_resample (bool): Reproduce FAR-FALCON-Terrains'
            ``_resample_locomotion_commands`` order exactly: deadband before the
            terrain masks, then the stance flag drawn *twice* (once inside the
            parent resample, once after the vy / heading masks). The velocity is
            zeroed if either draw says stand, but ``commands[:, 4]`` keeps the
            second draw, so ``(1 - p) * p`` of envs are "tapping" (walk flag,
            zero linear velocity, yaw still tracking the heading). Default off.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        p = cfg.params or {}
        self.heading_p_gain: float = float(p.get("heading_p_gain", 0.5))
        # Linear-velocity deadband: lin-vel commands with norm below this are
        # zeroed at resample time (matching FALCON). Default ``0.2``.
        self.lin_vel_deadband: float = float(p.get("lin_vel_deadband", 0.2))
        self._lin_vel_y_terrain_mask: list[str] | None = p.get("lin_vel_y_terrain_mask")
        self._heading_terrain_mask: list[str] | None = p.get("heading_terrain_mask")
        self._ang_vel_yaw_terrain_mask: list[str] | None = p.get("ang_vel_yaw_terrain_mask")
        # Optional per-terrain forward-velocity (vx) sampling ranges. Maps a
        # terrain-type name to ``[lo, hi]``; envs on that terrain sample
        # ``commands[:, 0]`` from the per-terrain range instead of the global
        # ``command_ranges["lin_vel_x"]``. Terrain types not listed (or absent
        # from the active terrain) keep the global range. ``None``/empty disables.
        self._lin_vel_x_terrain_ranges: dict[str, list[float]] | None = p.get("lin_vel_x_terrain_ranges")
        self.falcon_stance_resample: bool = bool(p.get("falcon_stance_resample", False))
        if self.falcon_stance_resample and self.stand_flag_from_velocity:
            raise ValueError("falcon_stance_resample and stand_flag_from_velocity are mutually exclusive.")
        self._forward_vec: torch.Tensor | None = None

    def _resample_lin_vel_x_per_terrain(self, env_ids: torch.Tensor) -> None:
        """Re-draw ``commands[:, 0]`` from per-terrain ranges where configured.

        For each terrain type in ``lin_vel_x_terrain_ranges`` that is present in
        the active terrain, envs currently on that terrain get their forward
        velocity resampled from the per-terrain ``[lo, hi]``. Envs on terrains
        without an override keep the global ``command_ranges["lin_vel_x"]`` value
        sampled by the caller.
        """
        ranges = self._lin_vel_x_terrain_ranges
        if not ranges:
            return
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return
        device = self.env.device
        for name, (lo, hi) in ranges.items():
            mask = self._terrain_mask([name], env_ids)
            # ``None`` => no terrain info available; skip silently. Empty/zero
            # mask => no env on this terrain in the batch; nothing to override.
            if mask is None:
                return
            sel = mask.bool()
            if not bool(sel.any()):
                continue
            n = int(sel.sum())
            sampled = torch_rand_float(float(lo), float(hi), (n, 1), device=device).squeeze(1)
            commands[env_ids[sel], 0] = sampled

    def setup(self) -> None:
        super().setup()
        self._forward_vec = torch.tensor(
            [[1.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.env.device,
        ).expand(self.env.num_envs, -1)

    def _resample(self, env_ids: torch.Tensor) -> None:
        commands = self.commands
        if commands is None or env_ids.numel() == 0:
            return
        if self.falcon_stance_resample:
            self._resample_falcon(env_ids)
            return

        device = self.env.device
        ranges = self.command_ranges
        ne = env_ids.shape[0]

        commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0], ranges["lin_vel_x"][1], (ne, 1), device=device
        ).squeeze(1)
        # Override the forward-velocity sample on terrains with a per-terrain
        # range (e.g. tighter, forward-only ranges on stairs/stones).
        self._resample_lin_vel_x_per_terrain(env_ids)
        commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0], ranges["lin_vel_y"][1], (ne, 1), device=device
        ).squeeze(1)

        # Zero the lateral velocity command on terrains outside the configured mask
        # (FALCON's ``lin_vel_y_commands_mask``).
        lin_vel_y_mask = self._terrain_mask(self._lin_vel_y_terrain_mask, env_ids)
        if lin_vel_y_mask is not None:
            commands[env_ids, 1] *= lin_vel_y_mask

        heading_range = ranges.get("heading", (-math.pi, math.pi))
        commands[env_ids, 3] = torch_rand_float(heading_range[0], heading_range[1], (ne, 1), device=device).squeeze(1)

        # Zero the heading command on terrains outside the configured mask
        # (FALCON's ``heading_commands_mask``). This also zeroes the derived
        # ``ang_vel_yaw`` since it is computed from the heading error.
        heading_mask = self._terrain_mask(self._heading_terrain_mask, env_ids)
        if heading_mask is not None:
            commands[env_ids, 3] *= heading_mask

        # Sample and persist the "designated stand" decision from ``stand_prob``.
        self._sample_stand(env_ids)

        # Derive initial ang_vel_yaw from the freshly sampled heading (gated by
        # the persistent stand decision, not the velocity-derived flag, to avoid
        # a circular dependency).
        self._update_ang_vel_from_heading(env_ids)

        # Suppress all velocity for designated-stand envs.
        commands[env_ids, :3] *= (~self._stand_mask[env_ids]).float().unsqueeze(1)

        # Zero out small lin-vel commands (matching FALCON).
        commands[env_ids, :2] *= (torch.norm(commands[env_ids, :2], dim=1) > self.lin_vel_deadband).unsqueeze(1)

        # Set the stance/locomotion flag from the resulting velocity.
        self._update_stand_flag(env_ids)

        self._resample_waist_and_height(env_ids)

    def _resample_falcon(self, env_ids: torch.Tensor) -> None:
        """FALCON ``..._diff_force_terrain._resample_locomotion_commands`` + waist / height.

        Mirrors the call chain ``locomotion._resample_locomotion_commands`` (sample,
        deadband) -> ``_resample_stance_commands`` -> terrain vy / heading masks ->
        ``_resample_stance_commands`` again. The yaw rate is derived from the heading
        each step and gated on the final stance flag, as in FALCON.
        """
        commands = self.commands
        assert commands is not None
        device = self.env.device
        ranges = self.command_ranges
        ne = env_ids.shape[0]

        commands[env_ids, 0] = torch_rand_float(
            ranges["lin_vel_x"][0], ranges["lin_vel_x"][1], (ne, 1), device=device
        ).squeeze(1)
        self._resample_lin_vel_x_per_terrain(env_ids)
        commands[env_ids, 1] = torch_rand_float(
            ranges["lin_vel_y"][0], ranges["lin_vel_y"][1], (ne, 1), device=device
        ).squeeze(1)
        heading_range = ranges.get("heading", (-math.pi, math.pi))
        commands[env_ids, 3] = torch_rand_float(heading_range[0], heading_range[1], (ne, 1), device=device).squeeze(1)
        commands[env_ids, :2] *= (torch.norm(commands[env_ids, :2], dim=1) > self.lin_vel_deadband).unsqueeze(1)

        # First stance draw (inside FALCON's parent ``_resample_locomotion_commands``).
        stand_prob = self._stand_prob_for(env_ids)
        walk_first = (torch.rand(ne, device=device) > stand_prob).float()
        commands[env_ids, :3] *= walk_first.unsqueeze(1)

        lin_vel_y_mask = self._terrain_mask(self._lin_vel_y_terrain_mask, env_ids)
        if lin_vel_y_mask is not None:
            commands[env_ids, 1] *= lin_vel_y_mask
        heading_mask = self._terrain_mask(self._heading_terrain_mask, env_ids)
        if heading_mask is not None:
            commands[env_ids, 3] *= heading_mask

        # Second stance draw: this one sets the flag.
        self._sample_stand(env_ids)
        commands[env_ids, :3] *= (~self._stand_mask[env_ids]).float().unsqueeze(1)
        self._update_stand_flag(env_ids)
        self._update_ang_vel_from_heading(env_ids)

        self._resample_waist_and_height(env_ids)

    def step(self) -> None:
        """Derive ``ang_vel_yaw`` from heading error every step, refresh the
        stance flag from the resulting velocity, then run the normal mid-episode
        resample logic from the parent."""
        self._update_ang_vel_from_heading()
        # Refresh ``commands[:, 4]`` so it tracks the just-updated yaw rate (a
        # heading-aligned walk env decays to ~0 velocity → flips to stance).
        # Only in velocity-derived mode; legacy keeps the resample-time flag.
        if self.stand_flag_from_velocity:
            self._update_stand_flag()
        super().step()

    def _update_ang_vel_from_heading(self, env_ids: torch.Tensor | None = None) -> None:
        commands = self.commands
        if commands is None or self._forward_vec is None:
            return
        env = self.env
        if not hasattr(env, "base_quat"):
            return

        if env_ids is not None and env_ids.numel() == 0:
            return

        base_quat = env.base_quat if env_ids is None else env.base_quat[env_ids]
        fwd = self._forward_vec if env_ids is None else self._forward_vec[env_ids]
        forward = quat_apply(base_quat, fwd, w_last=True)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        target_heading = commands[:, 3] if env_ids is None else commands[env_ids, 3]
        ang_vel = self.heading_p_gain * wrap_to_pi(target_heading - heading)

        lo = self.command_ranges["ang_vel_yaw"][0]
        hi = self.command_ranges["ang_vel_yaw"][1]
        ang_vel = torch.clamp(ang_vel, lo, hi)

        # Gate on the persistent stand decision (``_stand_sampled``) rather than
        # ``commands[:, 4]``: the flag is now derived from velocity, so gating on
        # it here would be circular (ang_vel -> flag -> ang_vel). In eval the
        # command is held at zero, so suppress entirely (``_stand_sampled`` is not
        # resampled in eval and would otherwise read as locomotion).
        if env.is_evaluating or self._stand_sampled is None:
            locomotion = torch.zeros_like(ang_vel) if env.is_evaluating else torch.ones_like(ang_vel)
        elif env_ids is None:
            locomotion = (~self._stand_sampled).float()
        else:
            locomotion = (~self._stand_mask[env_ids]).float()

        if env_ids is None:
            commands[:, 2] = ang_vel * locomotion
        else:
            commands[env_ids, 2] = ang_vel * locomotion

        # Apply terrain mask (FALCON's ``ang_vel_yaw_commands_mask``).
        ang_vel_mask = self._terrain_mask(self._ang_vel_yaw_terrain_mask, env_ids)
        if ang_vel_mask is not None:
            if env_ids is None:
                commands[:, 2] *= ang_vel_mask
            else:
                commands[env_ids, 2] *= ang_vel_mask


class UpperBodyRefPoseCommand(CommandTermBase):
    """Samples per-env upper-body reference from RPL-style motion library when configured.

    Optional FAR-FALCON-Terrains parity params (all default off):
        fix_upper_body_prob (float): Per resample interval, probability that an env's
            upper-body reference is frozen at a single motion time drawn from
            ``U(0, max_episode_length_s)`` (FALCON ``fix_upper_body_prob``). Training only.
        motion_time_lookahead_steps (int): Reference motion time is
            ``(episode_length + k) * dt + start``; FALCON uses ``k = 1``.
        reset_dof_to_motion (bool): On reset, set the upper-body joint positions and
            velocities to the (scaled) motion state instead of ``default * U(0.5, 1.5)``.
        reset_pose_use_prev_episode_time (bool): FALCON evaluates the reset pose with the
            *previous* episode's length (its ``episode_length_buf`` is not yet cleared
            in ``_reset_dofs``). Only used with ``reset_dof_to_motion``.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        self.ref_upper_dof_pos: torch.Tensor | None = None
        self.upper_dof_indices: torch.Tensor | None = None
        self.motion_lib: MotionLibRobot | None = None
        p = cfg.params or {}
        self.fix_upper_body_prob: float = float(p.get("fix_upper_body_prob", 0.0))
        self.motion_time_lookahead_steps: int = int(p.get("motion_time_lookahead_steps", 0))
        self.reset_dof_to_motion: bool = bool(p.get("reset_dof_to_motion", False))
        self.reset_pose_use_prev_episode_time: bool = bool(p.get("reset_pose_use_prev_episode_time", False))
        self.fix_upper_body: torch.Tensor | None = None
        self.fix_upper_body_motion_times: torch.Tensor | None = None

    def setup(self) -> None:
        env = self.env
        rc = env.robot_config
        idx = [rc.dof_names.index(n) for n in rc.upper_dof_names]
        self.upper_dof_indices = torch.tensor(idx, device=env.device, dtype=torch.long)
        env.upper_dof_indices = self.upper_dof_indices

        n_upper = len(idx)
        self.ref_upper_dof_pos = torch.zeros(env.num_envs, n_upper, dtype=torch.float32, device=env.device)
        env.ref_upper_dof_pos = self.ref_upper_dof_pos

        env.motion_reference_enabled = False
        env.motion_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.motion_start_times = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env.motion_len = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env.motion_start_idx = 0
        env.num_motions = 0
        self.fix_upper_body = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.fix_upper_body_motion_times = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        if rc.motion is None:
            return

        motion_cfg = replace(rc.motion, step_dt=env.dt)
        self.motion_lib = MotionLibRobot(motion_cfg, num_envs=env.num_envs, device=env.device)
        if env.is_evaluating:
            self.motion_lib.load_motions(random_sample=False)
        else:
            self.motion_lib.load_motions(random_sample=True)
        env.motion_reference_enabled = True
        env.num_motions = self.motion_lib._num_unique_motions

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if self.ref_upper_dof_pos is None or self.upper_dof_indices is None:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return
        if self.motion_lib is not None:
            self.env.motion_ids[idx] = self.motion_lib.motion_ids[idx]
            self.env.motion_len[idx] = self.motion_lib.get_motion_length(self.env.motion_ids[idx])
            if self.env.is_evaluating:
                self.env.motion_start_times[idx] = torch.zeros(idx.numel(), dtype=torch.float32, device=self.env.device)
            else:
                self.env.motion_start_times[idx] = self.motion_lib.sample_time(self.env.motion_ids[idx])
            self._resample_fix_upper_body(idx)
            if self.reset_dof_to_motion:
                self._reset_upper_dofs_to_motion(idx)
            self._set_ref_from_motion(idx)
            return
        if self.env.is_evaluating:
            default = self.env.default_dof_pos[idx][:, self.upper_dof_indices]
            self.ref_upper_dof_pos[idx] = default
            return
        self._set_default_upper_ref(idx)

    def step(self) -> None:
        if self.ref_upper_dof_pos is None:
            return
        if self.motion_lib is not None:
            if self.fix_upper_body_prob > 0.0 and not self.env.is_evaluating:
                command_cfg = getattr(self.manager, "command_cfg", None) if hasattr(self, "manager") else None
                resample_time = getattr(command_cfg, "locomotion_command_resampling_time", None)
                interval = int(resample_time / self.env.dt) if resample_time else 0
                if interval > 0:
                    ids = (self.env.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
                    self._resample_fix_upper_body(ids)
            self._set_ref_from_motion(None)
            return

    def _resample_fix_upper_body(self, env_ids: torch.Tensor) -> None:
        """FALCON: freeze the upper-body reference w.p. ``fix_upper_body_prob`` (training only)."""
        if self.fix_upper_body is None or self.fix_upper_body_motion_times is None or env_ids.numel() == 0:
            return
        if self.fix_upper_body_prob <= 0.0 or self.env.is_evaluating:
            self.fix_upper_body[env_ids] = False
            return
        n = env_ids.shape[0]
        device = self.env.device
        self.fix_upper_body[env_ids] = torch.rand(n, device=device) < self.fix_upper_body_prob
        self.fix_upper_body_motion_times[env_ids] = torch.rand(n, device=device) * float(self.env.max_episode_length_s)

    def _scale_upper(self, ref_upper: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        upper_scale = getattr(self.env, "upper_body_motion_scale", None)
        if upper_scale is None or self.env.is_evaluating:
            return ref_upper
        default_upper = self.env.default_dof_pos[idx][:, self.upper_dof_indices]
        return default_upper + (ref_upper - default_upper) * upper_scale[idx]

    def _reset_upper_dofs_to_motion(self, idx: torch.Tensor) -> None:
        """FALCON ``_reset_dofs``: upper-body joints start on the (scaled) motion state."""
        assert self.motion_lib is not None and self.upper_dof_indices is not None
        env = self.env
        if self.reset_pose_use_prev_episode_time and hasattr(env, "_pending_episode_lengths"):
            steps = env._pending_episode_lengths[idx]
        else:
            steps = env.episode_length_buf[idx]
        motion_times = steps.to(dtype=torch.float32) * env.dt + env.motion_start_times[idx]
        motion_res = self.motion_lib.get_motion_state(env.motion_ids[idx], motion_times, offset=None)
        dof_pos = self._scale_upper(motion_res["dof_pos"][:, self.upper_dof_indices], idx)
        dof_vel = motion_res["dof_vel"][:, self.upper_dof_indices]
        upper_scale = getattr(env, "upper_body_motion_scale", None)
        if upper_scale is not None and not env.is_evaluating:
            dof_vel = dof_vel * upper_scale[idx]
        rows = idx.unsqueeze(1)
        env.simulator.dof_pos[rows, self.upper_dof_indices] = dof_pos.to(env.simulator.dof_pos.dtype)
        env.simulator.dof_vel[rows, self.upper_dof_indices] = dof_vel.to(env.simulator.dof_vel.dtype)

    def _set_default_upper_ref(self, env_ids: torch.Tensor) -> None:
        if self.ref_upper_dof_pos is None or self.upper_dof_indices is None or env_ids.numel() == 0:
            return
        env = self.env
        default = env.default_dof_pos[env_ids][:, self.upper_dof_indices]
        # Set default upper-body targets.
        self.ref_upper_dof_pos[env_ids] = default

    def _set_ref_from_motion(self, env_ids: torch.Tensor | None) -> None:
        if self.motion_lib is None or self.ref_upper_dof_pos is None or self.upper_dof_indices is None:
            return

        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0:
            return

        steps = self.env.episode_length_buf[idx].to(dtype=torch.float32) + self.motion_time_lookahead_steps
        motion_times = steps * self.env.dt + self.env.motion_start_times[idx]
        if self.fix_upper_body is not None and self.fix_upper_body_motion_times is not None:
            motion_times = torch.where(self.fix_upper_body[idx], self.fix_upper_body_motion_times[idx], motion_times)
        env_origins = self._get_env_origins(idx)

        motion_res = self.motion_lib.get_motion_state(self.env.motion_ids[idx], motion_times, offset=env_origins)
        ref_upper = motion_res["dof_pos"][:, self.upper_dof_indices]
        self.ref_upper_dof_pos[idx] = self._scale_upper(ref_upper, idx)

    def _get_env_origins(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        simulator_scene = getattr(self.env.simulator, "scene", None)
        if simulator_scene is not None and hasattr(simulator_scene, "env_origins"):
            return simulator_scene.env_origins[env_ids]
        if hasattr(self.env.simulator, "env_origins"):
            return self.env.simulator.env_origins[env_ids]
        return None

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)


__all__ = ["HeadingLocomotionCommand", "LocomotionCommand", "UpperBodyRefPoseCommand"]
