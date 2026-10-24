"""Height map as an onboard LiDAR elevation map would deliver it, for the height-map student.

Same grid and value as the teachers' ``height_measurements`` (holosoma "falcon" scan: 16 x 10 points in the base
yaw frame, ``root_z - lidar_height_offset - terrain_z`` clipped to [min_height, max_height]), with the errors a
SLAM / elevation-mapping pipeline adds:

* per-episode map offset in x / y (odometry and registration drift) and in z (base height estimate);
* per-step, per-point Gaussian noise (range noise);
* the map is republished at a lower rate than the policy (``update_period_steps``) and with a latency
  (``delay_steps``: the published scan is that many policy steps old);
* at each republish a fraction of the cells keeps its previous value (occluded / not re-observed cells).

``LidarHeightMapNoise`` (randomization term) draws the per-episode parameters and holds the buffers;
``lidar_height_measurements`` (observation term) builds and publishes the scan.
"""

from __future__ import annotations

from typing import Any

from holosoma.managers.randomization.base import RandomizationTermBase
from holosoma.utils.rotations import quat_apply_yaw
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sampler import TermSampler


class LidarHeightMapNoise(RandomizationTermBase):
    """Per-episode error parameters and history buffers for :func:`lidar_height_measurements`.

    Config params (all required):
        xy_offset_range: [lo, hi] map offset per axis in metres (world frame), drawn per episode.
        z_offset_range: [lo, hi] map height offset in metres, drawn per episode.
        point_noise_std: per-step, per-point Gaussian noise in metres.
        delay_steps_range: [lo, hi] latency of the published map in policy steps, drawn per episode.
        update_period_steps: the map is republished every this many policy steps (random phase per episode).
        hole_prob_range: [lo, hi] fraction of cells left stale at each republish, drawn per episode.

    ``set_curriculum_scale`` (DR curriculum) scales the offsets, the noise, the latency and the hole fraction.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        p = cfg.params
        self.xy_offset_range = [float(v) for v in p["xy_offset_range"]]
        self.z_offset_range = [float(v) for v in p["z_offset_range"]]
        self.point_noise_std = float(p["point_noise_std"])
        self.delay_steps_range = [int(v) for v in p["delay_steps_range"]]
        self.update_period_steps = int(p["update_period_steps"])
        self.hole_prob_range = [float(v) for v in p["hole_prob_range"]]
        if self.update_period_steps < 1:
            raise ValueError(f"update_period_steps must be >= 1, got {self.update_period_steps}")
        self.scale = 1.0
        self.num_points: int | None = None

    def setup(self, sampler: TermSampler) -> None:
        n, dev = self.env.num_envs, self.env.device
        self.xy_offset = torch.zeros(n, 2, device=dev)
        self.z_offset = torch.zeros(n, 1, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.hole_prob = torch.zeros(n, 1, device=dev)
        # buffers are sized on the first observation call (number of scan points)
        self.needs_refill = torch.ones(n, dtype=torch.bool, device=dev)
        self.last_stamp = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._resample(torch.arange(n, device=dev, dtype=torch.long), sampler)

    def reset(self, env_ids: torch.Tensor | None, sampler: TermSampler) -> None:
        idx = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx, device=self.env.device, dtype=torch.long)
        if idx.numel() == 0:
            return
        self._resample(idx.to(self.env.device), sampler)
        self.needs_refill[idx] = True

    def step(self, sampler: TermSampler) -> None:
        return

    def set_curriculum_scale(self, scale: float) -> None:
        self.scale = float(scale)

    def _resample(self, env_ids: torch.Tensor, sampler: TermSampler) -> None:
        dev, s = self.env.device, self.scale
        ids = env_ids.cpu()
        xy = [v * s for v in self.xy_offset_range]
        self.xy_offset[env_ids, 0] = sampler.draw(xy, env_ids=ids, coords=(0,)).to(dev)
        self.xy_offset[env_ids, 1] = sampler.draw(xy, env_ids=ids, coords=(1,)).to(dev)
        self.z_offset[env_ids, 0] = sampler.draw([v * s for v in self.z_offset_range], env_ids=ids).to(dev)
        lo, hi = (round(v * s) for v in self.delay_steps_range)
        self.delay[env_ids] = sampler.draw_int(lo, max(hi, lo), env_ids=ids).to(dev)
        self.phase[env_ids] = sampler.draw_int(0, self.update_period_steps - 1, env_ids=ids).to(dev)
        self.hole_prob[env_ids, 0] = sampler.draw([v * s for v in self.hole_prob_range], env_ids=ids).to(dev)

    def ensure_buffers(self, num_points: int) -> None:
        if self.num_points == num_points:
            return
        n, dev = self.env.num_envs, self.env.device
        depth = self.delay_steps_range[1] + 1
        self.num_points = num_points
        self.history = torch.zeros(n, depth, num_points, device=dev)  # newest frame at self.write_idx
        self.write_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.published = torch.zeros(n, num_points, device=dev)
        self.needs_refill[:] = True


def lidar_height_measurements(
    env: Any,
    noise_term: str,
    min_height: float = -0.5,
    max_height: float = 0.5,
) -> torch.Tensor:
    """The teachers' height map with the errors drawn by the ``noise_term`` randomizer (see module doc)."""
    rand: LidarHeightMapNoise = env.randomization_manager.get_state(noise_term)
    terrain = env.terrain_manager.get_state("locomotion_terrain")
    # The teachers' scan grid and heightfield lookup (TerrainLocomotion, "falcon" mode), here at offset points.
    points = terrain._height_scan_points
    p = points.shape[1]
    rand.ensure_buffers(p)
    scan_xy = quat_apply_yaw(env.base_quat.repeat(1, p), points, True) + env.simulator.robot_root_states[:, :3].unsqueeze(1)
    scan_xy[..., :2] += rand.xy_offset.unsqueeze(1)
    terrain_z = terrain._falcon_heightfield_heights(scan_xy, None)
    root_z = env.simulator.robot_root_states[:, 2:3]
    frame = root_z + rand.z_offset - float(env.lidar_height_offset) - terrain_z
    frame = frame + rand.point_noise_std * rand.scale * torch.randn_like(frame)

    # Update each env once per policy step: the manager also calls this for terminal observations.
    stamp = env.episode_length_buf
    fresh = stamp != rand.last_stamp
    refill = rand.needs_refill & fresh
    if refill.any():
        rand.history[refill] = frame[refill].unsqueeze(1)
        rand.published[refill] = frame[refill]
        rand.needs_refill[refill] = False
    step = fresh & ~refill
    if step.any():
        depth = rand.history.shape[1]
        rand.write_idx[step] = (rand.write_idx[step] + 1) % depth
        rand.history[step, rand.write_idx[step]] = frame[step]
        republish = step & ((stamp + rand.phase) % rand.update_period_steps == 0)
        if republish.any():
            ids = republish.nonzero(as_tuple=True)[0]
            delayed = rand.history[ids, (rand.write_idx[ids] - rand.delay[ids]) % depth]
            keep_old = torch.rand_like(delayed) < rand.hole_prob[ids] * rand.scale
            rand.published[ids] = torch.where(keep_old, rand.published[ids], delayed)
    rand.last_stamp[fresh] = stamp[fresh]
    out = torch.nan_to_num(rand.published, nan=0.0, posinf=max_height, neginf=min_height)
    return out.clamp(min_height, max_height)
