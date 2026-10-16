"""Loco-manipulation specific termination terms."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def motion_ends(env, **_) -> torch.Tensor:
    """Terminate as a timeout when the assigned motion clip ends."""
    if not getattr(env, "motion_reference_enabled", False):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    motion_len = getattr(env, "motion_len", None)
    motion_start_times = getattr(env, "motion_start_times", None)
    if motion_len is None or motion_start_times is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    current_time = env.episode_length_buf.to(dtype=torch.float32) * env.dt + motion_start_times
    return current_time > motion_len
