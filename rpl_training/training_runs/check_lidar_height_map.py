"""Check the height-map student's LiDAR-style map (rpl_training.terms.lidar_height_map) against its configuration.

Builds the height-map distillation env, lets the MoE teacher drive (so the robots move over the terrain), and
reports, from the student's ``perception_obs`` and the teacher's ``perception_obs_wo_noise``:

* the drawn per-episode parameters (offsets, delay, stale-cell fraction) vs the configured ranges;
* how often the published map changes between policy steps (expected 1 / update_period_steps);
* at each republish, the fraction of cells that kept their previous value (expected ~ the drawn hole fraction);
* the student - teacher difference, overall and its per-env mean against the drawn z offset.

Usage::

    python -m rpl_training.training_runs.check_lidar_height_map --training.num-envs 256 --training.headless True

``CHECK_STEPS`` (default 300) sets the rollout length.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import rpl_training.config_values.registry  # noqa: F401  (registers the presets)

EXPERIMENT = "exp:g1-29dof-rpl-heightmap-distill"
NOISE_TERM = "lidar_height_map_noise"


def main() -> None:
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    args = sys.argv[1:]
    if not any(a.startswith("exp:") for a in args):
        args = [EXPERIMENT, *args]
    if "logger:disabled" not in args:
        args.append("logger:disabled")
    cfg = parse_config(get_annotated_experiment_config, args=args)

    from holosoma.utils.eval_utils import init_sim_imports

    simulation_app = init_sim_imports(cfg)

    import torch
    from holosoma.config_types.env import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app
    from loguru import logger

    try:
        device = "cuda:0"
        env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)
        algo = get_class(cfg.algo._target_)(
            device=device, env=env, config=cfg.algo.config, log_dir=Path(tempfile.mkdtemp()), multi_gpu_cfg=None
        )
        algo.setup()
        rand = env.randomization_manager.get_state(NOISE_TERM)
        params = cfg.randomization.setup_terms[NOISE_TERM].params
        logger.info(f"configured: {params}  curriculum scale now {rand.scale}")
        logger.info(
            f"drawn: xy_offset [{rand.xy_offset.min().item():+.3f}, {rand.xy_offset.max().item():+.3f}] m, "
            f"z_offset [{rand.z_offset.min().item():+.3f}, {rand.z_offset.max().item():+.3f}] m, "
            f"delay {rand.delay.min().item()}-{rand.delay.max().item()} steps, "
            f"hole_prob [{rand.hole_prob.min().item():.3f}, {rand.hole_prob.max().item():.3f}]"
        )

        steps = int(os.environ.get("CHECK_STEPS", "300"))
        obs = env.reset_all()
        obs = {k: v.to(device) for k, v in obs.items()}
        prev = obs["perception_obs"].clone()
        changed_steps = torch.zeros(env.num_envs, device=device)
        kept_frac, diffs, per_env_mean = [], [], []
        with torch.inference_mode():
            for _ in range(steps):
                _, per_key = algo._multi_actor_rollout_step(obs)
                actions = torch.cat([per_key[k]["actions_teacher"] for k in algo.keys], dim=1)
                obs, _, dones, _ = env.step({"actions": actions})
                obs = {k: v.to(device) for k, v in obs.items()}
                student, teacher = obs["perception_obs"], obs["perception_obs_wo_noise"]
                alive = ~dones.to(device).bool()
                changed = (student != prev).any(dim=1) & alive
                changed_steps += changed.float()
                if changed.any():
                    kept_frac.append((student[changed] == prev[changed]).float().mean(dim=1))
                # skip clipped cells (|value| = 0.5): the error there is truncated
                valid = (student.abs() < 0.499) & (teacher.abs() < 0.499)
                d = torch.where(valid, student - teacher, torch.nan)
                diffs.append(d[alive].flatten())
                per_env_mean.append(torch.nanmean(d, dim=1))
                prev = student.clone()
                for key in algo.keys:
                    algo.actors_teacher[key].reset(dones)

        d = torch.cat(diffs)
        d = d[~d.isnan()]
        kept = torch.cat(kept_frac)
        env_mean = torch.nanmean(torch.stack(per_env_mean), dim=0)
        z = rand.z_offset[:, 0]
        ok = ~env_mean.isnan()
        corr = torch.corrcoef(torch.stack([env_mean[ok], z[ok]]))[0, 1].item()
        logger.info("==> lidar height map check")
        logger.info(
            f"map changes on {changed_steps.mean().item() / steps:.3f} of steps "
            f"(expected {1 / params['update_period_steps']:.3f})"
        )
        logger.info(
            f"cells kept at a republish: mean {kept.mean().item():.3f} "
            f"(drawn hole_prob mean {rand.hole_prob.mean().item():.3f})"
        )
        logger.info(
            f"student - teacher [m]: mean {d.mean().item():+.4f} std {d.std().item():.4f} "
            f"p05 {d.quantile(0.05).item():+.4f} p95 {d.quantile(0.95).item():+.4f}"
        )
        logger.info(f"per-env mean difference vs drawn z offset: correlation {corr:.3f}")
    finally:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
