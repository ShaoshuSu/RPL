"""Roll out the distillation preset's MoE teacher (instead of the student) and report how it does.

A quick parity check for the env: FALCON's teachers were trained in FALCON-Terrains, so with a
matching observation layout and dynamics they should survive most episodes and climb the terrain
curriculum here too.

Usage::

    python -m rpl_training.training_runs.eval_teacher --training.num-envs 1024 --training.headless True

Any distillation-preset override applies (e.g. ``--algo.config.ckpt-teacher DIR``). ``EVAL_STEPS``
(default 2000) sets the rollout length.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import rpl_training.config_values.registry  # noqa: F401  (registers the presets)

EXPERIMENT = "exp:g1-29dof-rpl-distill"


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
        algo.setup()  # builds student + teacher modules and loads ``ckpt_teacher``

        steps = int(os.environ.get("EVAL_STEPS", "2000"))
        max_len = int(env.max_episode_length)
        obs = env.reset_all()
        obs = {k: v.to(device) for k, v in obs.items()}
        ep_len = torch.zeros(env.num_envs, device=device)
        finished_len, finished_timeout = [], []
        with torch.inference_mode():
            for _ in range(steps):
                _, per_key = algo._multi_actor_rollout_step(obs)
                actions = torch.cat([per_key[k]["actions_teacher"] for k in algo.keys], dim=1)
                obs, _, dones, infos = env.step({"actions": actions})
                obs = {k: v.to(device) for k, v in obs.items()}
                ep_len += 1
                done_ids = dones.to(device).nonzero(as_tuple=False).flatten()
                if done_ids.numel():
                    finished_len.append(ep_len[done_ids].clone())
                    finished_timeout.append(infos["time_outs"].to(device)[done_ids].float())
                    ep_len[done_ids] = 0
                for key in algo.keys:
                    algo.actors_teacher[key].reset(dones)

        lengths = torch.cat(finished_len) if finished_len else torch.zeros(1, device=device)
        timeouts = torch.cat(finished_timeout) if finished_timeout else torch.zeros(1, device=device)
        terrain = env.terrain_manager.get_state("locomotion_terrain")
        logger.info("==> teacher rollout")
        logger.info(f"steps={steps} envs={env.num_envs} max_episode_length={max_len} episodes={lengths.numel()}")
        logger.info(f"mean_episode_length={lengths.mean().item():.1f} timeout_fraction={timeouts.mean().item():.3f}")
        for i, name in enumerate(terrain.terrain_type_names):
            mask = terrain.terrain_type_idx == i
            if mask.any():
                levels = terrain.terrain_levels[mask].float()
                logger.info(f"terrain_level[{name}] mean={levels.mean().item():.2f} max={levels.max().item():.0f}")
    finally:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
