"""Train RPL / FALCON-Terrains policies on Holosoma.

Usage::

    # Per-terrain RL expert
    python -m rpl_training.train_agent exp:g1-29dof-falcon-hm-force-falcon --training.headless True

    # Multi-depth DAgger distillation from the shipped FALCON MoE teachers
    python -m rpl_training.train_agent exp:g1-29dof-falcon-hm-force-falcon-distill --training.headless True

Any Holosoma CLI override applies (``simulator:isaacsim``, ``--training.num-envs``, ``logger:wandb``, ...).
``--training.num-envs`` is the total across GPUs.
"""

from __future__ import annotations

from holosoma.train_agent import main

import rpl_training.config_values.registry  # noqa: F401  (registers the presets)

if __name__ == "__main__":
    main()
