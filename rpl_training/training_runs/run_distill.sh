#!/bin/bash
# Distill the FALCON MoE terrain teachers into the RPL multi-depth student (DAgger-MA, depth ViT).
# The four FALCON teacher checkpoints ship in rpl_training/data/models/moe_cnn/falcon_20251105;
# point --algo.config.ckpt-teacher at a directory with the same per-terrain layout to use your own.
# Env: SIM=isaacgym|isaacsim (default isaacgym), NGPUS=1, LOGGER=wandb|disabled,
#      WANDB_PROJECT, WANDB_ENTITY, RUN_NAME. Extra arguments pass through to rpl_training.train_agent.
# --training.num-envs is the total across GPUs (FALCON's 1024 x 8 GPUs = 8192, the preset default).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
rpl_launcher_init
rpl_activate_sim
rpl_training_command
exec "${TRAIN_COMMAND[@]}" -m rpl_training.train_agent \
    exp:g1-29dof-rpl-distill "${SIM_ARGS[@]}" \
    --training.headless True \
    "${LOGGER_ARGS[@]}" \
    "$@"
