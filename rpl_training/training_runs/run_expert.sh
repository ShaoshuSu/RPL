#!/bin/bash
# Train an RPL / FALCON-Terrains RL expert (PPO-MA, CNN height-map encoder).
# FALCON trains one expert per terrain; EXPERT=stones|stairs|slopes picks the preset that
# reproduces that expert's final FALCON stage, e.g.
#   NGPUS=4 EXPERT=stairs bash rpl_training/training_runs/run_expert.sh
# Without EXPERT the generic preset is used and the terrain mix goes on the CLI.
# Env: SIM=isaacgym|isaacsim (default isaacgym), NGPUS=1, LOGGER=wandb|disabled, EXPERT,
#      WANDB_PROJECT, WANDB_ENTITY, RUN_NAME. Extra arguments pass through to rpl_training.train_agent.
# --training.num-envs is the total across GPUs (FALCON's 4096 x 4 GPUs = 16384; the slopes preset keeps FALCON's 4096 total).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
rpl_launcher_init
rpl_activate_sim
rpl_training_command
exec "${TRAIN_COMMAND[@]}" -m rpl_training.train_agent \
    "exp:g1-29dof-rpl-expert${EXPERT:+-${EXPERT}}" "${SIM_ARGS[@]}" \
    --training.headless True \
    "${LOGGER_ARGS[@]}" \
    "$@"
