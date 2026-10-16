#!/bin/bash
# End-to-end check of the RPL setup: parity unit tests, then a short expert run and a
# short distillation run (shipped FALCON teachers). Logs stay local (LOGGER=disabled).
# Env: SIM (default isaacgym), ITERS=20, EXPERT_ENVS=4096, DISTILL_ENVS=1024, NGPUS=1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ITERS="${ITERS:-20}"
export LOGGER=disabled

echo "==> [1/3] unit tests"
SIM="${SIM:-isaacgym}" bash -c 'source rpl_training/training_runs/setup_env.sh && rpl_launcher_init && rpl_activate_sim && python -m pytest rpl_training/tests -q -p no:cacheprovider'

echo "==> [2/3] expert: ${EXPERT_ENVS:-4096} envs x ${ITERS} iterations"
bash rpl_training/training_runs/run_expert.sh --training.name smoke-expert \
    --training.num-envs "${EXPERT_ENVS:-4096}" \
    --algo.config.num-learning-iterations "$ITERS" --algo.config.save-interval 100000

echo "==> [3/3] distill: ${DISTILL_ENVS:-1024} envs x ${ITERS} iterations"
bash rpl_training/training_runs/run_distill.sh --training.name smoke-distill \
    --training.num-envs "${DISTILL_ENVS:-1024}" \
    --algo.config.num-learning-iterations "$ITERS" --algo.config.save-interval 100000

echo "==> smoke test passed"
