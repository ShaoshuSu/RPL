#!/bin/bash
# Start the RPL / FALCON-Terrains sim2sim MuJoCo environment (start this first).
# Extra arguments pass through to `python -m rpl_sim2sim.run_sim`. SIM_ENV (default rplmujoco) picks the env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${CONDA_ROOT:-$HOME/.holosoma_deps/miniconda3}/envs/${SIM_ENV:-rplmujoco}/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY not found; run rpl_sim2sim/setup_env.sh" >&2; exit 1; }
export PYTHONNOUSERSITE=1
cd "$ROOT"
exec "$PY" -m rpl_sim2sim.run_sim "$@"
