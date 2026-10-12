#!/bin/bash
# Build the two sim2sim conda envs from thirdparty/holosoma:
#   SIM_ENV  (default rplmujoco):    MuJoCo + holosoma + rpl  -> python -m rpl_sim2sim.run_sim
#   POLICY_ENV (default rplinference): holosoma_inference + rpl -> python -m rpl_sim2sim.run_policy
# Every install goes through the env's own interpreter with the user site disabled, so nothing
# lands in ~/.local or another Python first on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOLOSOMA="${HOLOSOMA_LOCAL_DIR:-$ROOT/thirdparty/holosoma}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/.holosoma_deps/miniconda3}"
SIM_ENV="${SIM_ENV:-rplmujoco}"
POLICY_ENV="${POLICY_ENV:-rplinference}"
export PYTHONNOUSERSITE=1 PIP_USER=0
unset PYTHONPATH

[[ -x "$CONDA_ROOT/bin/conda" ]] || { echo "ERROR: no conda at $CONDA_ROOT (run thirdparty/holosoma/scripts/setup_mujoco.sh once, or set CONDA_ROOT)" >&2; exit 1; }
[[ -f "$HOLOSOMA/src/holosoma/pyproject.toml" ]] || git -C "$ROOT" submodule update --init --recursive

make_env() {
    local name=$1
    [[ -x "$CONDA_ROOT/envs/$name/bin/python" ]] && return
    "$CONDA_ROOT/bin/conda" create -y -n "$name" python=3.10 pip libstdcxx-ng --override-channels -c conda-forge
}

make_env "$SIM_ENV"
SIM_PY="$CONDA_ROOT/envs/$SIM_ENV/bin/python"
"$CONDA_ROOT/bin/conda" install -y -n "$SIM_ENV" ffmpeg --override-channels -c conda-forge
"$SIM_PY" -m pip install 'mujoco>=3.0.0' mujoco-python-viewer pytest 'gymnasium==1.3.0'  # gymnasium: FALCON's depth renderer
"$SIM_PY" -m pip install -e "$HOLOSOMA/src/holosoma[unitree,booster]"
"$SIM_PY" -m pip install 'cyclonedds==0.10.2' loop_rate_limiters imageio-ffmpeg  # rpl_sim2sim.falcon_sim (FALCON's own sim)
SDK2PY="${SDK2PY_DIR:-$CONDA_ROOT/../unitree_sdk2_python}"  # editable: a wheel install drops its crc .so
[[ -d "$SDK2PY" ]] || git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$SDK2PY"
git -C "$SDK2PY" checkout 432dd06
"$SIM_PY" -m pip install --no-deps -e "$SDK2PY"
"$SIM_PY" -m pip install --no-deps -e "$ROOT"

make_env "$POLICY_ENV"
POLICY_PY="$CONDA_ROOT/envs/$POLICY_ENV/bin/python"
"$POLICY_PY" -m pip install -e "$HOLOSOMA/src/holosoma_inference[unitree,booster]" pytest
"$CONDA_ROOT/bin/conda" install -y -n "$POLICY_ENV" pinocchio --override-channels -c conda-forge
"$POLICY_PY" -m pip install --no-deps -e "$ROOT"

echo "[setup] Done. Terminal 1: bash rpl_sim2sim/run_sim.sh   Terminal 2: bash rpl_sim2sim/run_policy.sh"
