#!/bin/bash
#SBATCH --job-name=rpl_hsgym_setup
#SBATCH --partition=sairpro
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch_sair/shaoshus/locomotion/RPL_heightmap_student/logs/slurm/%x_%j.out

# Builds Holosoma's IsaacGym env `hsgym` for rpl_training on the SLURM cluster (no GPU needed).
# The cluster home is full, and holosoma's source_common.sh always uses $HOME/.holosoma_deps, so that path is a
# symlink to scratch; IsaacGym Preview 4 is linked from the existing copy instead of downloaded.
# Usage (login node, repo root): mkdir -p logs/slurm && sbatch rpl_training/training_runs/slurm/setup_env.sbatch
set -euo pipefail
REPO=/scratch_sair/shaoshus/locomotion/RPL_heightmap_student
DEPS=/scratch_sair/shaoshus/holosoma_deps
ISAACGYM=/scratch_sair/shaoshus/locomotion/isaacgym

mkdir -p "$DEPS"
[[ -L "$HOME/.holosoma_deps" || ! -e "$HOME/.holosoma_deps" ]] || { echo "$HOME/.holosoma_deps exists and is not a symlink" >&2; exit 1; }
ln -sfn "$DEPS" "$HOME/.holosoma_deps"
ln -sfn "$ISAACGYM" "$DEPS/isaacgym"
export PIP_CACHE_DIR="$DEPS/pip_cache"
# The latest Miniconda's conda-libmamba-solver crashes on conda-forge's index (job 17172:
# "PackageInfo.__init__(): incompatible constructor arguments"); the classic solver works.
export CONDA_SOLVER=classic
echo "node=$(hostname) deps=$(readlink -f "$HOME/.holosoma_deps")"; df -h "$DEPS" | tail -1

cd "$REPO"
bash rpl_training/training_runs/setup_env.sh

source "$DEPS/miniconda3/bin/activate" hsgym
python - <<'EOF'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for p in ["torch", "isaacgym", "holosoma", "rpl-training", "warp-lang"]:
    try:
        print(f"{p:14s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p:14s} not installed")
EOF
du -sh "$DEPS/miniconda3/envs/hsgym"
echo SETUP_DONE
