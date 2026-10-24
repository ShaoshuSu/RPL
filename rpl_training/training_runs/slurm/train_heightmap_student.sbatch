#!/bin/bash
#SBATCH --job-name=rpl_hm_student
#SBATCH --partition=sairpro
#SBATCH --qos=training
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch_sair/shaoshus/locomotion/RPL_heightmap_student/logs/slurm/%x_%j.out

# Distill the height-map student (exp:g1-29dof-rpl-heightmap-distill) on the SLURM cluster, IsaacGym in the hsgym
# env built by setup_env.sbatch. H200 works because the student needs no depth rendering.
#   SMOKE=1 sbatch --time=00:40:00 --qos=normal rpl_training/training_runs/slurm/train_heightmap_student.sbatch
#   sbatch --gres=gpu:h200:4 --export=ALL,NGPUS=4 rpl_training/training_runs/slurm/train_heightmap_student.sbatch
# Extra train_agent arguments: EXTRA_ARGS="--training.num-envs 4096 ...". Logs stay local (logger:disabled).
set -euo pipefail
REPO=/scratch_sair/shaoshus/locomotion/RPL_heightmap_student
NGPUS="${NGPUS:-1}"
SMOKE="${SMOKE:-0}"
RUN_NAME="${RUN_NAME:-heightmap_student}"
cd "$REPO"

# IsaacGym's gymtorch is JIT-compiled with the node's GCC 13 and needs GLIBCXX_3.4.32, newer than the conda
# libstdc++ that python/torch find through RPATH; preloading the system one is the fix that works on this cluster.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_${SLURM_JOB_ID}"
ulimit -s unlimited
export LOGGER=disabled NGPUS
source rpl_training/training_runs/setup_env.sh
rpl_launcher_init
rpl_activate_sim

# Prove a CUDA context can be created before IsaacGym spends minutes starting up.
python -c "import torch; [torch.zeros(1, device=f'cuda:{i}') for i in range(torch.cuda.device_count())]; print('cuda ok:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" \
    || { echo CUDA_PROBE_FAILED; exit 75; }
echo "node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES ngpus=$NGPUS smoke=$SMOKE commit=$(git rev-parse --short HEAD)"

ARGS=(exp:g1-29dof-rpl-heightmap-distill --training.headless True logger:disabled --training.name "$RUN_NAME")
if [[ "$SMOKE" == 1 ]]; then
    ARGS+=(--training.num-envs 64 --algo.config.num-learning-iterations 3 --algo.config.save-interval 1)
fi
rpl_training_command
# shellcheck disable=SC2086  # EXTRA_ARGS is a list of CLI arguments
"${TRAIN_COMMAND[@]}" -m rpl_training.train_agent "${ARGS[@]}" ${EXTRA_ARGS:-}
echo TRAIN_DONE
