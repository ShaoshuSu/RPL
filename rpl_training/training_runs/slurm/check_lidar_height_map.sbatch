#!/bin/bash
#SBATCH --job-name=rpl_hm_check
#SBATCH --partition=sairpro
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch_sair/shaoshus/locomotion/RPL_heightmap_student/logs/slurm/%x_%j.out

# Runs rpl_training/training_runs/check_lidar_height_map.py (same env setup as train_heightmap_student.sbatch).
set -euo pipefail
cd /scratch_sair/shaoshus/locomotion/RPL_heightmap_student
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_${SLURM_JOB_ID}"
ulimit -s unlimited
export LOGGER=disabled
source rpl_training/training_runs/setup_env.sh
rpl_launcher_init
rpl_activate_sim
python -m rpl_training.training_runs.check_lidar_height_map --training.num-envs 256 --training.headless True
echo CHECK_DONE
