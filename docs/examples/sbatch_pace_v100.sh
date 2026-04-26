#!/bin/bash
#SBATCH --job-name=ppo_baseline
#SBATCH --account=YOUR_ACCOUNT          # CHANGE THIS: your PI's account (e.g., gts-mypi)
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# ============================================================================
# EXAMPLE SBATCH SCRIPT FOR PACE PHOENIX (V100 GPU)
# ============================================================================
# This is a reference script showing the structure Applied Scientist generates.
# Modify account, partition, and paths for your setup.
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# --- Environment Setup ---
# Option 1: Use PACE shared anaconda
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh

# Option 2: Use personal miniconda (uncomment if preferred)
# source ~/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate your_env                  # CHANGE THIS: your conda environment name

# Add applied_scientist to PYTHONPATH (required if not pip installed)
export PYTHONPATH=/path/to/applied-scientist:${PYTHONPATH:-}  # CHANGE THIS

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Run Training ---
# CHANGE THESE PATHS to match your setup
python -m applied_scientist.run_experiment \
    --spec /path/to/applied-scientist/workspace/configs/experiments/ppo_baseline.yaml \
    --task-module /path/to/applied-scientist/tasks/examples/soccer_twos \
    --seed 1 \
    --time-budget 1800 \
    --log-path /path/to/applied-scientist/workspace/results/logs/ppo_baseline_s1 \
    --checkpoint-dir /path/to/applied-scientist/workspace/results/checkpoints/ppo_baseline_s1

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
