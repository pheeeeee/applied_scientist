#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --account=YOUR_ACCOUNT          # CHANGE THIS: your PI's account (e.g., gts-mypi)
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=test_env-%j.out

# ============================================================================
# ENVIRONMENT TEST SCRIPT
# ============================================================================
# Run this FIRST to verify your environment is correctly configured.
# Submit with: sbatch docs/examples/sbatch_test_env.sh
# Check output: cat test_env-<jobid>.out
#
# BEFORE RUNNING: Update the following placeholders:
#   - YOUR_ACCOUNT: your SLURM account
#   - YOUR_ENV: your conda environment name
#   - /path/to/applied-scientist: path to your applied-scientist installation
# ============================================================================

echo "=========================================="
echo "ENVIRONMENT TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

# --- Step 1: Test conda activation ---
echo "[1/6] Testing conda activation..."
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
# OR: source ~/miniconda3/etc/profile.d/conda.sh

conda activate your_env                  # CHANGE THIS: your conda environment name
if [ $? -eq 0 ]; then
    echo "  OK: Conda environment activated"
    echo "  Python: $(which python)"
else
    echo "  FAIL: Could not activate conda environment"
    exit 1
fi

# --- Step 2: Test GPU access ---
echo ""
echo "[2/6] Testing GPU access..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ $? -eq 0 ]; then
    echo "  OK: GPU accessible"
else
    echo "  FAIL: Cannot access GPU"
    exit 1
fi

# --- Step 3: Test PYTHONPATH ---
echo ""
echo "[3/6] Testing PYTHONPATH setup..."
export PYTHONPATH=/path/to/applied-scientist:${PYTHONPATH:-}  # CHANGE THIS
echo "  PYTHONPATH: $PYTHONPATH"

# --- Step 4: Test applied_scientist import ---
echo ""
echo "[4/6] Testing applied_scientist import..."
python -c "from applied_scientist.task.base import TaskAdapter; print('  OK: applied_scientist importable')" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: Cannot import applied_scientist"
    exit 1
fi

# --- Step 5: Test task adapter import ---
echo ""
echo "[5/6] Testing task adapter import..."
python -c "
from tasks.examples.soccer_twos.adapter import SoccerTwosAdapter
adapter = SoccerTwosAdapter()
print(f'  OK: Task adapter loaded')
print(f'  Task name: {adapter.name}')
print(f'  Metric: {adapter.metric}')
" 2>&1
if [ $? -ne 0 ]; then
    echo "  FAIL: Cannot import task adapter"
    exit 1
fi

# --- Step 6: Test Ray/PyTorch ---
echo ""
echo "[6/6] Testing Ray and PyTorch..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')

import ray
print(f'  Ray: {ray.__version__}')
" 2>&1

echo ""
echo "=========================================="
echo "ALL TESTS PASSED"
echo "=========================================="
echo ""
echo "Your environment is correctly configured for Applied Scientist."
echo "You can now run the full system with:"
echo "  python -m applied_scientist --config configs/examples/soccer_twos.yaml --task tasks/examples/soccer_twos"
