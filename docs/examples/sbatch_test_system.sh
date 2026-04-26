#!/bin/bash
#SBATCH --job-name=as_test
#SBATCH --account=YOUR_ACCOUNT          # CHANGE THIS: your PI's account (e.g., gts-mypi)
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=as_test_%j.out
#SBATCH --error=as_test_%j.err

# ============================================================================
# Applied Scientist System Test
# ============================================================================
# This script runs --test-run to validate the full pipeline
# Submit from the applied-scientist directory:
#   sbatch docs/examples/sbatch_test_system.sh
#
# BEFORE RUNNING: Update the following placeholders:
#   - YOUR_ACCOUNT: your SLURM account
#   - YOUR_ENV: your conda environment name
#   - Set ANTHROPIC_API_KEY in your environment
# ============================================================================

set -e

echo "=== Applied Scientist Test Run ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# --- Environment Setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_env                  # CHANGE THIS: your conda environment name

# Verify environment
echo "=== Environment Check ==="
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python --version
echo ""

# Check GPU
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Check API key (don't print the actual key)
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    echo "Set it before submitting: export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi
echo "ANTHROPIC_API_KEY: [set]"
echo ""

# Check network connectivity to Anthropic API
echo "=== Network Check ==="
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://api.anthropic.com/v1/messages 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "401" ]; then
    echo "Anthropic API: reachable (HTTP 401 = auth required, network OK)"
elif [ "$HTTP_CODE" = "000" ]; then
    echo "WARNING: Cannot reach Anthropic API (timeout or blocked)"
    echo "LLM calls will fail. Check firewall/proxy settings."
else
    echo "Anthropic API: HTTP $HTTP_CODE"
fi
echo ""

# --- Run Test ---
# Run from the applied-scientist directory
echo "=== Running Applied Scientist Test ==="
python -m applied_scientist \
    --config configs/examples/soccer_twos.yaml \
    --task tasks/examples/soccer_twos \
    --test-run

echo ""
echo "=== Test Complete ==="
echo "End: $(date)"
