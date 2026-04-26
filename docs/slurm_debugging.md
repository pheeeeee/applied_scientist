# SLURM Debugging Guide

This guide covers common SLURM job failures when running Applied Scientist on HPC clusters like PACE Phoenix.

## Quick Diagnosis Checklist

When a job fails, check these in order:

1. **Find the log file**: `ls -la workspace/results/logs/<experiment>/slurm-*.out`
2. **Check job status**: `sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed`
3. **View recent failures**: `sacct -u $USER --state=FAILED -S $(date -d '1 day ago' +%Y-%m-%d)`

## Common Errors and Solutions

### 1. "ModuleNotFoundError: No module named 'applied_scientist'"

**Cause**: PYTHONPATH doesn't include the applied_scientist package on compute nodes.

**Solution**: Add to your config's `setup_commands`:
```yaml
compute:
  slurm:
    setup_commands:
      - "source ~/miniconda3/etc/profile.d/conda.sh"
      - "conda activate your_env"
      - "export PYTHONPATH=/path/to/applied-scientist:${PYTHONPATH:-}"
```

### 2. "sbatch: error: Batch job submission failed: Invalid account"

**Cause**: Missing or incorrect `--account` parameter (required on PACE).

**Solution**: Add account to your config:
```yaml
compute:
  slurm:
    account: YOUR_ACCOUNT  # e.g., gts-mypi
    partition: gpu-v100
```

### 3. Job stuck in PENDING state

**Causes**:
- Queue is full
- Requested resources unavailable
- Account out of SUs

**Diagnosis**:
```bash
squeue -u $USER                    # See your pending jobs
squeue -p gpu-v100 --state=RUNNING # See what's running on partition
pace-quota                          # Check SU balance (PACE-specific)
```

### 4. "ImportError: libcudart.so.X.X: cannot open shared object file"

**Cause**: CUDA not loaded or version mismatch.

**Solution**: Load CUDA module in setup_commands:
```yaml
setup_commands:
  - "module load cuda/11.7"
  - "source ~/miniconda3/etc/profile.d/conda.sh"
  - "conda activate your_env"
```

### 5. Job FAILED with exit code 1, no useful output

**Cause**: Script failed before producing output (often environment setup).

**Debug steps**:
1. Check the generated sbatch script in `workspace/.scripts/`
2. Run the script manually on an interactive node:
   ```bash
   salloc --nodes=1 --gres=gpu:1 --mem=16G --time=00:30:00 -A gts-yourPI
   bash workspace/.scripts/<experiment>.sh
   ```
3. Watch for errors in conda activation, module loading, or imports

### 6. "CUDA out of memory"

**Cause**: Model or batch size too large for GPU memory.

**Solutions**:
- Reduce `train_batch_size` in spec
- Request larger GPU: `gres: "gpu:V100:1"` → `gres: "gpu:A100:1"`
- Check if multiple processes are sharing the GPU

### 7. Job TIMEOUT

**Cause**: Training exceeded walltime limit.

**Solutions**:
- Increase `time` in config: `time: "04:00:00"`
- Reduce `time_budget` in system config
- Enable early stopping: `early_stop_fraction: 0.4`

### 8. "ConnectionError" or "APIError" from LLM calls

**Cause**: Compute nodes can't reach external APIs.

**Diagnosis**:
```bash
# On compute node:
curl -s https://api.anthropic.com/v1/messages -o /dev/null -w "%{http_code}"
# 401 = reachable (auth required), timeout = firewalled
```

**Solution**: If firewalled, you may need to:
- Use a proxy (check with HPC admins)
- Run the orchestrator on a login node (agents make API calls, not training jobs)

## PACE Phoenix Specific

### Recommended Config for PACE

```yaml
compute:
  backend: slurm
  slurm:
    account: gts-yourPI           # REQUIRED on PACE
    partition: gpu-v100           # or gpu-a100
    gres: "gpu:V100:1"            # explicit GPU type
    mem: "48G"
    time: "02:00:00"
    setup_commands:
      # Use PACE anaconda OR your own miniconda, not both
      - "source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"
      # OR for personal miniconda:
      # - "source ~/miniconda3/etc/profile.d/conda.sh"
      - "conda activate your_env"
      - "export PYTHONPATH=/path/to/applied-scientist:${PYTHONPATH:-}"
```

### Checking GPU Availability

```bash
# See available GPU partitions
sinfo -p gpu-v100,gpu-a100 -N -l

# See your recent job history
sacct -u $USER -S $(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Debugging Workflow

### Step 1: Reproduce Locally

Before submitting to SLURM, test on an interactive node:

```bash
# Get interactive GPU session
salloc --nodes=1 --ntasks=4 --gres=gpu:1 --mem=32G --time=01:00:00 -A gts-yourPI

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate soccertwos

# Test imports
python -c "from applied_scientist.task.base import TaskAdapter; print('OK')"
python -c "import ray; print(ray.__version__)"

# Test a short training run
python -m applied_scientist.run_experiment \
  --spec workspace/configs/experiments/ppo_baseline.yaml \
  --task-module tasks/examples/soccer_twos \
  --seed 1 \
  --time-budget 60 \
  --log-path /tmp/test_log \
  --checkpoint-dir /tmp/test_ckpt
```

### Step 2: Check Generated Scripts

Applied Scientist saves sbatch scripts to `workspace/.scripts/` (if enabled).
Review them to verify:
- Correct conda environment
- PYTHONPATH includes applied_scientist
- Correct paths to spec files

### Step 3: Enable Verbose Logging

Add to your training script or adapter:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Step 4: Check System Events

Applied Scientist logs all job submissions and failures:
```bash
cat workspace/results/system_events.jsonl | grep -i fail
cat workspace/results/system_events.jsonl | grep -i error
```

## File Locations

| File | Purpose |
|------|---------|
| `workspace/results/logs/<exp>/slurm-<jobid>.out` | Job stdout/stderr |
| `workspace/results/logs/<exp>/progress.json` | Training progress (for early stopping) |
| `workspace/.scripts/<exp>.sh` | Generated sbatch script |
| `workspace/.state/gpu_slots.yaml` | Current GPU slot assignments |
| `workspace/results/system_events.jsonl` | System event log |

## Getting Help

1. Check this guide first
2. Review `system_events.jsonl` for error context
3. Test on interactive node to reproduce
4. Check PACE documentation: https://docs.pace.gatech.edu/
