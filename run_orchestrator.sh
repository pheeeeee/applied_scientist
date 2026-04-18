#!/bin/bash
#SBATCH --job-name=applied-scientist
#SBATCH --partition=cpu-small
#SBATCH --account=gts-jli3175
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=orchestrator-%j.out

source ~/.secrets/autoresearch.env
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate applied-scientist

cd /storage/project/r-jli3175-0/jphee3/autoresearch/applied-scientist
srun python -m applied_scientist --config configs/default.yaml --task tasks/examples/soccer_twos --slack
