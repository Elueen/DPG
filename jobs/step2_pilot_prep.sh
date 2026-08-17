#!/bin/bash --login
#SBATCH --job-name=dpg_pilot_prep
#SBATCH --output=jobs/logs/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

mkdir -p jobs/logs
conda activate dpg
cd /net/scratch/j90633xl/DPG
python experiments/step2/run_pilot_prep.py