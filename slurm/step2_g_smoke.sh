#!/bin/bash --login
#SBATCH --job-name=g_smoke
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-1
#SBATCH --output=slurm/logs/%j_interactions.out
#SBATCH --error=slurm/logs/%j_interactions.err
module load apps/binapps/anaconda3/2023.09
source activate dpg
cd /net/scratch/j90633xl/DPG
python experiments/step2/run_g_smoke.py