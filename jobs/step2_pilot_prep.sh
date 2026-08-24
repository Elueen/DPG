#!/bin/bash --login
#SBATCH --job-name=dpg_pilot_prepucc_ci
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-1
#SBATCH --output=slurm/logs/%j_ucc.out
#SBATCH --error=slurm/logs/%j_ucc.err
mkdir -p jobs/logs
module load apps/binapps/anaconda3/2023.09
source activate dpg
cd /net/scratch/j90633xl/DPG
python experiments/step2/run_pilot_prep.py