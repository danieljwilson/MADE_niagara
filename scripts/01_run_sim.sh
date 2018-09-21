#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=1:00:00
#SBATCH --job-name MADE_001
#SBATCH --output=$SCRATCH/01_MADE/output/001_%j.txt

cd $SLURM_SUBMIT_DIR

module load anaconda3/5.2.0
#source activate python3_MADE

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python $SCRATCH/01_MADE/scripts/01_run_sim.py
