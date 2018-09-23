#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=11:00:00
#SBATCH --job-name MADE_003
#SBATCH --output=/scratch/c/chutcher/wilsodj/01_MADE/output/003_%j.txt
#SBATCH --mail-type FAIL
#SBATCH --mail-user daniel.j.wilson@gmail.com

cd $SLURM_SUBMIT_DIR

module load anaconda3/5.2.0
#source activate python3_MADE

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python $SCRATCH/01_MADE/scripts/04_run_sim.py
