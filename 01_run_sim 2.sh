#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=1:00:00
#SBATCH --job-name test_01
#SBATCH --output=$SCRATCH/test_01_%j.txt

cd $SLURM_SUBMIT_DIR

module load anaconda3/5.2.0
#source activate python3_MADE

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python $HOME/_testing/Parallel/test_02.py
