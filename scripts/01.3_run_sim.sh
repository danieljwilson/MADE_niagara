#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=11:00:00
#SBATCH --job-name MADE_005_1.3
#SBATCH --output=/scratch/c/chutcher/wilsodj/MADE/005_1.3_%j.txt
#SBATCH --mail-type FAIL
#SBATCH --mail-user daniel.j.wilson@gmail.com

cd $SLURM_SUBMIT_DIR

module load anaconda3/5.2.0
#source activate python3_MADE

# SPECIFY VERSION
# This is input in command line at the end in single quotes (e.g. '003')
VERSION=$1

python $SCRATCH/MADE/version/$VERSION/code/01.3_sim.py
