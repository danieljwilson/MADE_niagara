#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=11:00:00
#SBATCH --job-name MADE_005_1.1
#SBATCH --output=/scratch/c/chutcher/wilsodj/MADE/005_1.1_%j.txt
#SBATCH --mail-type FAIL
#SBATCH --mail-user daniel.j.wilson@gmail.com

cd $SLURM_SUBMIT_DIR

module load anaconda3/5.2.0
#source activate python3_MADE

# version number, jobscript name, and node number should be passed in from master.sh

python $SCRATCH/MADE/version/$version/code/$jobscript.py "$version" "$node_num" "$jobscript"
