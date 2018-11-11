#!/bin/bash

version="008"

jobscript="01a_sim"

subject="15"

# change the second number in the loop to determine the number of nodes
for i in {1..2}
do
    node_num=$i
    echo "Starting node $i..."

    sbatch  --job-name=$version.$step.$node_num \
            --output=/scratch/c/chutcher/wilsodj/MADE/$version.$step.$node_num.%j.txt \
            --export=node_num=$node_num,version=$version,jobscript=$jobscript,subject=$subject \
            MADE/scripts/jobscript.sh
done
