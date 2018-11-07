#!/bin/bash

version="007"

jobscript="01a_sim"

# change the second number in the loop to determine the number of nodes
for i in {1..3}
do
    node_num=$i
    echo "Starting node $i..."

    sbatch  --job-name=$version.$step.$node_num \
            --output=/scratch/c/chutcher/wilsodj/MADE/$version.$step.$node_num.%j.txt \
            --export=node_num=$node_num,version=$version,jobscript=$jobscript \
            MADE/scripts/jobscript.sh
done
