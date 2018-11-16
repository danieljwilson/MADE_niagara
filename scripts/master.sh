#!/bin/bash

#-----------#
# SPECIFY   #
#-----------#

version="008"
jobscript="01a_sim"
# v1 or v2
exp_ver="v1"
# list of subjects being used
subjects=(1 8 9 15 18 29 31)


#-----------#
# LAUNCH    #
#-----------#

for subject in "${subjects[@]}"; do
    echo "Submitting subject $subject..."
    # change the second number in the loop to determine the number of nodes
    for i in {1..2}
    do
        node_num=$i
        echo "Launching node $i..."

        sbatch  --job-name=$version.$subject.$node_num \
                --output=/scratch/c/chutcher/wilsodj/MADE/$version.$subject.$exp_ver.$node_num.%j.txt \
                --export=node_num=$node_num,version=$version,jobscript=$jobscript,subject=$subject,exp_ver=$exp_ver \
                MADE/scripts/jobscript.sh
    done
done
