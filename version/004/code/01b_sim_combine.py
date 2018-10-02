import sys
import os
import pickle
import utils_addm_02 as utils_addm        # for importing custom module


#-----------------#
# COMBINE DICTS   #
#-----------------#

# Running from $SCRATCH server
rt_path = "/gpfs/fs0/scratch/c/chutcher/wilsodj/01_MADE/version/004/output/2018_09_28_0006/rt_dist_"

# List of the simulated distributions, e.g. ['01', '02', '03']
rt_dists = ['01', '02', '03', '04']

# Combine RT distribution files
combined_rt_dist = utils_addm.rt_dists_mean(rt_path, rt_dists)

# Save combined RT distribution file
utils_addm.pickle_save(rt_path, "combined.pickle", combined_rt_dist)
