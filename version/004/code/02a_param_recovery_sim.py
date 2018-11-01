#---------------------#
# PARAMETER RECOVERY  #
#---------------------#

import sys
import os
import collections
import time
import datetime
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
import pickle
import copy
sys.path.append('/scratch/c/chutcher/wilsodj/MADE/')
import modules.utils_addm_004 as utils_addm        # for importing custom module

"""Simulate Subjects

    1. Create values array for each subject (same number as experimental trials)
    2. Simulate trial(s) for each value
"""


#-----------------#
# OUTPUT PATH     #
#-----------------#
# Version
version_num = "004/"
# Which node
script_id = "param_recovery"

# Date
now = datetime.datetime.now().strftime('%Y_%m_%d_%H%M/')
print(version_num + now)

# Path
path = "MADE/" + "version/" + version_num + "output/" + now
progress_path = path + "progress/"
# Make directory
if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(progress_path)

#-------------#
# DATA FILES  #
#-------------#
# Trial Data
expdata_file_name = "MADE/data/made_v2/expdata.csv"
# Fixation Data
fixations_file_name = "MADE/data/made_v2/fixations.csv"

#------------------#
# aDDM PARAMETERS  #
#------------------#
# SCALING VALUE FOR DRIFT
drift_weight = np.linspace(0.005, 0.15, 16)
# BOUNDARY VALUE
upper_boundary = np.linspace(0.05, 0.35, 16)
# THETA VALUE
theta = np.linspace(0.1, 1, 12)
# START BIAS VALUE
sp_bias = np.linspace(-0.3, 0.3, 7)

parameter_combos = utils_addm.parameter_values(drift_weight, upper_boundary, theta, sp_bias)


#----------------#
# VALUES ARRAY   #
#----------------#
# How many trials did subjects complete?
number_of_trials = 300

# How many unique drift values (note that in this case these are binned as it was essentially continuous)
num_vals = len(utils_addm.values_for_simulation_addm(expdata_file_name, sims_per_val=1)[0])

# Calculate how many times to simulate each value to approximate the number of trials subjects completed
sims_per_val = np.ceil(number_of_trials / num_vals)

# Create values to simulate
values_array_addm = utils_addm.values_for_simulation_addm(expdata_file_name, sims_per_val)

# Total number of simulations
num_sims = values_array_addm.shape[1]


input_vals = utils_addm.Input_Values(
    parameter_combos=parameter_combos,
    values_array=values_array_addm,
    num_sims=num_sims,
    startVar=0,
    nonDec=0.8,   # change to add time based on each fixation
    # calculate based on difference between exp 1 & 2 RTS.
    nonDecVar=0,
    driftVar=0,
    maxRT=10,
    precision=0.001,        # ms precision
    s=0.1,                  # scaling factor (Ratcliff) (check if variance or SD)
)

#-----------------#
# FUNCTION TO MAP #
#-----------------#


def rt_dist_sim(parameters):
    # NEXT: dwell_array would be by individual
    dwell_array = utils_addm.create_dwell_array(num_sims, fixations_file_name, data='sim')
    # Run simulations
    rtDist = utils_addm.simul_addm_rt_dist(parameters, input_vals, dwell_array, 'count')

    # Return RT distribution
    return rtDist



#-------------------------------------------------#
# SIMULATE: RUN MAP FUNCTION WITH MULTIPROCESSING #
#-------------------------------------------------#
start = time.time()

with concurrent.futures.ProcessPoolExecutor() as executor:
    rt_dist = tuple(executor.map(rt_dist_sim, parameter_combos))

end = time.time()
run_duration = end - start
run_duration = datetime.timedelta(seconds=run_duration)

#-----------------#
# SAVE FILES      #
#-----------------#

# Save RT dist file
utils_addm.pickle_save(path, "rt_dist_" + script_id + ".pickle", rt_dist)

# Save Run time
f = open(path + "runtime_" + script_id + ".txt", 'w')
f.write("Run time: " + str(run_duration))
f.write("\n\nSimulations per value: " + str(sims_per_val))
f.write("\nNum sims: " + str(num_sims))
f.write("\nTotal number of simulations: " +
        str(parameter_combos.shape[0] * values_array_addm.shape[1]))
f.close()
