import sys
import os
import collections
import time
from datetime import datetime
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
import pickle
import utils_addm_02 as utils_addm        # for importing custom module


#-------------#
# DATA FILES  #
#-------------#
# Trial Data
expdata_file_name = "01_MADE/data/made_v2/expdata.csv"
# Fixation Data
fixations_file_name = "01_MADE/data/made_v2/fixations.csv"

#-----------------#
# OUTPUT PATH     #
#-----------------#
# Date
now = datetime.now().strftime('%Y_%m_%d_%H%M/')
# Version
version = "002/"
# Path
path = "01_MADE/output/" + version + now
# Make directory
os.makedirs(path)

#------------------#
# aDDM PARAMETERS  #
#------------------#
# SCALING VALUE FOR DRIFT
drift_weight = np.linspace(0.005, 0.15, 8)
# BOUNDARY VALUE
upper_boundary = np.linspace(0.05, 0.35, 8)
# THETA VALUE
theta = np.linspace(0.1, 1, 6)
parameter_combos = utils_addm.parameter_values(drift_weight, upper_boundary, theta)
# To save search space later
parameter_search_space = {"drift_weight": drift_weight,
                          "upper_boundary": upper_boundary,
                          "theta": theta}

#----------------#
# VALUES ARRAY   #
#----------------#
sims_per_val = 1000
# Get left and right values from data
values_array_addm = utils_addm.values_for_simulation_addm(expdata_file_name, sims_per_val)
# Create num_sims var
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
    rtDist = utils_addm.simul_addm_rt_dist(parameters, input_vals, dwell_array)

    return rtDist


#---------------------------------------#
# RUN MAP FUNCTION WITH MULTIPROCESSING #
#---------------------------------------#
start = time.time()

with concurrent.futures.ProcessPoolExecutor() as executor:
    rt_dist = tuple(executor.map(rt_dist_sim, parameter_combos))

end = time.time()
run_duration = end - start
print(run_duration)
#-----------------#
# SAVE FILES      #
#-----------------#

# Save RT dist file
utils_addm.pickle_save(path, "rt_dist.pickle", rt_dist)

# Save parameters
utils_addm.pickle_save(path, "parameters.pickle", parameter_search_space)
utils_addm.pickle_save(path, "input_vals.pickle", input_vals)
# Save Run time
f = open(path + "runtime.txt", 'w')
f.write("Run time = " + str(run_duration))
f.write("You ran {0} simulations for each value. Num sims variable = {1}. Maybe total number of simulations: {2}.".format(
    sims_per_val, num_sims, values_array.shape[0]))
f.close()
