import sys
import os
import collections
import time
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
import utils_addm_02 as utils_addm        # for importing custom module


#-------------#
# DATA FILES  #
#-------------#
# Trial Data
expdata_file_name = "_testing/Parallel/data/made_v2/expdata.csv"
# Fixation Data
fixations_file_name = "_testing/Parallel/data/made_v2/fixations.csv"


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
    rtDist = tuple(executor.map(rt_dist_sim, parameter_combos))

end = time.time()

print(f'\nTime to completion: {end-start:.2f}\n')
