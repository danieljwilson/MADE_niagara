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
sys.path.append('/scratch/c/chutcher/wilsodj/MADE/')
import modules.utils_addm_006 as utils_addm     # for importing custom module

#-----------------#
# OUTPUT PATH     #
#-----------------#
# Version
version_num = str(sys.argv[1]) + "/"
# Which node
node_num = str(sys.argv[2])

# Date
now = datetime.datetime.now().strftime('%Y_%m_%d_%H_')
print(version_num + now)

# Path
path = "MADE/" + "version/" + version_num + "output/" + now + "_" + str(sys.argv[3]) + "/"
# Make directory
os.makedirs(path, exist_ok=True)

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
# To save search space later
parameter_search_space = {"drift_weight": drift_weight,
                          "upper_boundary": upper_boundary,
                          "theta": theta,
                          "sp_bias": sp_bias}


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
    nonDec=0.3,
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
    rtDist = utils_addm.simul_addm_rt_dist(parameters, input_vals, dwell_array, 'percent')

    return rtDist


#---------------------------------------#
# RUN MAP FUNCTION WITH MULTIPROCESSING #
#---------------------------------------#
start = time.time()

with concurrent.futures.ProcessPoolExecutor() as executor:
    rt_dist = tuple(executor.map(rt_dist_sim, parameter_combos))

end = time.time()
run_duration = end - start
run_duration = datetime.timedelta(seconds=run_duration)

#-----------------#
# SAVE FILES      #
#-----------------#

# Save RT dist file & fix count
utils_addm.pickle_save(path, "rt_dist_" + node_num + ".pickle", rt_dist)

# Save parameters
utils_addm.pickle_save(path, "parameters.pickle", parameter_search_space)
utils_addm.pickle_save(path, "input_vals.pickle", input_vals)

# Save Run time
f = open(path + "runtime_" + node_num + ".txt", 'w')
f.write("Run time: " + str(run_duration))
f.write("\n\nSimulations per value: " + str(sims_per_val))
f.write("\nNum sims: " + str(num_sims))
f.write("\nTotal number of simulations: " +
        str(parameter_combos.shape[0] * values_array_addm.shape[1]))
f.close()
