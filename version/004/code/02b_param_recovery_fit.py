#---------------------#
# PARAMTER RECOVERY   #
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
import modules.utils_addm_004 as utils_addm     # for importing custom module
from itertools import repeat                    # for additional values in map function

"""Fit subjects to RT dist (MLE)

"""

#-------------#
# LOAD FILES  #
#-------------#
# RT dist
path_to_rt_dist = "MADE/version/004/output/2018_09_28_0006/rt_dist_combined.pickle"
rt_dist = pickle.load(open(path_to_rt_dist, "rb"))
print("loaded rt dists...")
# Sim Subjects
path_to_sub_sims = "MADE/version/004/output/2018_11_01_1513/rt_dist_param_recovery.pickle"
sub_sims = pickle.load(open(path_to_sub_sims, "rb"))
print("loaded sim subjects...")

#------------------#
# PREP VARIABLES   #
#------------------#

# Get rid of zeros
rt_dist = utils_addm.zero_to_eps(rt_dist)
print("replaced zeros with epsilon...")

# create iterator variable
subjects = range(len(sub_sims))


#--------------------#
# FITTING FUNCTIONS  #
#--------------------#

def nll(x):
    x = -2 * np.log(x)
    return (x)


def fit_subject(i):
    print(f'Process {os.getpid()} working on subject {i}')
    # for i in range(len(sub_sims)):
    for subject in sub_sims[i]:      # single subject
        # init dataframe for storing fits (maybe better with dict?? Can this be stored in tuple??)
        fit_df = pd.DataFrame(index=range(len(rt_dist)), columns=[
                              'drift', 'boundary', 'theta', 'sp_bias', 'NLL'])
        fit_df = fit_df.fillna(0)  # with 0s rather than NaNs
        p_count = 0   # for inserting the NLL in correct spot in dataframe

        for j in range(len(rt_dist)):

            for params in rt_dist[j]:    # test on each parameter combination
                nll_fit = 0                          # initialize the NLL to zero to start each param combo test

                # walk through the response (choice/RT) for each value combo
                for value in sub_sims[i][subject]:
                    # check fit between subject response and simulated paramter response rates...
                    nll_fit = nll_fit + \
                        np.nansum((sub_sims[i][subject][value][:, 1:3] * 4)
                                  * nll(rt_dist[j][params][value][:, 1:3]))

                # pulling out parameters from string and converting to numeric
                drift, boundary, theta, sp_bias = list(map(float, params.split('_')))
                # add to df
                fit_df.iloc[p_count] = drift, boundary, theta, sp_bias, nll_fit
                # augment counter to add to next row
                p_count += 1
        # sort df
        fit_df = fit_df.sort_values(by=['NLL'])
        # if you don't need to see all the fits you could uncomment the code below
        # fit_df = np.sum((fit_df.iloc[:20,0:4].T * weighted_fit_points).T) / np.sum(weighted_fit_points)

    return(fit_df)

#------------------#
# FIT IN PARALLEL  #
#------------------#


print('starting fitting...')
with concurrent.futures.ProcessPoolExecutor() as executor:
    NLL = tuple(executor.map(fit_subject, subjects))

# Save RT dist file
utils_addm.pickle_save(path_to_sub_sims, "p_recov_fits_full" + ".pickle", NLL)

#----------------------------#
# CALCULATE BEST PARAMETERS  #
#----------------------------#
print('calculating best fits...')
with concurrent.futures.ProcessPoolExecutor() as executor:
    subject_fits = tuple(executor.map(utils_addm.calc_best_fit_ps, NLL))

# Save RT dist file
utils_addm.pickle_save(path_to_sub_sims, "p_recov_fits_best" + ".pickle", subject_fits)
