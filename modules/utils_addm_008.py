#!/usr/bin/python

"""
util.py
Author: Daniel J Wilson, daniel.j.wilson@gmail.com

Utility functions for aDDM.
"""

import numpy as np
import pandas as pd
from typing import List                     # for writing function defs
import datetime as dt                       # for naming files
# for creating values list (for aDDM of all stimuli value combos)
import itertools
import os
import time                                 # for tracking timing of loops
import shelve
# to overwrite the avg. loop time/remining time with each loop
from IPython.display import clear_output
import pickle                               # for saving to pickle
import copy                                 # for deepcopy
import glob
# from tqdm import tqdm                       # for keeping track of progress
import collections
import scipy.stats
from itertools import repeat                    # for additional values in map function

import numba as nb
from numba import jit

Input_Values = collections.namedtuple('Input_Values', [
    'parameter_combos',
    'values_array',
    'num_sims',
    'max_fix',
    'startVar',
    'nonDec',
    'nonDecVar',
    'driftVar',
    'maxRT',
    'precision',
    's',
])


def values_for_simulation(values: List[float], sims_per_val: int) -> List[float]:
    """Return an array that repeats values by sims_per_val
    These will then be combined with each of the parameter combos.
    i.e. Each parameter combination will be run with all values for the number of simulations requested.

    Args:
        values: list, experimental values you wish to use.
        sims_per_val: int, number of simulations.
    Returns:
        values_array: numpy.array, all values repeated defined number of times.

    >>> values_for_simulation([0, 0.5, 5], 3)
    array([[ 0. ],
           [ 0. ],
           [ 0. ],
           [ 0.5],
           [ 0.5],
           [ 0.5],
           [ 5. ],
           [ 5. ],
           [ 5. ]])
    """

    values_array = np.repeat(values, sims_per_val)  # duplicate values * simsPerVal
    values_array = values_array.reshape(len(values_array), 1)  # make 2D array

    # PRINT SUMMARY
    print("You will run {0} simulations for each of {1} values for a total of {2} simulations.".format(
        sims_per_val, len(values), values_array.shape[0]))

    return values_array


def values_for_simulation_addm(expdata_file_name, sims_per_val):
    """Return an array that repeats values by sims_per_val
    These will then be combined with each of the parameter combos.
    i.e. Each parameter combination will be run with all values for the number of simulations requested.

    Args:
        expdata_file_name: str, name of the data file for the experiment.
        sims_per_val: int, number of simulations.
    Returns:
        values_array: numpy.array, all values repeated by number sims_per_val.

    """

    df = pd.read_csv(expdata_file_name, header=0, sep=",", index_col=None)

    item_left_val = np.unique(df.item_left_val)
    item_right_val = np.unique(df.item_right_val)

    values = list(itertools.product(item_left_val, item_right_val))
    values = np.around(values, decimals=2)   # currently produces 81 combos
    # transpose to take advantage of row major format of Python
    t_vals = np.transpose(values)

    values_array = np.repeat(t_vals, sims_per_val, axis=1)  # duplicate values * simsPerVal

    # PRINT SUMMARY
    print("You will run {0} simulations for each of {1} value combinations for a total of {2} simulations.".format(
        sims_per_val, len(values), values_array.shape[1]))

    return values_array


def parameter_values(drift_weight, upper_boundary, theta, sp_bias):
    """Return an array that contains all combinations of the parameters drift_weight and upper_boundary.

    Args:
        drift_weight: list, all values to try for drift scaling parameter
        upper_boundary: list, all values to try for boundary
        theta: list, weighting values of unattended attribute
    Returns:
        parameter_combos: numpy.array, all combinations of parameters to test

    >>> parameter_values([0,2], [1,3])
    Your parameter search space is size: (4, 2).

    # this is not shown but it is what the array looks like
    array([[ 0.,  1.],
           [ 0.,  3.],
           [ 2.,  1.],
           [ 2.,  3.]])
    """

    parameter_combos = np.array(list(itertools.product(
        drift_weight, upper_boundary, theta, sp_bias)))

    # CHECK SIZE OF ARRAY
    print("Your parameter search space is size: {0}.".format(parameter_combos.shape))

    return parameter_combos


def total_sim_count(parameter_combos, values_array):
    """Just provides a trivial calculation of the total number of simulations you will be running.

    Args:
        parameter_combos: numpy.array, all parameter combinations.
    Returns:
        Nothing
    """

    print("Loop # (parameter variations) = ", len(parameter_combos))
    print("Sim # (per loop) = ", values_array.shape[1])
    print("Total sims = ", len(parameter_combos) * values_array.shape[1])


def save_parameters(parameter_combos):
    """
    Will create a new folder with today's date (if it does not already exist) and create a csv file that saves the
    parameter combinations.

    Args:
        parameter_combos: numpy.array, all parameter combinations.
    Returns:
        Nothing
    """
    now = dt.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M")
    date = now.strftime("%Y-%m-%d")

    iterDf = pd.DataFrame(parameter_combos)           # create DF to save to CSV

    directory = ('outputs/' + str(date) + '/params')  # set Directory to save CSV

    if not os.path.exists(directory):                 # create the directory if it does not already exist
        os.makedirs(directory)

    iterDf.to_csv(directory + '/parameter_combos_' + str(time) + '.csv')


def save_sim_combo_csv(dfOut, loop_num):
    """
    """
    now = dt.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M")
    date = now.strftime("%Y-%m-%d")

    directory = ('outputs/' + str(date) + '/sims')                # set Directory to save CSV

    if not os.path.exists(directory):                   # create the directory if it does not already exist
        os.makedirs(directory)

    dfOut.to_csv(directory + '/' + str(loop_num) + '.csv')


def save_sim_subjects_csv(dfOut, loop_num, date):
    """
    """

    # passing in the date so that we don't start working across different folders
    # now = dt.datetime.now()
    # time = now.strftime("%Y-%m-%d-%H-%M")
    # date = now.strftime("%Y-%m-%d")

    directory = ('outputs/' + str(date) + '/sim_subj/')  # set Directory to save CSV

    if not os.path.exists(directory):                   # create the directory if it does not already exist
        os.makedirs(directory)

    # dfOut.to_csv(directory + params + '.csv')
    dfOut.to_csv(directory + str(loop_num) + '.csv')


def save_sim_combo_shelve(dfOut, loop_num):
    """
    """
    now = dt.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M")

    directory = ('outputs/' + str(date))                # set Directory to save CSV

    if not os.path.exists(directory):                   # create the directory if it does not already exist
        os.makedirs(directory)

    with shelve.open(directory + '/all_sims', 'c') as shelf:
        # maybe name it as parameters?
        shelf[str(loop_num)] = dfOut


def create_synth_dwell_array(input_vals, first_fix_synth_dist, mid_fix_synth_dist):
    """
    Creates an array of fixations from the syntetic distributions.
    input_vals(obj): contains values used to calculate dimensions of array
    first_fix_synth_dist(np.array): simulated 1st fixations for the subject
    mid_fix_synth_dist(np.array): simulated 2nd fixations for the subject
    """
    # Create empty array
    dwell_array = np.empty([input_vals.max_fix, len(input_vals.values_array[1])], dtype=np.int16)

    # Fill in the first row (each row = 1 fixation)
    dwell_array[0, :] = np.random.choice(first_fix_synth_dist,
                                         size=len(input_vals.values_array[1]),
                                         replace=True)
    # Fill in remaining rows
    dwell_array[1:, :] = np.random.choice(mid_fix_synth_dist,
                                          size=(input_vals.max_fix - 1,
                                                len(input_vals.values_array[1])),
                                          replace=True)

    dwell_array = np.cumsum(dwell_array, axis=0)    # cumulative sum of each column
    return(dwell_array)


def create_dwell_array(num_sims, fixations, data, exp_data=None):
    """
    Pulls out first and middle fixations from fixations_file_name.
    Creates distribution of fixation times that can be used in aDDM simulation

    Args:
        fixations: df, fixations selected according to train/test parameter
        exp_data: df, experimental data selected according to train/test parameter
        num_sims: int, total number of simulations being run for each parameter combination
        data: str, are we creating fixations for simulations or are we running test data
    """
    if data == 'sim':

        df = pd.read_csv(fixations, header=0, sep=",", index_col=None)

        # first create distributions of first/mid fixations
        first_fix_dist = df['fix_time'][df['fix_num'] == 1]
        mid_fix_dist = df['fix_time'][(df['fix_num'] > 1) & (df['rev_fix_num'] > 1)]

    if data == 'test':
        df = fixations

        # first create distributions of first/mid fixations
        first_fix_dist = df['fix_time'][df['fix_num'] == 1]
        mid_fix_dist = df['fix_time'][(df['fix_num'] > 1) & (df['rev_fix_num'] > 1)]

    # make sure that none of the final columns in the dwell array has a value smaller than maxRT
    min_time = 0

    #--------------#
    # SIM          #
    #------------------------------------------------------------------------------------ #
    # NOTE: for simulations (creating the initial distributions of RTs and choices)       #
    # we are not looking at actual trial data. Just creating fixation durations from the  #
    # distributions                                                                       #
    #-------------------------------------------------------------------------------------#

    if data == 'sim':
        while min_time < 10002:   # 10,000 = maxRT...this is a hack. Just keeps creating arrays until this is satisfied.
            # create column of first fixations
            dwell_array = np.reshape(
                ((np.random.choice(first_fix_dist, num_sims, replace=True))), (num_sims, 1))

            # create additional columns from middle fixation distribution
            for column in range(20):
                append_mid_fix_array = np.reshape(
                    ((np.random.choice(mid_fix_dist, num_sims, replace=True))), (num_sims, 1))
                dwell_array = np.append(dwell_array, append_mid_fix_array, axis=1)

            # make each column the sum of itself and the previous column
            for column in range(1, np.shape(dwell_array)[1]):
                dwell_array[:, column] = dwell_array[:, column] + dwell_array[:, (column - 1)]

            min_time = min(dwell_array[:, 20])

    #--------------#
    # TEST         #
    #------------------------------------------------------------------------------------ #
    # NOTE: the difference with the 'test' version is that we are using the actual        #
    # initial fixations and then filling out the rest of the fixations by sampling the    #
    # group or subject first/mid fixation distribution                                    #
    #-------------------------------------------------------------------------------------#

    if data == 'test':

        # calculuate size based on actual number of trials (odd or even)
        size = len(exp_data.trial) * num_sims

        while min_time < 10000:   # 10,000 = maxRT
            # add data from trial fixations
            dwell_array = np.reshape(
                ((np.random.choice(first_fix_dist, size, replace=True))), (size, 1))

            # create additional columns (number of columns determined by range()) from middle fixation distribution
            for column in range(21):
                append_mid_fix_array = np.reshape(
                    ((np.random.choice(mid_fix_dist, size, replace=True))), (size, 1))
                dwell_array = np.append(dwell_array, append_mid_fix_array, axis=1)

            # overwrite initial columns with actual fixation values
            row_start = 0
            row_end = 1000  # Need to change this to the number of sims!
            column = 0

            # from IPython.core.debugger import Tracer; Tracer()()
            for i in tqdm(range(len(fixations.trial))):
                if i == 0:    # first row
                    dwell_array[row_start:row_end, column] = fixations.fix_time[i]

                elif (fixations.trial[i]) == (fixations.trial[i - 1]):
                    column += 1
                    dwell_array[row_start:row_end, column] = fixations.fix_time[i]

                else:
                    column = 0
                    row_start += 1000
                    row_end += 1000
                    dwell_array[row_start:row_end, column] = fixations.fix_time[i]

            # make each column the sum of itself and the previous column
            for column in range(1, np.shape(dwell_array)[1]):
                dwell_array[:, column] = dwell_array[:, column] + dwell_array[:, (column - 1)]

            min_time = min(dwell_array[:, 21])

    # cast to int (does this make it faster?)
    dwell_array = dwell_array.astype(int)
    t_dwells = np.transpose(dwell_array)        # this is to make things row major

    return t_dwells


def simul_ddm(parameter_combos, values_array, num_sims, loop_num,
              startVar, nonDec, nonDecVar, driftVar, maxRT, respOptions, precision, s):

    t = 0                                               # initialize time to zero

    scaling = parameter_combos[loop_num, 0]               # drift scaling on loop
    upper = parameter_combos[loop_num, 1]                 # boundary on loop
    within_bounds = np.ones((num_sims, 1), dtype=bool)   # initialize location vector for each drift
    # -.5 to mean center (starting dist. uniform)
    ddms = np.zeros((num_sims, 1)) + startVar * (np.random.rand(num_sims, 1) - .5)
    resp = np.nan * np.ones((num_sims, 1))               # initialize choices
    rt = np.nan * np.ones((num_sims, 1))

    # Where are we:
    print("Simulating for scaling {0} and boundary {1}. Loop {2}/{3}".format(scaling,
                                                                             upper, loop_num, len(parameter_combos)))

    # upper value updated each loop
    lower = -1 * upper
    # drift scaling updated each loop
    drift = (scaling * values_array) + (driftVar * np.random.randn(num_sims, 1))

    # initUpper = upper                 # for collapsing bounds
    # AUC = np.zeros((len(drift),1))    # signed added up (not essential)
    # AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)

    while (t < (maxRT / precision)) and np.count_nonzero(within_bounds):  # any paths still going

        t = t + 1

        meanDrift = drift[within_bounds]

        # Roger (variance is at .1 so might not need square root)
        dy = meanDrift * precision + (s * (np.sqrt(precision)) *
                                      np.random.randn(np.sum(within_bounds,)))

        ddms[within_bounds] = ddms[within_bounds] + dy

#       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
#       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);

#        if ReturnTraces
#        traces(:,t) = ddms(TracesToRetain);
#        break

        # select ddms that have completed but don't have response data recorded
        within_bounds[ddms >= upper] = False
        within_bounds[ddms <= lower] = False
        # identifying ddms that crossed at time t (within_bounds = False, no logged response)
        # justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
        justDone = (within_bounds == False) & (np.isnan(resp))

        # log resp and RT data
        # sign 1 if value is positive or -1 if value is negative
        resp[justDone] = np.sign(ddms[justDone])
        rt[justDone] = t

    # COLLAPSING BOUNDARIES
    # upper = initUpper * exp(-1 * collapseRate * t);
    # lower = -1 * upper;
    # break

    rt = rt * precision     # convert to seconds
    rt = rt + nonDec      # add non decision time

    dfOut = save_to_df(rt, resp, drift, scaling, upper)  # save values in DF
    return dfOut


def time_estimate(loop_num, parameter_combos, total_sim_time):
    avg_loop_time = total_sim_time / loop_num
    print("Avg loop time: {0:0.2f}".format(avg_loop_time))
    remain_est = (len(parameter_combos) * avg_loop_time) - (loop_num * avg_loop_time)
    m, s = divmod(remain_est, 60)
    h, m = divmod(m, 60)
    print("Remaining time estimate: {0}:{1}:{2}".format(int(h), int(m), int(s)))


def simul_addm_test_data(test_data, dwell_array,
                         startVar, nonDec, nonDecVar, driftVar, maxRT, precision, s):
    """

    Args:
        expdata_combined_test: df, all trials divided into dicts by subject
        dwell_array: np.array, fixation times incorporating actual subject fixations

    """

    # Loop through subjects in dict
    for subject in tqdm(test_data.keys()):

        t = 0                                                   # initialize time to zero
        # find the length of vector containing all trials
        num_sims = len(test_data[subject].trial)

        scaling = test_data[subject].loc[0, 'est_scaling']       # drift scaling on loop
        upper = test_data[subject].loc[0, 'est_boundary']
        theta = test_data[subject].loc[0, 'est_theta']           # boundary on loop
        # initialize location vector for each drift
        within_bounds = np.ones((num_sims), dtype=bool)
        # -.5 to mean center (starting dist. uniform)
        ddms = np.zeros(num_sims) + startVar * (np.random.rand(num_sims) - .5)
        resp = np.nan * np.ones(num_sims)                       # initialize choices
        rt = np.nan * np.ones(num_sims)
        drift = np.zeros(num_sims)

        # Determines which drift rate each sim starts with (and switches to), i.e. which stimulus is attended to first
        # Does this by looking at the first_fix and then setting it to True (1/House) or False (0/Face)
        current_drift = np.array(test_data[subject].first_fix, dtype=bool)
        # set lower boundary, upper value updated each loop
        lower = -1 * upper
        # tools for swapping the drift dynamically
        # vector that will link to the correct dwell time before changing drift
        indexing_vector = np.zeros(num_sims).astype(int)
        # just a count of all rows to use for indexing with the indexing_vector
        all_rows = np.arange(num_sims)
        # Get the first array of fixation change times
        change_time = dwell_array[all_rows, indexing_vector]

        face = np.array(test_data[subject].val_face)
        house = np.array(test_data[subject].val_house)
        # Left/Face/0 and Right/House/1 drift (implies which item is being attended to)
        # + (driftVar * np.random.randn(num_sims,1))
        drift_left = scaling * (face + (theta * house))
        # + (driftVar * np.random.randn(num_sims,1))
        drift_right = scaling * (house + (theta * face))

        # initUpper = upper                 # for collapsing bounds
        # AUC = np.zeros((len(drift),1))    # signed added up (not essential)
        # AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)

        while (t < (maxRT / precision)) and np.count_nonzero(within_bounds):  # any paths still going

            t = t + 1

            # at the end of each pass check if any simulations have reached their change time
            # if so...
            # add one to the indexing vector for that simulation
            indexing_vector[np.where(t >= change_time)] += 1
            # swap the current drift for that simulation
            # create an array of all sims where fixation should switch
            change_fixation = np.array(np.where(t >= change_time))
            # in place flips boolean vals based on change_fixation array
            np.logical_not.at(current_drift, change_fixation)
            # then recreate the change_time value based on the new indexing_vector
            change_time = dwell_array[all_rows, indexing_vector]

            # initialize all drift rates
            np.copyto(drift, drift_left)                                # np.copyto(to, from)

            # overwrite those that are True with other drift rate
            np.copyto(drift, drift_right, where=current_drift)          # np.copyto(to, from, mask)

            # only dealing with those that are still going...
            # SHOULD this be a copyto?
            meanDrift = drift[within_bounds]

            # Roger (variance is at .1 so might not need square root)
            dy = meanDrift * precision + (s * (np.sqrt(precision)) *
                                          np.random.randn(np.sum(within_bounds,)))

            ddms[within_bounds] = ddms[within_bounds] + dy

    #       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
    #       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);

    #        if ReturnTraces
    #        traces(:,t) = ddms(TracesToRetain);
    #        break

            # select ddms that have completed but don't have response data recorded
            within_bounds[ddms >= upper] = False
            within_bounds[ddms <= lower] = False
            # identifying ddms that crossed at time t (within_bounds = False, no logged response)
            # justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
            justDone = (within_bounds == False) & (np.isnan(resp))

            # log resp and RT data
            # sign 1 if value is positive or -1 if value is negative
            resp[justDone] = np.sign(ddms[justDone])
            rt[justDone] = t

        # COLLAPSING BOUNDARIES
        # upper = initUpper * exp(-1 * collapseRate * t);
        # lower = -1 * upper;
        # break

        rt = rt * precision     # convert to seconds
        rt = rt + nonDec        # add non decision time

        test_data[subject].loc[:, 'sim_rt'] = rt
        test_data[subject].loc[:, 'sim_resp'] = resp
        test_data[subject].loc[:, 'sim_drift_left'] = drift_left
        test_data[subject].loc[:, 'sim_drift_right'] = drift_right
        # add FIXATION COUNT!!

        test_data[subject] = test_data[subject].dropna()    # get rid of sims that didn't terminate

    return test_data


def simul_addm(parameter_combos, values_array_addm, dwell_array, num_sims, loop_num,
               startVar, nonDec, nonDecVar, driftVar, maxRT, precision, s):
    """

    Args:
        data: str, are we creating simulation disribution or simulating from test data
    """

    t = 0                                                   # initialize time to zero

    #------------------#
    # GET PARAM COMBO  #
    #------------------#

    scaling = parameter_combos[loop_num, 0]                  # drift scaling on loop
    upper = parameter_combos[loop_num, 1]                    # boundary
    lower = -1 * upper
    theta = parameter_combos[loop_num, 2]                    # discount on non-observed

    #-----------------#
    # INIT VARS       #
    #-----------------#

    # initialize boolean vector indicating whether ddms still within bounds
    within_bounds = np.ones((num_sims), dtype=bool)
    # -.5 to mean center (starting dist. uniform)
    ddms = np.zeros(num_sims) + startVar * (np.random.rand(num_sims) - .5)

    resp = np.nan * np.ones(num_sims)                       # initialize choices
    rt = np.nan * np.ones(num_sims)                         # initialize rt
    drift = np.zeros(num_sims)                              # initialize drift

    # tools for swapping the drift dynamically
    # Determine which stimulus is attended to first
    current_drift = np.random.randint(0, 2, size=num_sims, dtype=bool)
    # vector that will link to the correct dwell time before changing drift
    indexing_vector = np.zeros(num_sims).astype(int)
    # just a count of all rows to use for indexing with the indexing_vector
    all_rows = np.arange(num_sims)
    # Get the first array of fixation change times
    change_time = dwell_array[indexing_vector, all_rows]

    # Left and Right drift (applies to the attended item)
    # + (driftVar * np.random.randn(num_sims,1))
    drift_left = scaling * (values_array_addm[0, :] + (theta * values_array_addm[1, :]))
    # + (driftVar * np.random.randn(num_sims,1))
    drift_right = scaling * (values_array_addm[1, :] + (theta * values_array_addm[0, :]))

    # initUpper = upper                 # for collapsing bounds
    # AUC = np.zeros((len(drift),1))    # signed added up (not essential)
    # AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)

    #-----------------#
    # LOOP.           #
    #-----------------#

    while (t < (maxRT / precision)) and np.count_nonzero(within_bounds):  # any paths still going

        t = t + 1

        # at the end of each pass check if any simulations have reached their change time
        # if so...
        # add one to the indexing vector for that simulation
        indexing_vector[np.where(t >= change_time)] += 1

        # swap the current drift for that simulation
        # create an array of all sims where fixation should switch
        change_fixation = np.array(np.where(t >= change_time))
        # in place flips boolean vals based on change_fixation array
        np.logical_not.at(current_drift, change_fixation)
        # then recreate the change_time value based on the new indexing_vector
        change_time = dwell_array[indexing_vector, all_rows]

        # initialize all drift rates as LEFT drift rate
        np.copyto(drift, drift_left)                                # np.copyto(to, from)

        # overwrite those that are TRUE (current_drift var) with RIGHT drift rate
        np.copyto(drift, drift_right, where=current_drift)          # np.copyto(to, from, mask)

        # only dealing with those that are still going...
        # SHOULD this be a copyto?
        meanDrift = drift[within_bounds]

        # Roger (variance is at .1 so might not need square root)
        dy = meanDrift * precision + (s * (np.sqrt(precision)) *
                                      np.random.randn(np.sum(within_bounds,)))

        ddms[within_bounds] = ddms[within_bounds] + dy

#       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
#       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);

#        if ReturnTraces
#        traces(:,t) = ddms(TracesToRetain);
#        break

        # select ddms that have completed but don't have response data recorded
        within_bounds[ddms >= upper] = False
        within_bounds[ddms <= lower] = False
        # identifying ddms that crossed at time t (within_bounds = False, no logged response)
        # justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
        justDone = (within_bounds == False) & (np.isnan(resp))

        # log resp and RT data
        # sign 1 if value is positive or -1 if value is negative
        resp[justDone] = np.sign(ddms[justDone])
        rt[justDone] = t

    # COLLAPSING BOUNDARIES
    # upper = initUpper * exp(-1 * collapseRate * t);
    # lower = -1 * upper;
    # break

    #-----------------#
    # OUTPUT          #
    #-----------------#

    rt = rt * precision     # convert to seconds
    rt = rt + nonDec        # add non decision time

    dfOut = save_to_df(rt, resp, drift_left, drift_right, values_array_addm,
                       scaling, upper, theta)  # save values in DF
    # add correct column?
    dfOut = dfOut.dropna()                          # get rid of sims that didn't terminate
    return dfOut


def save_to_df(rt, resp, drift_left, drift_right, values_array_addm, scaling, upper, theta):
    """Create dataframe to hold output values from simulation"""
    df = pd.DataFrame(rt, columns=['rt'])
    df['resp'] = resp
    df['drift_left'] = drift_left
    df['drift_right'] = drift_right
    df['val_face'] = values_array_addm[0, :]
    df['val_house'] = values_array_addm[1, :]
    df['summed_val'] = values_array_addm[0, :] + values_array_addm[1, :]
    df['scaling'] = scaling
    df['upper_boundary'] = upper
    df['theta'] = theta
    return df


def create_drift_array(x, y, drift_a1, drift_b1, drift_aM, drift_bM):
    """
    Build an array of drift values that takes into account the initial discounted drift

    Args:
        x(int): number of simulations
        y(int): max number of fixations
        drift_a1(np.array): 'a' drift for first fixation
        drift_b1(np.array): 'b' drift for first fixation
        drift_aM(np.array): 'a' drift for subsequent fixations
        drift_bM(np.array): 'b' drift for subsequent fixations
    """
    # initialize
    drift_array = np.empty((x, y))
    drift_array[:, 0] = np.random.randint(2, size=x)

    # create alternating fixation vectors (1,0... and 0,1...)
    a = np.empty((y,), int)
    a[::2] = 1
    a[1::2] = 0
    b = np.empty((y,), int)
    b[::2] = 0
    b[1::2] = 1

    # update drift array
    drift_array[drift_array[:, 0] == 1] = a
    drift_array[drift_array[:, 0] == 0] = b

    # create unique vals for first vs mid columns
    drift_array[:, 1:] += 2

    # transpose so it works with putmask. Also, main body of code is row major...
    drift_array = np.transpose(drift_array)

    # map drifts to array
    # calculate first fix drift left
    np.putmask(drift_array, drift_array == 0, drift_a1)
    # calculate first fix drift right
    np.putmask(drift_array, drift_array == 1, drift_b1)
    # calculate non-first fix drifts left
    np.putmask(drift_array, drift_array == 2, drift_aM)
    # calculate non-first fix drifts right
    np.putmask(drift_array, drift_array == 3, drift_bM)

    return(drift_array)

#----------------------------#
# MAP VERSION FOR PARALLEL   #
#----------------------------#
#@jit(nb.types.UniTuple(nb.float64[:], 2)(nb.int64, nb.float64[:], nb.int64, nb.float64[:], nb.float64, nb.int64, nb.float64, nb.float64), nopython=True)


def simul_addm_rt_dist(parameters, input_vals, dwell_array, dist_type):
    """

    Args:
        dist_type (str): can be either 'percent' or 'count'
    """

    t = 0                                                   # initialize time to zero

    #------------------#
    # GET PARAM COMBO  #
    #------------------#

    scaling = parameters[0]                  # drift scaling on loop
    upper = parameters[1]                    # boundary
    lower = -1 * upper
    theta = parameters[2]              # discount on non-observed
    sp_bias = parameters[3]

    # print(f'Process {os.getpid()} simulating for parameters: scaling = {scaling}, boundary = {upper}, theta = {theta}')

    #-----------------#
    # INIT VARS       #
    #-----------------#

    # initialize boolean vector indicating whether ddms still within bounds
    within_bounds = np.ones((input_vals.num_sims), dtype=bool)

    # Starting Point
    # -.5 to mean center (starting dist. uniform)
    ddms = np.zeros(input_vals.num_sims)\
        + input_vals.startVar * (np.random.rand(input_vals.num_sims) - .5)\
        + (sp_bias * upper)                             # starting point bias (bias proportion * bounday)

    resp = np.nan * np.ones(input_vals.num_sims)        # initialize choices
    rt = np.nan * np.ones(input_vals.num_sims)          # initialize rt
    drift = np.zeros(input_vals.num_sims)               # initialize drift
    fixations = np.zeros(input_vals.num_sims)           # initialize fixation count

    #---------------------------------#
    # Drift Varaibles for Alternating #
    #---------------------------------#

    # Calculate Drift Values
    # Left and Right drift (applies to the attended item) - make this based on item not L/R
    # right now am making Left = A, and Right = B
    first_unattended_value = 0

    drift_a1 = scaling * \
        (input_vals.values_array[0, :] + (theta * first_unattended_value))
    drift_b1 = scaling * \
        (input_vals.values_array[1, :] + (theta * first_unattended_value))

    drift_aM = scaling * \
        (input_vals.values_array[0, :] + (theta * input_vals.values_array[1, :]))
    drift_bM = scaling * \
        (input_vals.values_array[1, :] + (theta * input_vals.values_array[0, :]))

    # init drift array
    drift_array = create_drift_array(
        input_vals.num_sims, dwell_array.shape[0], drift_a1, drift_b1, drift_aM, drift_bM)

    # vector that will link to the correct dwell time before changing drift
    indexing_vector = np.zeros(input_vals.num_sims).astype(int)
    # just a count of all rows to use for indexing with the indexing_vector
    all_rows = np.arange(input_vals.num_sims)
    # Get the first array of fixation change times
    change_time = dwell_array[indexing_vector, all_rows]

    # initUpper = upper                 # for collapsing bounds
    # AUC = np.zeros((len(drift),1))    # signed added up (not essential)
    # AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)

    #-----------------#
    # LOOP.           #
    #-----------------#

    while (t < (input_vals.maxRT / input_vals.precision)) and np.count_nonzero(within_bounds):  # any paths still going

        t = t + 1

        # at the end of each pass check if any simulations have reached their change time
        # if so...
        # add one to the indexing vector for that simulation
        indexing_vector[np.where(t >= change_time)] += 1

        # write the change_time value based on the new indexing_vector
        change_time = dwell_array[indexing_vector, all_rows]
        # drift value based on indexing_vector
        drift = drift_array[indexing_vector, all_rows]

        # only dealing with those that are still going...
        # SHOULD this be a copyto?
        meanDrift = drift[within_bounds]

        # Roger (variance is at .1 so might not need square root)
        dy = meanDrift * input_vals.precision + \
            (input_vals.s * (np.sqrt(input_vals.precision)) * np.random.randn(np.sum(within_bounds,)))

        ddms[within_bounds] = ddms[within_bounds] + dy

#       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
#       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);

#        if ReturnTraces
#        traces(:,t) = ddms(TracesToRetain);
#        break

        # select ddms that have completed but don't have response data recorded
        within_bounds[ddms >= upper] = False
        within_bounds[ddms <= lower] = False
        # identifying ddms that crossed at time t (within_bounds = False, no logged response)
        # justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
        justDone = (within_bounds == False) & (np.isnan(resp))

        # log resp and RT data
        # sign 1 if value is positive or -1 if value is negative
        resp[justDone] = np.sign(ddms[justDone])
        # adding 100ms for each CHANGE of fixation.
        rt[justDone] = t + (100 * indexing_vector[justDone])
        # this will give us the TOTAL fixations (change of fixation + 1)
        fixations[justDone] = indexing_vector[justDone] + 1

    # COLLAPSING BOUNDARIES
    # upper = initUpper * exp(-1 * collapseRate * t);
    # lower = -1 * upper;
    # break

    #-----------------#
    # OUTPUT          #
    #-----------------#

    rt = rt * input_vals.precision     # convert to seconds
    rt = rt + input_vals.nonDec        # add non decision time

    df = pd.DataFrame(rt, columns=['rt'])
    df['resp'] = resp
    # df['drift_left'] = drift_left
    # df['drift_right'] = drift_right
    df['val_face'] = input_vals.values_array[0, :]
    df['val_house'] = input_vals.values_array[1, :]
    df['summed_val'] = input_vals.values_array[0, :] + input_vals.values_array[1, :]
    df['scaling'] = scaling
    df['upper_boundary'] = upper
    df['theta'] = theta
    df['sp_bias'] = sp_bias
    df['fixations'] = fixations

    # add correct column?
    df = df.dropna()                          # get rid of sims that didn't terminate

    #-----------------#
    # RT DIST         #
    #-----------------#

    # from IPython.core.debugger import Tracer; Tracer()()

    # Create upper level dict to hold param combos
    rtDist = {}
    fix_count = {}

    # Create bins for RT distribution
    bins = np.arange(input_vals.nonDec, 10.1, .1)             # start, stop, step (used for hist)
    # delete the last value so bin number = count number
    binz = np.delete(bins, -1)

    # List of all unique value pairs
    value_pairs = np.transpose(np.unique(input_vals.values_array, axis=1)
                               )   # transposing to get rows of value pairs

    # Name for outer dict based on parameters, with an integer (as STRING) leading
    extracted_parameters = str(round(scaling, 3)) + '_' + \
        str(round(upper, 3)) + '_' + str(round(theta, 3)) + '_' + \
        str(round(sp_bias, 3))
    # Create dict to hold values
    rtDist[extracted_parameters] = {}
    fix_count[extracted_parameters] = {}

    y = 0  # counter for value_pairs
    # create subsets of RT for each drift value
    # should this be vectorized or mapped?
    for x in value_pairs:
        data = df[(df.val_face == value_pairs[y][0]) & (df.val_house == value_pairs[y][1])]

        data0 = data[data.resp == -1]             		# select reject responses
        data1 = data[data.resp == 1]              		# select accept responses

        # Create RT distrib (counts/bin)
        count0, bins = np.histogram(data0.rt, bins)  	# unpack the reject counts in each bin
        count1, bins = np.histogram(data1.rt, bins)  	# unpack the accept counts in each bin

        length = float(np.sum(count0) + np.sum(count1))    	# number of non NaN values

        # initialize array to hold Distribs
        distrib = np.ndarray((len(count0), 4))

        if dist_type == "percent":                          # for sim distributions
            distrib[:, 0] = binz                      		# bin values from START of bin
            distrib[:, 1] = count0 / length              	# reject
            distrib[:, 2] = count1 / length              	# accept

        elif dist_type == "count":                          # for sim subjects
            distrib[:, 0] = binz                      	    # bin values from START of bin
            distrib[:, 1] = count0              		    # reject
            distrib[:, 2] = count1              		    # accept

        # calculate mean number of fixations for this value pair
        mean_fix = np.sum(data.fixations) / len(data.fixations)
        # create dict key based on value pair
        vp = str(x[0]) + '_' + str(x[1])
        # select the rows with given drift  # remove all columns except rt and resp
        rtDist[extracted_parameters][vp] = distrib
        fix_count[extracted_parameters][vp] = mean_fix

        y += 1

    # print(f'Finished processing parameters: scaling = {scaling}, boundary = {upper}, theta = {theta}')

    return(rtDist, fix_count)

#-------------------#
# FITTING           #
#-------------------#


def rt_dists_mean(rt_path, rt_dists):
    """Combine RT distributions that have been created across nodes.

    Args:
        rt_path (str): Path and the common part of the file name, e.g. "/gpfs/fs0/scratch/c/chutcher/wilsodj/01_MADE/version/004/output/2018_09_28_0006/rt_dist_"
        rt_dists (list): All simulated distributions to be combined, e.g. ['01', '02', '03']

    Returns:
        tuple: a nested dict, access by rtdist[i][parameter_combo][value_combo], where 'i' is an int from [0,len(rtdist)]

    Example:
        # Usage
        combined_rt_dist = utils_addm.rt_dists_mean(rt_path, rt_dists)

        # Saving
        utils_addm.pickle_save(rt_path, "combined.pickle", combined_rt_dist)
    """
    # Load rt_dists
    dfs = {}
    for i in range(0, len(rt_dists)):
        print('Loading rt dist {0}...'.format(i))
        dfs[i] = pickle.load(open(rt_path + rt_dists[i] + ".pickle", "rb"))

    # Deep copy - this will be modified with the mean of all rt_dists
    rt_full = copy.deepcopy(dfs[0])

    # Get mean values of rt_dists and save to rt_full
    i = 0
    while i < len(rt_full):                      # going through one set of paramaters at a time
        for param_combo in rt_full[i].keys():    # this is pulling out the actual paramter values
            # this is pulling out the value combo (left and right)
            for value in rt_full[i][param_combo]:
                # initialize summed_value with first rt_dist
                summed_rt_dist = dfs[0][i][param_combo][value]
                for j in range(1, len(rt_dists)):              # iterate through rest of rt_dists to combine
                    summed_rt_dist = summed_rt_dist + dfs[j][i][param_combo][value]
                rt_full[i][param_combo][value] = summed_rt_dist / \
                    len(rt_dists)    # divide to get mean
        i += 1
    # don't need dfs anymore
    del(dfs)

    return rt_full


def zero_to_eps(rt_dist):
    """
    Convert zeros to epsilon to avoid inf in log

    Args:
        rt_dist (tuple): tuple of dicts of all simulated rt proportions
    """
    epsilon = np.finfo(float).eps

    for i in range(len(rt_dist)):
        for param_combo in rt_dist[i]:
            for value in rt_dist[i][param_combo]:
                rt_dist[i][param_combo][value][rt_dist[i][param_combo][value] == 0] = epsilon

    return(rt_dist)


def fit_subject(i, sub_sims, rt_dist):
    # track which subjects are being processed
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
                nll_fit = 0              # initialize the NLL to zero to start each param combo test

                # walk through the response (choice/RT) for each value combo
                for value in sub_sims[i][subject]:
                    # check fit between subject response and simulated paramter response rates...
                    nll_fit = nll_fit + \
                        np.nansum((sub_sims[i][subject][value][:, 1:3]) *
                                  nll(rt_dist[j][params][value][:, 1:3]))

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


def calc_best_fit_ps(NLL):
    """
    Calculate a half-guassian weighted fit of the top 20 parameter combinations
    """

    param_num = 4
    num_averaged = 20

    fit_points = np.linspace(0., 3., num=num_averaged)
    # Weight each fit by position using a normal dist
    weighted_fit_points = scipy.stats.norm(0, 1).pdf(fit_points)
    # Calculated the weighted mean of the first 20 (or whatever you set it to) fits
    best_fit = np.sum((NLL.iloc[:num_averaged, 0:param_num].T * weighted_fit_points).T) / \
        np.sum(weighted_fit_points)
    return(best_fit)


def calc_best_fit_1(NLL):
    """
    Pull out the top fit
    """

    param_num = 4

    # Calculated the weighted mean of the first 20 (or whatever you set it to) fits
    best_fit = NLL.iloc[0, 0:param_num]
    return(best_fit)


# HELPER FUNCTIONS (BELOW)


def nll(x):
    """Take negative log likelihood of number
    """
    x = -2 * np.log(x)
    return (x)

# HELPER FUNCTIONS (ABOVE)


def combine_sims(input_filepath, output_filepath):
    """
    """
    all_files = glob.glob(os.path.join(input_filepath, "*.csv")
                          )                        # advisable to use os.path.join as this makes concatenation OS independent

    dfs = {}  # create dict to hold dataframes
    x = 0

    # Fill up the dictionary with all CSVs
    for f in tqdm(all_files):
        dfs[x] = pd.read_csv(f)
        dfs[x] = dfs[x].drop(dfs[x].columns[[0]], axis=1)  # get rid of unnamed column
        dfs[x] = reduce_mem_usage(dfs[x])
        x += 1

    # create the directory if it does not already exist
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    pickle_out = open(output_filepath + "sims_dict.pickle", "wb")
    pickle.dump(dfs, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()


def rtDistFunc(nonDec, values_array, path_to_save):

    # open pickle file with all simulations
    pickle_in = open(path_to_save + "sims_dict.pickle", "rb")
    all_sims = pickle.load(pickle_in)
    pickle_in.close()

    # Create upper level dict to hold param combos
    rtDist = {}

    # from IPython.core.debugger import Tracer; Tracer()()

    # Create bins for RT distribution
    bins = np.arange(nonDec, 10.1, .1)             # start, stop, step (used for hist)
    # delete the last value so bin number = count number
    binz = np.delete(bins, -1)

    # loop through the simulations for each parameter combo that was simulated
    for param_combo in tqdm(all_sims):
        # List of all unique value pairs
        # transposing to get rows of value pairs
        value_pairs = np.transpose(np.unique(values_array, axis=1))

        # Name for outer dict based on the weight, boundary and theta, with an integer (as STRING) leading
        extracted_parameters = str(round(all_sims[param_combo].scaling[0], 3)) + '_' + str(round(
            all_sims[param_combo].upper_boundary[0], 3)) + '_' + str(round(all_sims[param_combo].theta[0], 3))
        # Create nested dict to hold values
        rtDist[extracted_parameters] = {}

        y = 0  # counter for value_pairs
        # create subsets of RT for each drift value
        for x in value_pairs:
            data = all_sims[param_combo][(all_sims[param_combo].val_face == value_pairs[y][0]) & (
                all_sims[param_combo].val_house == value_pairs[y][1])]

            data0 = data[data.resp == -1]             # select reject responses
            data1 = data[data.resp == 1]              # select accept responses

            # Create RT distrib (counts/bin)
            count0, bins = np.histogram(data0.rt, bins)  # unpack the reject counts in each bin
            count1, bins = np.histogram(data1.rt, bins)  # unpack the accept counts in each bin

            length = float(sum(count0) + sum(count1))  # number of non NaN values

            # initialize array to hold Distribs
            distrib = np.ndarray((len(count0), 3))
            distrib[:, 0] = binz                       # bin values from START of bin
            distrib[:, 1] = count0 / length              # reject
            distrib[:, 2] = count1 / length              # accept

            # create dict key based on value pair
            vp = str(x[0]) + '_' + str(x[1])
            # select the rows with given drift  # remove all columns except rt and resp
            rtDist[extracted_parameters][vp] = distrib

            y += 1

    pickle_out = open(path_to_save + "rt_dist.pickle", "wb")
    pickle.dump(rtDist, pickle_out)
    pickle_out.close()


def fit_sim_subjects(subject_dict, rt_dist, nonDec, path_to_save):
    """Fits individual subjects, finding the negative log liklihood of each paramter combo.

    Args:
        subject_dict (pickled dict): value combos, rts and choices for subjects
        rt_dist (pickled dict): simulated distributions for parameter combos and values
        nonDec (float): non decision time as added to rts
        path_to_save (str): where the output pickle file will be stored
    """

    # create dict to store fitted subjects
    sim_subj = {}
    # value to replace zero values
    epsilon = np.finfo(float).eps

    # go through all simulated subjects
    for subject in tqdm(subject_dict):

        scaling_rounded = round(subject_dict[subject].scaling[0], 3)
        upper_boundary_rounded = round(subject_dict[subject].upper_boundary[0], 3)
        theta_rounded = round(subject_dict[subject].theta[0], 3)

        subj_id = str(scaling_rounded) + '_' + \
            str(upper_boundary_rounded) + '_' + str(theta_rounded)

        for key in rt_dist.keys():                                              # go through all parameter combinations

            rt_distList = []                                                    # create list to store data

            # go through all trials for subject 's'
            for x in range(len(subject_dict[subject])):

                value_combo = str(subject_dict[subject].val_face[x]) + \
                    '_' + str(subject_dict[subject].val_house[x])

                # define row of rt_dist that matches subj rt
                row = int((subject_dict[subject].rt[x] - nonDec) / .1)
                if row > 91:                                                    # hacky way to deal with 10 seconds
                    row = 91

                if (subject_dict[subject].resp[x] == -1):                       # if reject
                    rt_distList.append(rt_dist[key][value_combo][row, 1])
                    # sim_subj[s][key][x] = rt_dist[key][driftVal][row,1]

                else:                                                           # if accept
                    rt_distList.append(rt_dist[key][value_combo][row, 2])
                    # sim_subj[s][key][x] = rt_dist[key][driftVal][row,2]
            # import pdb; pdb.set_trace()

            # Create column from list
            subject_dict[subject][key] = rt_distList
            subject_dict[subject] = subject_dict[subject].replace(
                to_replace=0, value=epsilon)     # Remove values of 0 and replace with Epsilon

        # Convert to natural log and Remove unnecessary columns
        sim_subj[subj_id] = np.log(subject_dict[subject].iloc[:, 7:]) * -1
        # Add parameter values
        sim_subj[subj_id].scaling = scaling_rounded
        sim_subj[subj_id].upper_boundary = upper_boundary_rounded
        sim_subj[subj_id].theta = theta_rounded

    # Save to pickle
    path_to_save = path_to_save + "sim_subj_MLE.pickle"
    pickle_out = open(path_to_save, "wb")
    pickle.dump(sim_subj, pickle_out)
    pickle_out.close()

    print(f'Saved to {path_to_save}.')


def fit_subjects(subject_dict, rt_dist, nonDec, path_to_save):
    """
    Fits individual subjects, finding the negative log liklihood of each paramter combo.

    Args:
        subject_dict: pickled dict, value combos, rts and choices for subjects
        rt_dist: pickled dict, simulated distributions for parameter combos and values
        nonDec: float, non decision time as added to rts
        path_to_save: str, where the output pickle file will be stored
    """

    # create dict to store fitted subjects
    fit_subj = {}
    # value to replace zero values
    epsilon = np.finfo(float).eps

    # go through all simulated subjects
    for subject in tqdm(subject_dict):

        for key in rt_dist.keys():                                              # go through all parameter combinations

            rt_distList = []                                                    # create list to store data

            # go through all trials for subject based on index value
            for x in subject_dict[subject].index:
                value_combo = str(subject_dict[subject].val_face[x]) + \
                    '_' + str(subject_dict[subject].val_house[x])

                # define row of rt_dist that matches subj rt
                row = int((subject_dict[subject].rt[x] - nonDec) / .1)
                if row > 91:                                                    # hacky way to deal with 10 seconds
                    row = 91

                if (subject_dict[subject].resp[x] == -1):                       # if reject
                    rt_distList.append(rt_dist[key][value_combo][row, 1])
                    # sim_subj[s][key][x] = rt_dist[key][driftVal][row,1]

                else:                                                           # if accept
                    rt_distList.append(rt_dist[key][value_combo][row, 2])
                    # sim_subj[s][key][x] = rt_dist[key][driftVal][row,2]

            #--------
            # Getting a warning here: A value is trying to be set on a copy of a slice from a DataFrame.
            # Try using .loc[row_indexer,col_indexer] = value instead
            #
            # subject_dict[subject].loc[]
            #--------
            # Create column from list
            subject_dict[subject][key] = rt_distList
            subject_dict[subject] = subject_dict[subject].replace(
                to_replace=0, value=epsilon)     # Remove values of 0 and replace with Epsilon

        # Convert to natural log and Remove unnecessary columns
        fit_subj[subject] = np.log(subject_dict[subject].iloc[:, 9:]) * -1

    # make directory if it does not exist
    directory = path_to_save                            # set Directory to save file

    if not os.path.exists(directory):                   # create the directory if it does not already exist
        os.makedirs(directory)

    # Save to pickle
    path_to_save = directory + "/MLE.pickle"
    pickle_out = open(path_to_save, "wb")
    pickle.dump(fit_subj, pickle_out)
    pickle_out.close()

    print(f'Saved to {path_to_save}.')


def sort_sim_subject_fit(sim_subj_mle, sim_subjects, path_to_save):
    """
    Create dict of sorted summed MLE values by subject

    Args:

    """

    subject_fit = {}
    i = 0

    for x in tqdm(sim_subj_mle):
        subject_fit[x] = pd.DataFrame(sim_subj_mle[x].iloc[:, 3:].sum(axis=0))
        subject_fit[x] = subject_fit[x].sort_values([0])

        scaling = []
        boundary = []
        theta = []

        for m in range(len(subject_fit[x].index)):
            # create list of weight values
            scaling.append(float(subject_fit[x].index[m].split('_')[0]))
            # create list of boundary values
            boundary.append(float(subject_fit[x].index[m].split('_')[1]))
            theta.append(float(subject_fit[x].index[m].split('_')[2])
                         )          # create list of boundary values

        # create column with weight values
        subject_fit[x]['scaling'] = scaling
        # create column with boundary values
        subject_fit[x]['boundary'] = boundary
        # create column with theta values
        subject_fit[x]['theta'] = theta

        subject_fit[x] = subject_fit[x].reset_index(
            drop=True)                  # rename index to integer values

        subject_fit[x]['act_scaling'] = round(sim_subjects[i].scaling[0], 3)
        subject_fit[x]['act_boundary'] = round(sim_subjects[i].upper_boundary[0], 3)
        subject_fit[x]['act_theta'] = round(sim_subjects[i].theta[0], 3)

        i = i + 1

        # Save to pickle
    path_to_save = path_to_save + "summed_sim_subj_MLE.pickle"
    pickle_out = open(path_to_save, "wb")
    pickle.dump(subject_fit, pickle_out)
    pickle_out.close()
    print(f'Saved to {path_to_save}.')


def sort_subject_fit(subj_mle, path_to_save):
    """
    Create dict of sorted summed MLE values by subject

    Args:

    """

    subject_fit = {}
    i = 0

    for x in tqdm(subj_mle):
        subject_fit[x] = pd.DataFrame(subj_mle[x].iloc[:, 3:].sum(axis=0))
        subject_fit[x] = subject_fit[x].sort_values([0])

        scaling = []
        boundary = []
        theta = []

        for m in range(len(subject_fit[x].index)):
            # create list of weight values
            scaling.append(float(subject_fit[x].index[m].split('_')[0]))
            # create list of boundary values
            boundary.append(float(subject_fit[x].index[m].split('_')[1]))
            theta.append(float(subject_fit[x].index[m].split('_')[2])
                         )          # create list of boundary values

        # create column with weight values
        subject_fit[x]['scaling'] = scaling
        # create column with boundary values
        subject_fit[x]['boundary'] = boundary
        # create column with theta values
        subject_fit[x]['theta'] = theta

        subject_fit[x] = subject_fit[x].reset_index(
            drop=True)                  # rename index to integer values

    # Save to pickle
    path_to_save = path_to_save + "summed_MLE.pickle"
    pickle_out = open(path_to_save, "wb")
    pickle.dump(subject_fit, pickle_out)
    pickle_out.close()
    print(f'Saved to {path_to_save}.')


def pickle_read(path_to_file):
    pickle_in = open(path_to_file, "rb")
    out_file = pickle.load(pickle_in)
    pickle_in.close()
    return out_file


def pickle_save(path, name, file):
    filename = path + name
    outfile = open(filename, 'wb')
    pickle.dump(file, outfile)
    outfile.close()


# Function to reduce size of dataframes
def reduce_mem_usage(props):
    # start_mem_usg = props.memory_usage().sum() / 1024**2
    # print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            # print("******************************")
            # print("Column: ",col)
            # print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ",props[col].dtype)
            # print("******************************")

    # Print final result
    # print("___MEMORY USAGE AFTER COMPLETION:___")
    # mem_usg = props.memory_usage().sum() / 1024**2
    # print("Memory usage is: ",mem_usg," MB")
    # print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props
