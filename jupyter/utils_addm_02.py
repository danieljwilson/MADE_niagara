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
import itertools                            # for creating values list (for aDDM of all stimuli value combos)
import os
import time                                 # for tracking timing of loops
import shelve
from IPython.display import clear_output    # to overwrite the avg. loop time/remining time with each loop
import pickle                               # for saving to pickle
import glob
from tqdm import tqdm                       # for keeping track of progress
import collections


Input_Values = collections.namedtuple('Input_Values', [
    'parameter_combos',
    'values_array',
    'dwell_array',
    'num_sims',
    'loop_num',
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

    values_array = np.repeat(values, sims_per_val) # duplicate values * simsPerVal
    values_array = values_array.reshape(len(values_array),1) # make 2D array

    # PRINT SUMMARY
    print("You will run {0} simulations for each of {1} values for a total of {2} simulations.".format(sims_per_val, len(values), values_array.shape[0]))

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
    values = np.around(values,decimals=2)   # currently produces 81 combos
    t_vals = np.transpose(values)           # transpose to take advantage of row major format of Python
    
    values_array = np.repeat(t_vals, sims_per_val, axis=1) # duplicate values * simsPerVal
    
    # PRINT SUMMARY
    print("You will run {0} simulations for each of {1} value combinations for a total of {2} simulations.".format(sims_per_val, len(values), values_array.shape[1]))

    return values_array


def parameter_values(drift_weight, upper_boundary, theta):
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
    
    parameter_combos = np.array(list(itertools.product(drift_weight, upper_boundary, theta)))

    # CHECK SIZE OF ARRAY
    print("Your parameter search space is size: {0}.".format(parameter_combos.shape))
    save_parameters(parameter_combos)

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
    print("Total sims = ", len(parameter_combos)* values_array.shape[1])

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
    #now = dt.datetime.now()
    #time = now.strftime("%Y-%m-%d-%H-%M")
    #date = now.strftime("%Y-%m-%d")

    directory = ('outputs/' + str(date) + '/sim_subj/') # set Directory to save CSV

    if not os.path.exists(directory):                   # create the directory if it does not already exist
        os.makedirs(directory)

    #dfOut.to_csv(directory + params + '.csv')
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

        df = pd.DataFrame.from_csv(fixations, header=0, sep=",", index_col=None)
        
        # first create distributions of first/mid fixations
        first_fix_dist = df['fix_time'][df['fix_num']==1]
        mid_fix_dist = df['fix_time'][(df['fix_num']>1) & (df['rev_fix_num']>1)]


    if data == 'test':
        df = fixations
        
        # first create distributions of first/mid fixations
        first_fix_dist = df['fix_time'][df['fix_num']==1]
        mid_fix_dist = df['fix_time'][(df['fix_num']>1) & (df['rev_fix_num']>1)]

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
        while min_time < 10000:   # 10,000 = maxRT
            # create column of first fixations
            dwell_array = np.reshape(((np.random.choice(first_fix_dist, num_sims, replace=True))), (num_sims,1))

            # create additional columns from middle fixation distribution
            for column in range(20):
                append_mid_fix_array = np.reshape(((np.random.choice(mid_fix_dist, num_sims, replace=True))), (num_sims,1))
                dwell_array = np.append(dwell_array, append_mid_fix_array, axis=1)

            # make each column the sum of itself and the previous column
            for column in range(1, np.shape(dwell_array)[1]):
                dwell_array[:,column] = dwell_array[:,column] + dwell_array[:,(column-1)] 

            min_time = min(dwell_array[:,20])


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
            dwell_array = np.reshape(((np.random.choice(first_fix_dist, size , replace=True))), (size,1))

            # create additional columns (number of columns determined by range()) from middle fixation distribution
            for column in range(21):
                append_mid_fix_array = np.reshape(((np.random.choice(mid_fix_dist, size, replace=True))), (size,1))
                dwell_array = np.append(dwell_array, append_mid_fix_array, axis=1)

            # overwrite initial columns with actual fixation values
            row_start = 0
            row_end = 1000  # Need to change this to the number of sims!
            column = 0

            #from IPython.core.debugger import Tracer; Tracer()()
            for i in tqdm(range(len(fixations.trial))):
                if i == 0:    # first row
                    dwell_array[row_start:row_end, column] = fixations.fix_time[i]
                    

                elif (fixations.trial[i]) == (fixations.trial[i-1]):
                    column +=1
                    dwell_array[row_start:row_end, column] = fixations.fix_time[i]

                else:
                    column = 0
                    row_start +=1000
                    row_end += 1000
                    dwell_array[row_start:row_end, column] = fixations.fix_time[i]

            # make each column the sum of itself and the previous column
            for column in range(1, np.shape(dwell_array)[1]):
                dwell_array[:,column] = dwell_array[:,column] + dwell_array[:,(column-1)] 

            min_time = min(dwell_array[:,21])
        
    # cast to int (does this make it faster?)
    dwell_array = dwell_array.astype(int)
    t_dwells = np.transpose(dwell_array)        # this is to make things row major
    
    return t_dwells


def simul_ddm(parameter_combos, values_array, num_sims,loop_num,
    startVar, nonDec, nonDecVar, driftVar, maxRT, respOptions, precision, s):

    t = 0                                               # initialize time to zero

    scaling = parameter_combos[loop_num,0]               # drift scaling on loop
    upper = parameter_combos[loop_num,1]                 # boundary on loop
    within_bounds = np.ones((num_sims,1), dtype=bool)   # initialize location vector for each drift
    ddms = np.zeros((num_sims,1)) + startVar * (np.random.rand(num_sims,1) - .5)  # -.5 to mean center (starting dist. uniform)
    resp = np.nan * np.ones((num_sims,1))               # initialize choices
    rt = np.nan * np.ones((num_sims,1))  
 
    # Where are we:
    print("Simulating for scaling {0} and boundary {1}. Loop {2}/{3}".format(scaling, upper, loop_num, len(parameter_combos)))

    # upper value updated each loop
    lower = -1 * upper
    # drift scaling updated each loop
    drift = (scaling * values_array) +  (driftVar * np.random.randn(num_sims,1))

    #initUpper = upper                 # for collapsing bounds
    #AUC = np.zeros((len(drift),1))    # signed added up (not essential)
    #AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)

    while (t < (maxRT/precision)) and np.count_nonzero(within_bounds): # any paths still going   

        t = t+1
    
        meanDrift = drift[within_bounds]
    
        # Roger (variance is at .1 so might not need square root)
        dy = meanDrift * precision + (s * (np.sqrt(precision)) * np.random.randn(np.sum(within_bounds,)))

        ddms[within_bounds] = ddms[within_bounds] + dy
    
#       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
#       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);
    
#        if ReturnTraces
#        traces(:,t) = ddms(TracesToRetain);
#        break
    
        # select ddms that have completed but don't have response data recorded
        within_bounds[ddms>=upper] = False
        within_bounds[ddms<=lower] = False
        # identifying ddms that crossed at time t (within_bounds = False, no logged response)
        #justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
        justDone = (within_bounds == False) & (np.isnan(resp))  
        
        # log resp and RT data
        resp[justDone] = np.sign(ddms[justDone])  # sign 1 if value is positive or -1 if value is negative
        rt[justDone] = t
    
    # COLLAPSING BOUNDARIES
    #upper = initUpper * exp(-1 * collapseRate * t);
    #lower = -1 * upper;
    #break
    
    rt = rt * precision     # convert to seconds
    rt = rt + nonDec      # add non decision time
    
    dfOut = save_to_df(rt, resp, drift, scaling, upper)  #save values in DF
    return dfOut


def time_estimate(loop_num, parameter_combos, total_sim_time):
    avg_loop_time = total_sim_time/loop_num
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
        num_sims = len(test_data[subject].trial)                # find the length of vector containing all trials

        scaling = test_data[subject].loc[0,'est_scaling']       # drift scaling on loop
        upper = test_data[subject].loc[0,'est_boundary']
        theta = test_data[subject].loc[0,'est_theta']           # boundary on loop
        within_bounds = np.ones((num_sims), dtype=bool)         # initialize location vector for each drift
        ddms = np.zeros(num_sims) + startVar * (np.random.rand(num_sims) - .5)  # -.5 to mean center (starting dist. uniform)
        resp = np.nan * np.ones(num_sims)                       # initialize choices
        rt = np.nan * np.ones(num_sims)  
        drift = np.zeros(num_sims)
     
        # Determines which drift rate each sim starts with (and switches to), i.e. which stimulus is attended to first
        # Does this by looking at the first_fix and then setting it to True (1/House) or False (0/Face)
        current_drift = np.array(test_data[subject].first_fix, dtype = bool)
        # set lower boundary, upper value updated each loop
        lower = -1 * upper
        # tools for swapping the drift dynamically
        indexing_vector = np.zeros(num_sims).astype(int)        # vector that will link to the correct dwell time before changing drift
        all_rows = np.arange(num_sims)                          # just a count of all rows to use for indexing with the indexing_vector
        change_time = dwell_array[all_rows, indexing_vector]    # Get the first array of fixation change times 

        face = np.array(test_data[subject].val_face)
        house = np.array(test_data[subject].val_house)
        # Left/Face/0 and Right/House/1 drift (implies which item is being attended to)
        drift_left  = scaling * (face + (theta * house)) # + (driftVar * np.random.randn(num_sims,1))
        drift_right = scaling * (house + (theta * face)) # + (driftVar * np.random.randn(num_sims,1))

        #initUpper = upper                 # for collapsing bounds
        #AUC = np.zeros((len(drift),1))    # signed added up (not essential)
        #AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)

        while (t < (maxRT/precision)) and np.count_nonzero(within_bounds): # any paths still going   
            
            t = t+1

            # at the end of each pass check if any simulations have reached their change time
            # if so...
            # add one to the indexing vector for that simulation
            indexing_vector[np.where(t >=change_time)] += 1
            # swap the current drift for that simulation
            change_fixation = np.array(np.where(t >=change_time))       # create an array of all sims where fixation should switch
            np.logical_not.at(current_drift, change_fixation)           # in place flips boolean vals based on change_fixation array
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
            dy = meanDrift * precision + (s * (np.sqrt(precision)) * np.random.randn(np.sum(within_bounds,)))

            ddms[within_bounds] = ddms[within_bounds] + dy
        
    #       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
    #       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);
        
    #        if ReturnTraces
    #        traces(:,t) = ddms(TracesToRetain);
    #        break
        
            # select ddms that have completed but don't have response data recorded
            within_bounds[ddms>=upper] = False
            within_bounds[ddms<=lower] = False
            # identifying ddms that crossed at time t (within_bounds = False, no logged response)
            #justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
            justDone = (within_bounds == False) & (np.isnan(resp))  
            
            # log resp and RT data
            resp[justDone] = np.sign(ddms[justDone])  # sign 1 if value is positive or -1 if value is negative
            rt[justDone] = t
        
        # COLLAPSING BOUNDARIES
        #upper = initUpper * exp(-1 * collapseRate * t);
        #lower = -1 * upper;
        #break
        
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

    scaling = parameter_combos[loop_num,0]                  # drift scaling on loop
    upper = parameter_combos[loop_num,1]                    # boundary
    lower = -1 * upper
    theta = parameter_combos[loop_num,2]                    # discount on non-observed


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
    current_drift = np.random.randint(0,2, size = num_sims, dtype=bool) # Determine which stimulus is attended to first                                                            
    indexing_vector = np.zeros(num_sims).astype(int)        # vector that will link to the correct dwell time before changing drift
    all_rows = np.arange(num_sims)                          # just a count of all rows to use for indexing with the indexing_vector
    change_time = dwell_array[indexing_vector, all_rows]    # Get the first array of fixation change times 

    # Left and Right drift (applies to the attended item)
    drift_left  = scaling * (values_array_addm[0,:] + (theta * values_array_addm[1,:])) # + (driftVar * np.random.randn(num_sims,1))
    drift_right = scaling * (values_array_addm[1,:] + (theta * values_array_addm[0,:])) # + (driftVar * np.random.randn(num_sims,1))
    
    #initUpper = upper                 # for collapsing bounds
    #AUC = np.zeros((len(drift),1))    # signed added up (not essential)
    #AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)


    #-----------------#
    # LOOP.           #
    #-----------------#

    while (t < (maxRT/precision)) and np.count_nonzero(within_bounds): # any paths still going   
        
        t = t+1

        # at the end of each pass check if any simulations have reached their change time
        # if so...
        # add one to the indexing vector for that simulation
        indexing_vector[np.where(t >=change_time)] += 1

        # swap the current drift for that simulation
        change_fixation = np.array(np.where(t >=change_time))       # create an array of all sims where fixation should switch
        np.logical_not.at(current_drift, change_fixation)           # in place flips boolean vals based on change_fixation array
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
        dy = meanDrift * precision + (s * (np.sqrt(precision)) * np.random.randn(np.sum(within_bounds,)))

        ddms[within_bounds] = ddms[within_bounds] + dy
    
#       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
#       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);
    
#        if ReturnTraces
#        traces(:,t) = ddms(TracesToRetain);
#        break
    
        # select ddms that have completed but don't have response data recorded
        within_bounds[ddms>=upper] = False
        within_bounds[ddms<=lower] = False
        # identifying ddms that crossed at time t (within_bounds = False, no logged response)
        #justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
        justDone = (within_bounds == False) & (np.isnan(resp))  
        
        # log resp and RT data
        resp[justDone] = np.sign(ddms[justDone])  # sign 1 if value is positive or -1 if value is negative
        rt[justDone] = t
    
    # COLLAPSING BOUNDARIES
    #upper = initUpper * exp(-1 * collapseRate * t);
    #lower = -1 * upper;
    #break
    
    #-----------------#
    # OUTPUT          #
    #-----------------#

    rt = rt * precision     # convert to seconds
    rt = rt + nonDec        # add non decision time

    dfOut = save_to_df(rt, resp, drift_left, drift_right, values_array_addm, scaling, upper, theta)  #save values in DF
    # add correct column?
    dfOut = dfOut.dropna()                          # get rid of sims that didn't terminate
    return dfOut


def save_to_df(rt, resp, drift_left, drift_right, values_array_addm, scaling, upper, theta):
    """Create dataframe to hold output values from simulation"""
    df = pd.DataFrame(rt, columns= ['rt'])
    df['resp'] = resp
    df['drift_left'] = drift_left
    df['drift_right'] = drift_right
    df['val_face'] = values_array_addm[0,:]
    df['val_house'] = values_array_addm[1,:]
    df['summed_val'] = values_array_addm[0,:] + values_array_addm[1,:] 
    df['scaling'] = scaling
    df['upper_boundary'] = upper
    df['theta'] = theta
    return df


#----------------------------#
# MAP VERSION FOR PARALLEL   #
#----------------------------#

def simul_addm_rt_dist(parameters, input_vals, dwell_array):
    """

    Args:
        data: str, are we creating simulation disribution or simulating from test data 
    """

    t = 0                                                   # initialize time to zero

    #------------------#
    # GET PARAM COMBO  #
    #------------------#

    scaling = parameters[0]                  # drift scaling on loop
    upper = parameters[1]                    # boundary
    lower = -1 * upper
    theta = parameters[2]              # discount on non-observed

    print(f'Process {os.getpid()} simulating for parameters: scaling = {scaling}, boundary = {upper}, theta = {theta}')

    #-----------------#
    # INIT VARS       #
    #-----------------#

    # initialize boolean vector indicating whether ddms still within bounds
    within_bounds = np.ones((input_vals.num_sims), dtype=bool)       
    
    # -.5 to mean center (starting dist. uniform)
    ddms = np.zeros(input_vals.num_sims)\
    + input_vals.startVar * (np.random.rand(input_vals.num_sims) - .5)  

    resp = np.nan * np.ones(input_vals.num_sims)        # initialize choices
    rt = np.nan * np.ones(input_vals.num_sims)          # initialize rt
    drift = np.zeros(input_vals.num_sims)               # initialize drift
 

    # tools for swapping the drift dynamically
    current_drift = np.random.randint(0,2, size = input_vals.num_sims, dtype=bool) # Determine which stimulus is attended to first                                                            
    indexing_vector = np.zeros(input_vals.num_sims).astype(int)     # vector that will link to the correct dwell time before changing drift
    all_rows = np.arange(input_vals.num_sims)                       # just a count of all rows to use for indexing with the indexing_vector
    change_time = dwell_array[indexing_vector, all_rows]    		# Get the first array of fixation change times 

    # Left and Right drift (applies to the attended item)
    drift_left  = scaling * (input_vals.values_array[0,:] + (theta * input_vals.values_array[1,:])) # + (driftVar * np.random.randn(num_sims,1))
    drift_right = scaling * (input_vals.values_array[1,:] + (theta * input_vals.values_array[0,:])) # + (driftVar * np.random.randn(num_sims,1))
    
    #initUpper = upper                 # for collapsing bounds
    #AUC = np.zeros((len(drift),1))    # signed added up (not essential)
    #AUC2 = np.zeros((len(drift),1))   # absolute value added up (not essential)


    #-----------------#
    # LOOP.           #
    #-----------------#

    while (t < (input_vals.maxRT/input_vals.precision)) and np.count_nonzero(within_bounds): # any paths still going   
        
        t = t+1

        # at the end of each pass check if any simulations have reached their change time
        # if so...
        # add one to the indexing vector for that simulation
        indexing_vector[np.where(t >=change_time)] += 1

        # swap the current drift for that simulation
        change_fixation = np.array(np.where(t >=change_time))       # create an array of all sims where fixation should switch
        np.logical_not.at(current_drift, change_fixation)           # in place flips boolean vals based on change_fixation array
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
        dy = meanDrift * input_vals.precision + (input_vals.s * (np.sqrt(input_vals.precision)) * np.random.randn(np.sum(within_bounds,)))

        ddms[within_bounds] = ddms[within_bounds] + dy
    
#       AUC(within_bounds) = AUC(within_bounds) + abs(ddms(within_bounds));
#       AUC2(within_bounds) = AUC2(within_bounds) + ddms(within_bounds);
    
#        if ReturnTraces
#        traces(:,t) = ddms(TracesToRetain);
#        break
    
        # select ddms that have completed but don't have response data recorded
        within_bounds[ddms>=upper] = False
        within_bounds[ddms<=lower] = False
        # identifying ddms that crossed at time t (within_bounds = False, no logged response)
        #justDone = np.asarray([within_bounds[i]==False and np.isnan(resp[i]) for i in range(len(within_bounds))])
        justDone = (within_bounds == False) & (np.isnan(resp))  
        
        # log resp and RT data
        resp[justDone] = np.sign(ddms[justDone])  # sign 1 if value is positive or -1 if value is negative
        rt[justDone] = t
    
    # COLLAPSING BOUNDARIES
    #upper = initUpper * exp(-1 * collapseRate * t);
    #lower = -1 * upper;
    #break
    
    #-----------------#
    # OUTPUT          #
    #-----------------#

    rt = rt * input_vals.precision     # convert to seconds
    rt = rt + input_vals.nonDec        # add non decision time

    df = pd.DataFrame(rt, columns= ['rt'])
    df['resp'] = resp
    df['drift_left'] = drift_left
    df['drift_right'] = drift_right
    df['val_face'] = input_vals.values_array[0,:]
    df['val_house'] = input_vals.values_array[1,:]
    df['summed_val'] = input_vals.values_array[0,:] + input_vals.values_array[1,:] 
    df['scaling'] = scaling
    df['upper_boundary'] = upper
    df['theta'] = theta

    # add correct column?
    df = df.dropna()                          # get rid of sims that didn't terminate

    #-----------------#
    # RT DIST         #
    #-----------------#

    # from IPython.core.debugger import Tracer; Tracer()()
    
    # Create upper level dict to hold param combos
    rtDist = {}

    # Create bins for RT distribution
    bins = np.arange(input_vals.nonDec,10.1,.1)             # start, stop, step (used for hist)
    binz = np.delete(bins,-1)                    # delete the last value so bin number = count number

    # List of all unique value pairs
    value_pairs = np.transpose(np.unique(input_vals.values_array, axis=1))   # transposing to get rows of value pairs

    # Name for outer dict based on the weight, boundary and theta, with an integer (as STRING) leading
    extracted_parameters = str(round(scaling, 3)) + '_' + str(round(upper, 3)) + '_' + str(round(theta, 3))
    # Create dict to hold values
    rtDist[extracted_parameters] = {}

    y = 0 # counter for value_pairs
    # create subsets of RT for each drift value
    for x in value_pairs:
        data = df[(df.val_face == value_pairs[y][0]) & (df.val_house == value_pairs[y][1])]
        
        data0 = data[data.resp == -1]             		# select reject responses
        data1 = data[data.resp == 1]              		# select accept responses

        # Create RT distrib (counts/bin)
        count0, bins = np.histogram(data0.rt, bins)  	# unpack the reject counts in each bin
        count1, bins = np.histogram(data1.rt, bins)  	# unpack the accept counts in each bin

        length = float(sum(count0) + sum(count1))    	# number of non NaN values

        # initialize array to hold Distribs
        distrib = np.ndarray((len(count0), 3))
        distrib[:,0] = binz                      		# bin values from START of bin
        distrib[:,1] = count0 /length              		# reject
        distrib[:,2] = count1 /length              		# accept

        # create dict key based on value pair
        vp = str(x[0]) + '_' + str(x[1])
        # select the rows with given drift  # remove all columns except rt and resp
        rtDist[extracted_parameters][vp] = distrib

        y+=1

    print(f'Finished processing parameters: scaling = {scaling}, boundary = {upper}, theta = {theta}')

    return(rtDist)

#-------------------#
# FITTING           #
#-------------------#

def combine_sims(input_filepath, output_filepath):
    """
    """
    all_files = glob.glob(os.path.join(input_filepath, "*.csv"))                        # advisable to use os.path.join as this makes concatenation OS independent

    dfs = {}  # create dict to hold dataframes
    x = 0
    
    # Fill up the dictionary with all CSVs
    for f in tqdm(all_files):
        dfs[x] = pd.read_csv(f)
        dfs[x] = dfs[x].drop(dfs[x].columns[[0]], axis=1)  # get rid of unnamed column
        dfs[x] = reduce_mem_usage(dfs[x])
        x+=1
    
    if not os.path.exists(output_filepath):               # create the directory if it does not already exist
        os.makedirs(output_filepath)
        
    pickle_out = open(output_filepath + "sims_dict.pickle","wb")
    pickle.dump(dfs, pickle_out, pickle.HIGHEST_PROTOCOL)
    pickle_out.close()


def rtDistFunc(nonDec, values_array, path_to_save):

    # open pickle file with all simulations
    pickle_in = open(path_to_save + "sims_dict.pickle","rb")
    all_sims = pickle.load(pickle_in)
    pickle_in.close()
    
    # Create upper level dict to hold param combos
    rtDist = {}
    
    # from IPython.core.debugger import Tracer; Tracer()()

    # Create bins for RT distribution
    bins = np.arange(nonDec,10.1,.1)             # start, stop, step (used for hist)
    binz = np.delete(bins,-1)                    # delete the last value so bin number = count number

    for param_combo in tqdm(all_sims):                 # loop through the simulations for each parameter combo that was simulated
        # List of all unique value pairs
        value_pairs = np.transpose(np.unique(values_array, axis=1))   # transposing to get rows of value pairs

        # Name for outer dict based on the weight, boundary and theta, with an integer (as STRING) leading
        extracted_parameters = str(round(all_sims[param_combo].scaling[0] ,3)) + '_' + str(round(all_sims[param_combo].upper_boundary[0] ,3)) + '_' + str(round(all_sims[param_combo].theta[0] ,3))
        # Create nested dict to hold values
        rtDist[extracted_parameters] = {}

        y = 0 # counter for value_pairs
        # create subsets of RT for each drift value
        for x in value_pairs:
            data = all_sims[param_combo][(all_sims[param_combo].val_face == value_pairs[y][0]) & (all_sims[param_combo].val_house == value_pairs[y][1])]
            
            data0 = data[data.resp == -1]             # select reject responses
            data1 = data[data.resp == 1]              # select accept responses

            # Create RT distrib (counts/bin)
            count0, bins = np.histogram(data0.rt, bins)  # unpack the reject counts in each bin
            count1, bins = np.histogram(data1.rt, bins)  # unpack the accept counts in each bin

            length = float(sum(count0) + sum(count1)) # number of non NaN values

            # initialize array to hold Distribs
            distrib = np.ndarray((len(count0), 3))
            distrib[:,0] = binz                       # bin values from START of bin
            distrib[:,1] = count0 /length              # reject
            distrib[:,2] = count1 /length              # accept

            # create dict key based on value pair
            vp = str(x[0]) + '_' + str(x[1])
            # select the rows with given drift  # remove all columns except rt and resp
            rtDist[extracted_parameters][vp] = distrib

            y+=1
    
    pickle_out = open(path_to_save + "rt_dist.pickle","wb")
    pickle.dump(rtDist, pickle_out)
    pickle_out.close()

def fit_sim_subjects(subject_dict, rt_dist, nonDec, path_to_save):
    """
    Fits individual subjects, finding the negative log liklihood of each paramter combo.

    Args:
        subject_dict: pickled dict, value combos, rts and choices for subjects
        rt_dist: pickled dict, simulated distributions for parameter combos and values
        nonDec: float, non decision time as added to rts
        path_to_save: str, where the output pickle file will be stored
    """

    sim_subj = {}                                                               # create dict to store fitted subjects
    epsilon = np.finfo(float).eps                                               # value to replace zero values

    for subject in tqdm(subject_dict):                                          # go through all simulated subjects
        
        scaling_rounded = round(subject_dict[subject].scaling[0], 3)
        upper_boundary_rounded = round(subject_dict[subject].upper_boundary[0], 3)
        theta_rounded = round(subject_dict[subject].theta[0], 3)
        
        subj_id = str(scaling_rounded) + '_' + str(upper_boundary_rounded) + '_' + str(theta_rounded)
        
        for key in rt_dist.keys():                                              # go through all parameter combinations
            
            rt_distList = []                                                    # create list to store data

            for x in range(len(subject_dict[subject])):                         # go through all trials for subject 's'
                
                value_combo = str(subject_dict[subject].val_face[x]) + '_' + str(subject_dict[subject].val_house[x])
                
                row = int((subject_dict[subject].rt[x]-nonDec)/.1)              # define row of rt_dist that matches subj rt
                if row > 91:                                                    # hacky way to deal with 10 seconds
                    row = 91
                
                if (subject_dict[subject].resp[x] == -1):                       # if reject
                    rt_distList.append(rt_dist[key][value_combo][row,1])
                    #sim_subj[s][key][x] = rt_dist[key][driftVal][row,1]

                else:                                                           # if accept
                    rt_distList.append(rt_dist[key][value_combo][row,2])
                    #sim_subj[s][key][x] = rt_dist[key][driftVal][row,2]
            #import pdb; pdb.set_trace()

            subject_dict[subject][key] = rt_distList                            # Create column from list
            subject_dict[subject] = subject_dict[subject].replace(to_replace=0, value=epsilon)     # Remove values of 0 and replace with Epsilon

        sim_subj[subj_id] = np.log(subject_dict[subject].iloc[:, 7:]) * -1      # Convert to natural log and Remove unnecessary columns
        sim_subj[subj_id].scaling = scaling_rounded                             # Add parameter values
        sim_subj[subj_id].upper_boundary = upper_boundary_rounded
        sim_subj[subj_id].theta = theta_rounded
        
    # Save to pickle
    path_to_save = path_to_save + "sim_subj_MLE.pickle"
    pickle_out = open(path_to_save,"wb")
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

    fit_subj = {}                                                               # create dict to store fitted subjects
    epsilon = np.finfo(float).eps                                               # value to replace zero values

    for subject in tqdm(subject_dict):                                          # go through all simulated subjects
                
        for key in rt_dist.keys():                                              # go through all parameter combinations
            
            rt_distList = []                                                    # create list to store data

            for x in subject_dict[subject].index:                               # go through all trials for subject based on index value
                value_combo = str(subject_dict[subject].val_face[x]) + '_' + str(subject_dict[subject].val_house[x])
                
                row = int((subject_dict[subject].rt[x]-nonDec)/.1)              # define row of rt_dist that matches subj rt
                if row > 91:                                                    # hacky way to deal with 10 seconds
                    row = 91
                
                if (subject_dict[subject].resp[x] == -1):                       # if reject
                    rt_distList.append(rt_dist[key][value_combo][row,1])
                    #sim_subj[s][key][x] = rt_dist[key][driftVal][row,1]

                else:                                                           # if accept
                    rt_distList.append(rt_dist[key][value_combo][row,2])
                    #sim_subj[s][key][x] = rt_dist[key][driftVal][row,2]

            #--------
            # Getting a warning here: A value is trying to be set on a copy of a slice from a DataFrame.
            # Try using .loc[row_indexer,col_indexer] = value instead
            #    
            # subject_dict[subject].loc[]
            #--------           
            subject_dict[subject][key] = rt_distList                            # Create column from list
            subject_dict[subject] = subject_dict[subject].replace(to_replace=0, value=epsilon)     # Remove values of 0 and replace with Epsilon

        fit_subj[subject] = np.log(subject_dict[subject].iloc[:, 9:]) * -1      # Convert to natural log and Remove unnecessary columns
        

    # make directory if it does not exist
    directory = path_to_save                            # set Directory to save file

    if not os.path.exists(directory):                   # create the directory if it does not already exist
        os.makedirs(directory)

    # Save to pickle
    path_to_save = directory + "/MLE.pickle"
    pickle_out = open(path_to_save,"wb")
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
            scaling.append(float(subject_fit[x].index[m].split('_')[0]))        # create list of weight values
            boundary.append(float(subject_fit[x].index[m].split('_')[1]))       # create list of boundary values
            theta.append(float(subject_fit[x].index[m].split('_')[2]))          # create list of boundary values
            
        subject_fit[x]['scaling'] = scaling                                     # create column with weight values
        subject_fit[x]['boundary'] = boundary                                   # create column with boundary values
        subject_fit[x]['theta'] = theta                                         # create column with theta values
        
        subject_fit[x] = subject_fit[x].reset_index(drop=True)                  # rename index to integer values

        subject_fit[x]['act_scaling'] = round(sim_subjects[i].scaling[0], 3)
        subject_fit[x]['act_boundary'] = round(sim_subjects[i].upper_boundary[0], 3)
        subject_fit[x]['act_theta'] = round(sim_subjects[i].theta[0], 3)
        
        i = i+1
        
        # Save to pickle
    path_to_save = path_to_save + "summed_sim_subj_MLE.pickle"
    pickle_out = open(path_to_save,"wb")
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
            scaling.append(float(subject_fit[x].index[m].split('_')[0]))        # create list of weight values
            boundary.append(float(subject_fit[x].index[m].split('_')[1]))       # create list of boundary values
            theta.append(float(subject_fit[x].index[m].split('_')[2]))          # create list of boundary values
            
        subject_fit[x]['scaling'] = scaling                                     # create column with weight values
        subject_fit[x]['boundary'] = boundary                                   # create column with boundary values
        subject_fit[x]['theta'] = theta                                         # create column with theta values
        
        subject_fit[x] = subject_fit[x].reset_index(drop=True)                  # rename index to integer values
        

    # Save to pickle
    path_to_save = path_to_save + "summed_MLE.pickle"
    pickle_out = open(path_to_save,"wb")
    pickle.dump(subject_fit, pickle_out)
    pickle_out.close()
    print(f'Saved to {path_to_save}.')

def pickle_read(path_to_file):
    pickle_in = open(path_to_file,"rb")
    out_file = pickle.load(pickle_in)
    pickle_in.close()
    return out_file


# Function to reduce size of dataframes
def reduce_mem_usage(props):
    #start_mem_usg = props.memory_usage().sum() / 1024**2 
    #print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            #print("******************************")
            #print("Column: ",col)
            #print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
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
            #print("dtype after: ",props[col].dtype)
            #print("******************************")
    
    # Print final result
    #print("___MEMORY USAGE AFTER COMPLETION:___")
    #mem_usg = props.memory_usage().sum() / 1024**2 
    #print("Memory usage is: ",mem_usg," MB")
    #print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props


