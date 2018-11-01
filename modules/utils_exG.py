#!/usr/bin/python

"""
util.py
Author: Daniel J Wilson, daniel.j.wilson@gmail.com
Date: April 20, 2018

Utility functions for creating fixation distributions using ExGUtils.
Note: Runs on Python 2

https://pypi.org/project/ExGUtils/
"""

import numpy as np
import pandas as pd
import os
import pickle                               # for saving to pickle
import glob

# ExGUtils Imports
from ExGUtils.pyexg import stats, drand, exp_rvs, gauss_rvs, exg_rvs
from ExGUtils.pyexg import *

def create_subject_dwell_samples(number, fixations_file, output_filepath):
	"""
	Function Description
	"""

	#-------------------------#
	# Import fixation data    #
	#-------------------------#

	fixations = pd.read_csv(fixations_file)


	#-------------------------#
	# Clean data              #
	#-------------------------#

	# Cleaned fixations of less than 100ms
	fixations_cleaned = fixations[fixations.fix_time > 100]

	# Indicate percentage of removed fixations
	removed = ((float(len(fixations)) - float(len(fixations_cleaned)))/float(len(fixations))) * 100

	print('Cleaning fixation values below 100 ms removed {0}% of all fixations.\n'.format(round(removed, 2)))

	
	#------------------------------#
	# Create First/Mid Fix Dists.  #
	# for Group and Indiv.         #
	#------------------------------#

	# GROUP
	group_first_fix = fixations['fix_time'][fixations['fix_num']==1]
	group_mid_fix = fixations['fix_time'][(fixations['fix_num']>1) & (fixations['rev_fix_num']>1)]

	# INDIVIDUAL
	subj_first_fix = {}
	subj_mid_fix = {}

	for i in np.unique(fixations.parcode):
	    subj_first_fix[i] = fixations['fix_time'][(fixations['fix_num']==1) & (fixations['parcode']==i)]
	    subj_mid_fix[i] = fixations['fix_time'][(fixations['fix_num']>1) & (fixations['rev_fix_num']>1) & (fixations['parcode']==i)]


	#------------------------------#
	# Create fixation dists.       #
	# based on indiv. params.      #
	#------------------------------#

	N = number

	subj_first_fix_synth = {}
	subj_mid_fix_synth = {}

	print('Creating subject fixation distributions...')

	for i in np.unique(fixations.parcode):
	    x_first = subj_first_fix[i].values.tolist()
	    x_mid = subj_mid_fix[i].values.tolist()
	    
	    [mlk_f, slk_f, tlk_f] =maxLKHD(x_first)
	    [mlk_m, slk_m, tlk_m] =maxLKHD(x_mid)
	    
	    subj_first_fix_synth[i] = [exg_rvs(mlk_f, slk_f, tlk_f) for ii in xrange(N)]
	    subj_mid_fix_synth[i] = [exg_rvs(mlk_m, slk_m , tlk_m) for ii in xrange(N)]
	    print('{0}...'.format(i))


	#------------------------------#
	# Save distributions file      #
	#------------------------------#

	if not os.path.exists(output_filepath):				# create the directory if it does not already exist
		os.makedirs(output_filepath)

	pickle_out = open(output_filepath + "subj_first_fix_synth.pickle","wb")
	pickle.dump(subj_first_fix_synth, pickle_out)
	pickle_out.close()

	pickle_out = open(output_filepath + "subj_mid_fix_synth.pickle","wb")
	pickle.dump(subj_mid_fix_synth, pickle_out)
	pickle_out.close()







