#!/usr/bin/env python

"""
Multi Attribute Decision Making Task: Utility File

Author: Daniel J Wilson
Contact: daniel.j.wilson@gmail.com
Github: danieljwilson
"""

#------------------#
# PsychoPy Imports #
#------------------#
from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile

#----------------#
# Python Imports #
#----------------#
import numpy as np
import random
from collections import namedtuple
from collections import Counter
from itertools import permutations 


def init_stims_weights(expInfo):
    if expInfo['face_version'] in range(0,2) and expInfo['house_version'] in range(0,2): # test for valid value
        # CREATE namedtuple
        Stimuli = namedtuple('Stimuli', 'face face_val house house_val') 

        #---------#
        # STIMULI #
        #---------#
        # LOAD list of stimuli
        num = range(0,101)                      # number of images
        Stimuli.face = ['faceMorph' + ("{:03}".format(num[i])) + '.jpg' for i in num]
        Stimuli.house = ['houseMorph' + ("{:03}".format(num[i])) + '.jpg' for i in num]

        # SET stimuli values based on good/bad face and house 
        # determined by values of expInfo['face_version']/expInfo['house_version']
        stim_values = np.linspace(-1,1,101)     # list of stimuli values
        Stimuli.face_val = stim_values * (1 - (2 * expInfo['face_version'])) 
        Stimuli.house_val = stim_values * (1 - (2 * expInfo['house_version']))

        # RANDOMIZE stimuli (with replacement)
        Rand_Stimuli = namedtuple('Rand_Stimuli', 'face face_val face_weight house house_val house_weight') 
        
        # create shuffled index of 404 values
        rand_face = list(range(101)) * 4        # mult by 4 gives 404 trials
        np.random.shuffle(rand_face)
        rand_house = list(range(101)) * 4       # mult by 4 gives 404 trials
        np.random.shuffle(rand_house)
        # assign house/face based on shuffled indices
        Rand_Stimuli.face = [Stimuli.face[i] for i in rand_face]
        Rand_Stimuli.face_val = [Stimuli.face_val[i] for i in rand_face]
        Rand_Stimuli.house = [Stimuli.house[i] for i in rand_house]
        Rand_Stimuli.house_val = [Stimuli.house_val[i] for i in rand_house]
        
        return(Rand_Stimuli)
    
    else:
        win = visual.Window([800,600],allowGUI=True,
                        monitor='DELL_donna', units='deg')
        visual.TextStim(win, pos=[0,+3],\
                        text='Check value for face_house_version.\n\nNeeds to be 0 or 1.\n\nPress any key to quit.').draw()
        win.flip()
        event.waitKeys()
        core.quit()

    #---------#
    # WEIGHTS #
    #---------# 
        # A Rand weights 1,2,3
        # B Rand weights 0.1, 0.33, 0.5, 1, 2, 3 10
        
    # CREATE weight arrays
    # 10 elements * 20 = 200 (non [1,1] elements)
    weights_a = (list(permutations([1, 2, 3], 2)) + 2 * [(2,2),(3,3)]) * 20

    # 54 elements * 5 = 270 (non [1,1] elements)
    weights_b = (list(permutations([0.1, 0.33, 0.5, 1, 2, 3, 10], 2))\
                                    + 2 * [(0.1,0.1),(0.33,0.33),(0.5,0.5),(2,2),(3,3),(10,10)]) * 5

    # CHOOSE version of weights
        # create ones and non-ones permutations, join and shuffle
    weights_non_ones = np.array(weights_a)
    weights_ones = np.tile((1,1), (404 - len(weights_non_ones),1))
    weights = np.concatenate((weights_non_ones, weights_ones))
    np.random.shuffle(weights) # inplace
    
    # SET weights
    Rand_Stimuli.face_weight = [val for val in weights[:,0]]
    Rand_Stimuli.house_weight = [val for val in weights[:,1]]
    
    # Use this to check how many occurences of each weight combo
    # Counter(map(tuple, weights))



def test_vals(win, Rand_Stimuli, learn_trial_num, task_trial_num, max_blocks=4, min_accuracy = 0.8):
    # Need to achieve certain average accuracy
    trial_type = 'learn'
    trial_count = np.arange(learn_trial_num)
    trialClock = core.Clock()
    block = 1
    accuracy = 0.

    while accuracy < min_accuracy:
        # initialize random index
        image_index = np.random.randint(0,task_trial_num, learn_trial_num)
        print(image_index)
        
        for trial in trial_count:
            # utils.learn_trial()
                # directions text
                message1 = visual.TextStim(win, pos=[0,+6], height = 0.8,\
                                           text='Enter the value in the format X.XX.')
                message2 = visual.TextStim(win, pos=[0,+5], height = 0.6,\
                                           text='For example: 0.25.')
                message3 = visual.TextStim(win, pos=[0,+4], height = 0.65,\
                                           text='Remember to use a minus sign if it is a negative value...')

                # random image 
                face = visual.ImageStim(win, image='images/faces_a/' + Rand_Stimuli.face[image_index[trial]],\
                                        pos=(0, 0), size=7, units='deg')
                house = visual.ImageStim(win, image='images/houses_a/' + Rand_Stimuli.house[image_index[trial]],\
                                         pos=(0, 0), size=7, units='deg')
                
                # choose house or face
                if trial%2 == 0:
                    image = face
                    actual_val = Rand_Stimuli.face_val[image_index[trial]]
                else:
                    image = house
                    actual_val = Rand_Stimuli.house_val[image_index[trial]]
                
                # init values
                theseKeys=""
                inputText=""
                n = len(theseKeys)
                i = 0
                rt = trialClock.getTime()
                
                # subject input
                while i < n:
                    if theseKeys[i] == 'return' and len(inputText) > 3:
                        # pressing RETURN means time to stop
                        continueRoutine = False
                        break

                    elif theseKeys[i] == 'backspace':
                        inputText = inputText[:-1] #lose the final character
                        i = i + 1

                    elif theseKeys[i] == 'period':
                        inputText  = inputText + "."
                        i = i+1

                    elif theseKeys[i] == 'minus':
                        inputText  = inputText + "-"
                        i = i+1

                    elif theseKeys[i] in ['lshift', 'rshift']:
                        shift_flag = True
                        i = i+1

                    else:
                        if len(theseKeys[i]) ==1:
                            #we only have 1 char so should be a normal key
                            #otherwise it might be 'ctrl' or similar so ignore it
                            if shift_flag:
                                inputText += chr( ord(theseKeys[i]) - ord(' '))
                                shift_flag = False
                            else:
                                inputText += theseKeys[i]
                        i = i + 1
                    entry = visual.TextStim(win, pos=[0,-6], height = 0.8, text='$%.2f' %float(inputText))
                    # draw to screen (loop waiting for characters)
                    message1.draw()
                    message2.draw()
                    message3.draw()
                    image.draw()
                    entry.draw()
                    win.flip()

                # calculate trial error
                input_val = float(inputText) # add logic so if empty
                error = input_val - actual_val
                # calculate accuracy rate
                total_error = total_error + error
                accuracy = 1- (total_error / (trial + 1))
                
                # give feedback 
                message1 = visual.TextStim(win, pos=[0,+6], height = 0.8,\
                                           text='You guessed: %.2f.' %float(input_val))
                message2 = visual.TextStim(win, pos=[0,+5], height = 0.6,\
                                           text='Actual value: %.2f.' %actual_val)
                message1.draw()
                message2.draw()
                win.flip()
                core.wait(2)
                # save values
                
                
                event.clearEvents('mouse')  # only really needed for pygame windows
                   
        if accuracy >=min_accuracy:
            # move to Instructions 2
            trial = next
        elif block >= max_blocks:
            # message that they failed to be accurate enough
            message1 = visual.TextStim(win, pos=[0,+3], height = 0.7,
                                      text='You have not acheived minimum accuracy levels.\n\nPlease stay on this screen and let the research assistant know.')
            message1.draw()
            win.flip()
        else:
            block+=1