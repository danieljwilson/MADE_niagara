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


##################################
# 0.3 INIT STIM VALS AND WEIGHTS #
##################################

def init_stims_weights(expInfo):
    """
    Function to initialize two namedtuples. The first contains all stimuli
    associated with their values. The second is a randomized version that contains
    all trials for the experiment, along with the addition of weights.

    The weights version can be set by choosing either `weights_a` or `weights_b`

    Keyword arguments:
    expInfo -- dict, contains experimental parameters, set in main script

    Returns:
    Stimuli -- namedtuple, contains all stimuli associated with their values
    Rand_Stimuli -- namedtuple, contains all trials: stimuli, values and weights
    """
    if expInfo['face_version'] in range(0, 2) \
            and expInfo['house_version'] in range(0, 2):  # test for valid value
        # CREATE namedtuple
        Stimuli = namedtuple('Stimuli', 'face face_val house house_val')

        #---------#
        # STIMULI #
        #---------#
        # LOAD list of stimuli
        num = range(0, 101)                      # number of images
        Stimuli.face = ['faceMorph' + ("{:03}".format(num[i])) + '.jpg' for i in num]
        Stimuli.house = ['houseMorph' + ("{:03}".format(num[i])) + '.jpg' for i in num]

        # SET stimuli values based on good/bad face and house
        # determined by values of expInfo['face_version']/expInfo['house_version']
        stim_values = np.linspace(-1, 1, 101)     # list of stimuli values
        Stimuli.face_val = stim_values * (1 - (2 * expInfo['face_version']))
        Stimuli.house_val = stim_values * (1 - (2 * expInfo['house_version']))

        # RANDOMIZE stimuli (with replacement)
        Rand_Stimuli = namedtuple(
            'Rand_Stimuli', 'face face_val face_weight house house_val house_weight')

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

        #---------#
        # WEIGHTS #
        #---------#
            # A Rand weights 1,2,3
            # B Rand weights 0.1, 0.33, 0.5, 1, 2, 3 10

        # CREATE weight arrays
        # 10 elements * 20 = 200 (non [1,1] elements)
        weights_a = (list(permutations([1, 2, 3], 2)) + 2 * [(2, 2), (3, 3)]) * 20

        # 54 elements * 5 = 270 (non [1,1] elements)
        weights_b = (list(permutations([0.1, 0.33, 0.5, 1, 2, 3, 10], 2))
                     + 2 * [(0.1, 0.1), (0.33, 0.33), (0.5, 0.5), (2, 2), (3, 3), (10, 10)]) * 5

        # CHOOSE version of weights
        # create ones and non-ones permutations, join and shuffle
        weights_non_ones = np.array(weights_a)
        weights_ones = np.tile((1, 1), (404 - len(weights_non_ones), 1))
        weights = np.concatenate((weights_non_ones, weights_ones))
        np.random.shuffle(weights)  # inplace

        # SET weights
        Rand_Stimuli.face_weight = [val for val in weights[:, 0]]
        Rand_Stimuli.house_weight = [val for val in weights[:, 1]]

        # Use this to check how many occurences of each weight combo
        # Counter(map(tuple, weights))

        return(Stimuli, Rand_Stimuli)

    else:
        win = visual.Window([800, 600], allowGUI=True,
                            monitor='DELL_donna', units='deg')
        visual.TextStim(win, pos=[0, +3],
                        text='Check value for face_house_version.\n\nNeeds to be 0 or 1.\n\nPress any key to quit.').draw()
        win.flip()
        event.waitKeys()
        core.quit()


def show(win):
    """
    Function to simply add the "continue..." text at the bottom of the screen,
    then update the screen and wait for a keystroke to advance.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    """
    visual.TextStim(win, pos=[0, -7], height=0.5,
                    text='Press any key to continue...').draw()
    win.flip()
    event.waitKeys()

####################
# 1.1 INSTRUCTIONS #
####################


def instructions_1(win, Stimuli):
    """
    Function to show instructions for "1 LEARN VALUES" section of MADE.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    Stimuli -- namedtuple that contains house and face image/value info
    """

    #################### INTRO ####################
    # section
    visual.TextStim(win, pos=[0, 0], units='norm', height=0.28, color=-1,
                    text='Section 1').draw()
    show(win)
    # intro1
    visual.TextStim(win, pos=[0, 0], height=0.8,
                    text='Hello.\n\nIn this experiment we are interested in how people make decisions.').draw()
    show(win)
    # intro2
    visual.TextStim(win, pos=[0, 0], height=0.8,
                    text='The experiment should take about 60 minutes.').draw()
    show(win)
    # intro3
    visual.TextStim(win, pos=[0, 0], height=0.7,
                    text='In the experiment, you will see images of FACES and HOUSES that have different meanings.\n\nSome images are associated with WINNING money. Others are associated with LOSING money.\n\nYou will first learn to associate different images with their appropriate outcomes (values).').draw()
    show(win)
    # intro4
    visual.TextStim(win, pos=[0, 0], height=0.7,
                    text='You will then be tested to make sure that you have learned the values associated with the different FACES and HOUSES.').draw()
    show(win)
    # intro5
    visual.TextStim(win, pos=[0, 0], height=0.7,
                    text='Make sure you pay attention as you learn about the images, since you will eventually be making choices about them in which you could end up winning or losing money.').draw()
    show(win)

    #################### FACES ####################
    # face screen
    visual.TextStim(win, pos=[0, 0],
                    text='First we will show you the faces.').draw()
    show(win)
    # face screen val 0
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[0],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='This face is worth ${:.2f}'.format(Stimuli.face_val[0])).draw()
    visual.TextStim(win, pos=[0, -5], height=0.6,
                    text='*You will need to remember this for the experiment.').draw()
    show(win)
    # face screen val 100
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[100],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='This face is worth ${:.2f}'.format(Stimuli.face_val[100])).draw()
    show(win)
    # face screen morph1
    visual.TextStim(win, pos=[0, 0],
                    text='In the experiment the faces will be blended together (morphed).').draw()
    show(win)
    # face screen morph2
    visual.TextStim(win, pos=[0, 0],
                    text='The value will depend on how MUCH of each face is present in the morph.').draw()
    show(win)
    # face screen morph3
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[50],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='For example, this is a 50/50 morph, so the value is $0.00.').draw()
    show(win)
    # face screen morph4
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[60],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='This is a 60/40 morph, so the value is ${:.2f}.'.format(Stimuli.face_val[60])).draw()
    show(win)
    # face screen morph5
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[75],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='This is a 75/25 morph, so the value is ${:.2f}.'.format(Stimuli.face_val[75])).draw()
    show(win)

    #################### HOUSES ####################
    # house screen
    visual.TextStim(win, pos=[0, 0],
                    text='There are also houses').draw()
    show(win)
    # house screen val 0
    visual.ImageStim(win, image='images/houses_a/' + Stimuli.house[0],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='This house is worth ${:.2f}'.format(Stimuli.house_val[0])).draw()
    show(win)
    # house screen val 100
    visual.ImageStim(win, image='images/houses_a/' + Stimuli.house[100],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='This house is worth ${:.2f}'.format(Stimuli.house_val[100])).draw()
    show(win)
    # house screen val 100
    visual.ImageStim(win, image='images/houses_a/' + Stimuli.house[50],
                     pos=(0, 0), size=7, units='deg').draw()
    visual.TextStim(win, pos=[0, +6],
                    text='This is an example of a 50/50 house morph, worth $0.00').draw()
    show(win)

    #################### TEST EXPLANATION ####################
    # test explanation1
    visual.TextStim(win, pos=[0, 0],
                    text='You will now practice estimating the value of face and house morphs...').draw()
    show(win)
    # test explanation2
    visual.TextStim(win, pos=[0, 0], height=0.8,
                    text='There will be 20 trials.\n\nYour goal is to achieve an accuracy rate of 80% or better.\n\n(This means that you are within $0.20 of the actual value on average)').draw()
    show(win)
    # test explanation3
    visual.TextStim(win, pos=[0, 0], height=0.8,
                    text='Not to worry, you have 4 rounds to achieve this accuracy level.').draw()
    show(win)
    # test explanation4
    visual.TextStim(win, pos=[0, 0], height=1.5,
                    text='Start...').draw()
    show(win)

######################
# 1.2: STIM VAL TEST #
######################


def test_vals(win, expInfo, Rand_Stimuli, task_trial_num, trial_num, max_blocks=4, min_accuracy=80, trial_type='learn'):
    """
    Function to test how well subjects are learning the values of the stimuli.

    Trial information is saved to `datafile`.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    expInfo -- dict, contains experimental parameters, set in main script
    Rand_Stimuli -- namedtuple, contains all trials: stimuli, values and weights
    learn_trial_num -- int, number of trials in this learning task (per block)
    task_trial_num -- int, number of trials in the actual task
    max_blocks -- int, how many times through before we give up on subjects
    min_accuracy -- int, how accurate do subjects need to be on average
    trial_type -- str, specify whether these are learn or recall trials
    """

    # Need to achieve certain average accuracy
    trial_count = np.arange(trial_num)

    accuracy = 0.
    block = 1
    trialClock = core.Clock()

    #--------------------#
    # Create data file   #
    #--------------------#
    if trial_type == 'learn':
        fileName = 'subject_data/' + str(expInfo['subject']) + '_learning_' + expInfo['dateStr']
    if trial_type == 'recall':
        fileName = 'subject_data/' + str(expInfo['subject']) + '_recall_' + expInfo['dateStr']

    dataFile = open(fileName + '.csv', 'w')  # a simple text file with 'comma-separated-values'
    dataFile.write(
        'date, psychopy_version, exp_version, face_version, house_version, subject, block, trial, image, actual_val, input_val, rt, error, total_error, accuracy, min_accuracy\n')

    #--------------#
    # Run trials   #
    #--------------#
    while accuracy < min_accuracy:
        # initialize random index
        image_index = np.random.randint(0, task_trial_num, trial_num)

        # reset total error
        total_error = 0

        for trial in trial_count:
            # utils.learn_trial()
            # directions text
            message1 = visual.TextStim(win, pos=[0, +7], height=0.8,
                                       text='Enter the value in the format X.XX')
            message2 = visual.TextStim(win, pos=[0, +6], height=0.7,
                                       text='For example: 0.25.')
            message3 = visual.TextStim(win, pos=[0, +5], height=0.5,
                                       text='Remember to use a minus sign if it is a negative value...')
            message4 = visual.TextStim(win, pos=[0, -7], height=0.5,
                                       text='Press return/enter to submit answer')
            round = visual.TextStim(win, pos=[-8, -7], height=0.8, color=[-0.5, -0.5, 1],
                                    text='Round: {}'.format(block))
            round = visual.TextStim(win, units='norm', pos=[0, -.95], height=0.06,
                                    color=[-0.5, -0.5, 1],
                                    text='Round: {}'.format(block))
            bottom_bar = visual.Rect(win, units='norm', width=2, height=.13,
                                     pos=[0, -.95], color=-1)

            # random image
            face = visual.ImageStim(win, image='images/faces_a/' + Rand_Stimuli.face[image_index[trial]],
                                    pos=(0, 0), size=7, units='deg')
            house = visual.ImageStim(win, image='images/houses_a/' + Rand_Stimuli.house[image_index[trial]],
                                     pos=(0, 0), size=7, units='deg')

            # choose house or face
            if trial % 2 == 0:
                image = face
                image_save = Rand_Stimuli.face[image_index[trial]]
                actual_val = Rand_Stimuli.face_val[image_index[trial]]
            else:
                image = house
                image_save = Rand_Stimuli.house[image_index[trial]]
                actual_val = Rand_Stimuli.house_val[image_index[trial]]

            # init values
            inputText = ""
            keystroke = [""]
            rt = trialClock.getTime()

            # subject input (using an xor loop)
            while not(keystroke[0] == 'return' and len(inputText) > 3):
                # create entry val text if inputText not empty
                if len(inputText) == 0:
                    entry = visual.TextStim(win, pos=[0, -5], height=0.8,
                                            text='_.__')
                elif len(inputText) > 0:
                    entry = visual.TextStim(win, pos=[0, -5], height=0.8,
                                            text='%s' % inputText)

                # draw to screen
                bottom_bar.draw()
                entry.draw()
                message1.draw()
                message2.draw()
                message3.draw()
                message4.draw()
                image.draw()
                round.draw()
                win.flip()

                keystroke = event.waitKeys(
                    keyList=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'minus', 'period', 'return', 'backspace'])

                # delete last character if there is at least one character
                if keystroke[0] == 'backspace' and len(inputText) > 0:
                    inputText = inputText[:-1]

                elif keystroke[0] == 'period' and inputText.count('.')<1:
                    inputText = inputText + "."

                elif keystroke[0] == 'minus' and len(inputText) < 1:
                    inputText = inputText + "-"

                elif keystroke[0] in ['lshift', 'rshift']:
                    shift_flag = True

                else:
                    if len(keystroke) == 1 and len(inputText)<5:
                        # we only have 1 char so should be a normal key
                        # otherwise it might be 'ctrl' or similar so ignore it
                        if keystroke[0] != 'return' and keystroke[0] != 'backspace' and keystroke[0] != 'minus' and keystroke[0] != 'period':
                            inputText += keystroke[0]
            # calculate rt (not really important here)
            rt = trialClock.getTime() - rt
            # calculate trial error
            input_val = float(inputText)  # add logic so if empty
            error = abs(input_val - actual_val)

            # calculate accuracy rate
            total_error = total_error + error
            accuracy = (1 - (total_error / (trial + 1))) * 100

            if trial_type == 'learn':
                # give feedback
                message1 = visual.TextStim(win, pos=[0, +6], height=0.8,
                                           text='You guessed: {:.2f}'.format(float(input_val)))
                message2 = visual.TextStim(win, pos=[0, +5], height=0.6,
                                           text='Actual value: {:.2f}'.format(actual_val))
                message3 = visual.TextStim(win, pos=[0, 0], height=0.9,
                                           text='Accuracy rate: {:.2f}'.format(accuracy))
                message4 = visual.TextStim(win, pos=[0, -2], height=0.6,
                                           text='Goal accuracy is {:.2f}'.format(min_accuracy))
                message5 = visual.TextStim(win, pos=[0, -7], height=0.5,
                                           text='Press any key to continue...')
                # draw to screen
                message1.draw()
                message2.draw()
                message5.draw()
                win.flip()
                event.waitKeys()
                message1.draw()
                message2.draw()
                message3.draw()
                message4.draw()
                message5.draw()
                win.flip()
                event.waitKeys()
            # save values
            dataFile.write('%s,%s,%s,%i,%i,%i,%i,%i,%s,%.2f,%.2f,%.3f,%.2f,%.2f,%.2f,%.2f\n' % (expInfo['dateStr'],
                                                                                                expInfo['psychopy_version'],
                                                                                                expInfo['exp_version'],
                                                                                                expInfo['face_version'],
                                                                                                expInfo['house_version'],
                                                                                                expInfo['subject'],
                                                                                                block,
                                                                                                trial,
                                                                                                image_save,
                                                                                                actual_val,
                                                                                                input_val,
                                                                                                rt,
                                                                                                error,
                                                                                                total_error,
                                                                                                accuracy,
                                                                                                min_accuracy))

        # have we achieved min accuracy over the block?
        if accuracy >= min_accuracy and trial_type=='learn':
            # done writing to file
            dataFile.close()
            visual.TextStim(win, pos=[0, 0], height=0.9,
                            text='Accuracy achieved...ready for task practice.').draw()
            show(win)
        if block >= max_blocks and trial_type=='learn':
            # message that they failed to be accurate enough
            message1 = visual.TextStim(win, pos=[0, +3], height=0.7,
                                       text='You have not achieved minimum accuracy levels.\n\nPlease stay on this screen and let the research assistant know.')
            message1.draw()
            win.flip()
            # done writing to file
            dataFile.close()
            core.wait(10)
            event.waitKeys()
            core.quit()
        if block >= max_blocks and trial_type=='recall':
            # done writing to file
            dataFile.close()
            visual.TextStim(win, pos=[0, 0], height=0.9,
                            text='Section 4 Complete\n\nGreat job!').draw()
            show(win)
        else:
            block += 1


####################
# 2.1 INSTRUCTIONS #
####################


def instructions_2(win, Stimuli):
    """
    Function to show instructions for "2 PRACTICE" section of MADE.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    Stimuli -- namedtuple that contains house and face image/value info
    """

    #################### INTRO ####################
    # section
    visual.TextStim(win, pos=[0, 0], units='norm', height=0.28, color=-1,
                    text='Section 2').draw()
    show(win)
    # explain_combo1
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='In the experiment you will need to consider the SUM of the values of TWO images - one FACE and one HOUSE - at the same time.').draw()
    show(win)
    # explain_combo2
    visual.TextStim(win, pos=[0, +6], height=0.65,
                    text='For example this face has a value of ${0:.2f}\nThe house has a value of ${1:.2f}\nTherefore, their combined value is:\n${0:.2f} (face) + ${1:.2f} (house) = ${2:.2f}'.format(Stimuli.face_val[25], Stimuli.house_val[60], Stimuli.face_val[25] + Stimuli.house_val[60])).draw()
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[25],
                     pos=(-4.5, 0), size=7, units='deg').draw()
    visual.ImageStim(win, image='images/houses_a/' + Stimuli.house[60],
                     pos=(4.5, 0), size=7, units='deg').draw()
    show(win)

    # explain_combo3
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='If the COMBINED value is positive, you should ACCEPT.\n\nIf the COMBINED value is negative, you should REJECT.').draw()
    show(win)
    # explain_combo4
    visual.TextStim(win, pos=[0, +6], height=0.65,
                    text='Another example: the face has a value of ${0:.2f}\nThe house has a value of ${1:.2f}\nTherefore, their combined value is:\n${0:.2f} (face) + ${1:.2f} (house) = ${2:.2f}'.format(Stimuli.face_val[80], Stimuli.house_val[35], Stimuli.face_val[80] + Stimuli.house_val[35])).draw()
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[80],
                     pos=(-4.5, 0), size=7, units='deg').draw()
    visual.ImageStim(win, image='images/houses_a/' + Stimuli.house[35],
                     pos=(4.5, 0), size=7, units='deg').draw()
    show(win)
    # explain_combo5
    visual.TextStim(win, pos=[0, 0], height=1.4,
                    text='HOWEVER...').draw()
    show(win)
    # explain_combo6
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='There will also be MULTIPLIERS associated with each image.').draw()
    show(win)
    # explain_combo7
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='Okay, but what is a MULTIPLIER??').draw()
    show(win)
    # explain_combo8
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='A multiplier CHANGES the value of the face or house it is associated with.').draw()
    show(win)
    # explain_combo9
    visual.TextStim(win, pos=[0, +6], height=0.65,
                    text='For example, normally this face has a value of ${0:.2f}\n\nBut with a 2X multiplier it is worth ${1:.2f}\nWith a 0.1X multiplier it would be worth: ${2:.2f}'.format(Stimuli.face_val[20],
                                                                                                                                                                                               2 * Stimuli.face_val[20],
                                                                                                                                                                                               0.1 * Stimuli.face_val[20])).draw()
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[20],
                     pos=(0, 0), size=7, units='deg').draw()
    show(win)
    # explain_combo10
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='This can CHANGE your decision.').draw()
    show(win)
    # explain_combo11
    visual.TextStim(win, pos=[0, +4], height=0.65,
                    text='For example, normally this combo was worth ${2:.2f}\n(face: ${0:.2f}, house: ${1:.2f})\n\nBUT if there is a 3X HOUSE multiplier and only a 1X FACE multiplier the actual value is ${3:.2f}\nSince FACE value remains ${0:.2f} (1X ${0:.2f}), but house value is now ${4:.2f} (3X ${1:.2f})'.format(Stimuli.face_val[80],
                                                                                                                                                                                                                                                                                                                             Stimuli.house_val[35],
                                                                                                                                                                                                                                                                                                                             Stimuli.face_val[80] + Stimuli.house_val[35],
                                                                                                                                                                                                                                                                                                                             Stimuli.face_val[80] + 3 * Stimuli.house_val[35],
                                                                                                                                                                                                                                                                                                                             3 * Stimuli.house_val[35])).draw()
    visual.ImageStim(win, image='images/faces_a/' + Stimuli.face[80],
                     pos=(-4.5, -2.5), size=7, units='deg').draw()
    visual.ImageStim(win, image='images/houses_a/' + Stimuli.house[35],
                     pos=(4.5, -2.5), size=7, units='deg').draw()
    show(win)
    # explain_combo12
    visual.TextStim(win, pos=[0, 0], height=1.4,
                    text='ALSO...').draw()
    visual.TextStim(win, pos=[0, -2], height=0.5,
                    text='(almost done)').draw()
    show(win)
    # explain_combo13
    visual.TextStim(win, pos=[0, +4], height=0.8,
                    text='You will see the multipliers on screen the whole time, BUT you will need to press the LEFT and RIGHT arrow keys to view each image.').draw()
    visual.ImageStim(win, image='images/instructions/left_right_arrows.png',
                     pos=(0, -2.5), size=7, units='deg').draw()
    show(win)
    # explain_combo14
    visual.TextStim(win, pos=[0, +4], height=0.8,
                    text='You will be able to ACCEPT with the UP arrow or REJECT with the DOWN arrow.').draw()
    visual.ImageStim(win, image='images/instructions/up_down_arrows.png',
                     pos=(0, -2.5), size=7, units='deg').draw()
    show(win)
    # explain_combo12
    visual.TextStim(win, pos=[0, +1], height=0.9,
                    text='That is a lot to take in.\n\nSo, time to practice...').draw()
    visual.TextStim(win, pos=[0, -3], height=0.75,
                    text='You will have 25 practice trials\nto get the hang of it.').draw()
    show(win)


#########################
# 2.2 TRIALS (PRACTICE) #
#########################

def task_trials(win, expInfo, Rand_Stimuli, practice_trial_num,
                task_trial_num, blocks, trial_type):
    """
    Function that shows two stimuli and requires subjects to either 'accept' or
    'reject'. Feedback is provided after the choice is made.

    Trial information is saved to `dataFile` which outputs a csv in the
    `subject_data` folder.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    expInfo -- dict, contains experimental parameters, set in main script
    Rand_Stimuli -- namedtuple, contains all trials: stimuli, values and weights
    practice_trial_num -- int, number of trials in this learning task (per block)
    task_trial_num -- int, number of trials in the actual task
    blocks -- int, how many blocks of trials (usually 1 for practice)
    trial_type -- str, are these trials `practice` or the actual `task`?
    """

    # set trials/block
    if trial_type == 'practice':
        trial_count = np.arange(practice_trial_num).reshape((blocks,practice_trial_num))
    elif trial_type == 'task':
        # note that `blocks` should divide evenly into `task_trial_num`
        trial_count = np.arange(task_trial_num).reshape((blocks, int(task_trial_num/blocks)))
    else:
        print('trial_type must be either practice or task...')
        core.quit()
    # init earnings
    earnings = 0
    # timing object
    trialClock = core.Clock()

    #--------------------#
    # Create data file   #
    #--------------------#
    if trial_type == 'practice':
        fileName = 'subject_data/' + str(expInfo['subject']) + '_practice_' + expInfo['dateStr']
    if trial_type == 'task':
        fileName = 'subject_data/' + str(expInfo['subject']) + '_trials_' + expInfo['dateStr']

    dataFile = open(fileName + '.csv', 'w')  # a simple text file with 'comma-separated-values'
    dataFile.write(
        'date, psychopy_version, exp_version, face_version, house_version, left, subject, block, trial, rt, response, correct, summed_val_total, face_image, face_val_base, face_mult, face_val_total, house_image, house_val_base, house_mult, house_val_total, fix_num, fix_rt, fix_stim\n')

    #--------------#
    # Run trials   #
    #--------------#
    if trial_type == 'practice':
        # initialize random index
        image_index = np.random.randint(0, task_trial_num, practice_trial_num)
    # iterate through blocks
    for block in range(blocks):
        # iterate through trials
        for trial in trial_count[block]:
            #################### SET STIM AND VALUES ####################
            if trial_type == 'practice':
                # images
                face = visual.ImageStim(win, image='images/faces_a/' + Rand_Stimuli.face[image_index[trial]],
                                        pos=(0, -3), size=8, units='deg')
                house = visual.ImageStim(win, image='images/houses_a/' + Rand_Stimuli.house[image_index[trial]],
                                         pos=(0, -3), size=8, units='deg')
                # stim base vals
                face_val_base = Rand_Stimuli.face_val[image_index[trial]]
                house_val_base = Rand_Stimuli.house_val[image_index[trial]]
                # multiplier vals
                face_mult = Rand_Stimuli.face_weight[image_index[trial]]
                house_mult = Rand_Stimuli.house_weight[image_index[trial]]

            if trial_type == 'task':
                # images
                face = visual.ImageStim(win, image='images/faces_a/' + Rand_Stimuli.face[trial],
                                        pos=(0, -3), size=8, units='deg')
                house = visual.ImageStim(win, image='images/houses_a/' + Rand_Stimuli.house[trial],
                                         pos=(0, -3), size=8, units='deg')
                # stim base vals
                face_val_base = Rand_Stimuli.face_val[trial]
                house_val_base = Rand_Stimuli.house_val[trial]
                # multiplier vals
                face_mult = Rand_Stimuli.face_weight[trial]
                house_mult = Rand_Stimuli.house_weight[trial]

            # stim total vals
            face_val_total = face_val_base * face_mult
            house_val_total = house_val_base * house_mult
            # summed val
            summed_val_total = face_val_total + house_val_total

            #################### ONSCREEN TEXT/DESIGN ####################
            accept_text = visual.TextStim(win, units='norm', pos=[0, .9], height=0.085,
                                       text='Do you accept or reject this combo?')
            left_right_text = visual.TextStim(win, units='norm', pos=[0, .75], height=0.07,
                                       text='Press LEFT/RIGHT to view the options')
            face_text = visual.TextStim(win, pos=[-4, 3.5], height=1.1, color=-1,
                                       text='FACE')
            house_text = visual.TextStim(win, pos=[+4, 3.5], height=1.1, color=-1,
                                       text='HOUSE')
            face_mult_text = visual.TextStim(win, pos=[-4, 4], height=1.1, color=1,
                                       text=str(face_mult) + 'X')
            house_mult_text = visual.TextStim(win, pos=[4, 4], height=1.1, color=1,
                                       text=str(house_mult) + 'X')
            up_down_text = visual.TextStim(win, units='norm', pos=[0, -.95], height=0.06,
                                       text='Press UP to accept, DOWN to reject')

            # onscreen shapes
            left_side_color = visual.Rect(win, units='norm', width=1, height=2,
                                          pos=[-0.5,0], color=(-0.02,.18,1))
            right_side_color = visual.Rect(win, units='norm', width=1, height=2,
                                          pos=[0.5,0], colorSpace='rgb255',
                                          color=(1,0.35,-0.02))
            bottom_bar = visual.Rect(win, units='norm', width=2, height=.13,
                                     pos=[0, -.95], color=-1)
            top_bar = visual.Rect(win, units='norm', width=2, height=.3, pos=[0, .85],
                                  color=-1)
            left_mult_box = visual.Rect(win, width=3, height=1.5, pos=[-5, 3], color=-1)
            right_mult_box = visual.Rect(win, width=3, height=1.5, pos=[5, 3], color=-1)
            left_select_box = visual.Rect(win, width=9, height=9, pos=[0, -3],
                                          color=(-0.02,.18,1))
            right_select_box = visual.Rect(win, width=9, height=9, pos=[0, -3],
                                          color=(1,0.35,-0.02))

            #################### SET STIM SIDES ####################
            if expInfo['left'] == 'house':
                left_stim = house
                right_stim = face
                left_mult = house_mult_text
                right_mult = face_mult_text
                left_label = house_text
                right_label = face_text
            else:
                left_stim = face
                right_stim = house
                left_mult = face_mult_text
                right_mult = house_mult_text
                left_label = face_text
                right_label = house_text

            left_label.setPos([-5, 4.5])
            left_mult.setPos([-5, 3])
            right_label.setPos([5, 4.5])
            right_mult.setPos([5, 3])


            #################### START TRIAL ####################
            # init values
            inputText = ""
            this_trial = ""
            previous = 'weights'
            STIM_LIST = [left_side_color, right_side_color, left_mult_box,
                         right_mult_box, bottom_bar, top_bar,
                         accept_text, left_right_text, up_down_text, left_mult,
                         right_mult, left_label, right_label]

            fix_array = []
            rt_array = []
            fix_num = 0

            for s in STIM_LIST:
                s.draw()
            # show
            win.flip()

            # start timer
            start_rt = trialClock.getTime()
            fix_start_time = trialClock.getTime()

            # subject input (using an xor loop)
            while this_trial!= 'over':

                # # show only weights if no key pressed
                # if len(keystroke) == 0:
                #     for s in STIM_LIST:
                #         s.draw()
                #     # show
                #     win.flip()

                keystroke = event.waitKeys(
                    keyList=['left', 'right', 'up', 'down'])
                # LEFT
                if len(keystroke) >0 and keystroke[0] == 'left':
                    # check if last keystroke was also left...
                    if previous == 'left':
                        previous = 'left'

                    else:
                        fix_rt = trialClock.getTime() - fix_start_time
                        fix_array.append(previous)
                        rt_array.append(fix_rt)
                        previous = 'left'
                        fix_start_time = trialClock.getTime()

                    for s in STIM_LIST:
                        s.draw()
                    # add stim
                    left_select_box.draw()
                    left_stim.draw()
                    # show
                    win.flip()
                    #event.waitKeys()

                # RIGHT
                elif len(keystroke)>0 and keystroke[0] == 'right':
                    # check if last keystroke was also right...
                    if previous == 'right':
                        previous = 'right'

                    else:
                        fix_rt = trialClock.getTime() - fix_start_time
                        fix_array.append(previous)
                        rt_array.append(fix_rt)
                        previous = 'right'
                        fix_start_time = trialClock.getTime()

                    for s in STIM_LIST:
                        s.draw()
                    # add stim
                    right_select_box.draw()
                    right_stim.draw()
                    # show
                    win.flip()
                    #event.waitKeys()

                # Choice
                elif keystroke[0] == 'up':
                    total_rt = trialClock.getTime() - start_rt
                    fix_rt = trialClock.getTime() - fix_start_time
                    fix_array.append(previous)
                    rt_array.append(fix_rt)
                    # end loop
                    this_trial = 'over'
                    response = 1
                    if summed_val_total >= 0.:
                        correct_response = 1
                        earnings = earnings + summed_val_total
                    else:
                        correct_response = 0
                        earnings = earnings + summed_val_total

                elif keystroke[0] == 'down':
                    total_rt = trialClock.getTime() - start_rt
                    fix_rt = trialClock.getTime() - fix_start_time
                    fix_array.append(previous)
                    rt_array.append(fix_rt)
                    # end loop
                    this_trial = 'over'
                    response = 0
                    if summed_val_total <= 0.:
                        correct_response = 1
                        earnings+=0
                    else:
                        correct_response = 0
                        earnings = earnings + summed_val_total

            # give feedback
            if correct_response == 1:
                feedback = visual.TextStim(win, units='norm', pos=[0, 0], height=.2, color=(0,1,0),
                                           text='CORRECT')
            elif correct_response == 0:
                feedback = visual.TextStim(win, units='norm', pos=[0, 0], height=.2, color=(1,0,0),
                                           text='INCORRECT')

            feedback.draw()
            bottom_bar.draw()
            visual.TextStim(win, units='norm', pos=[0, -.95], height=0.06,
                            text='Press SPACE to continue...').draw()
            win.flip()
            event.waitKeys(keyList=['space'])

            # save values
            if trial_type == 'practice':
                for fix_num in range(len(fix_array)):
                    dataFile.write(
                        '%s,%s,%s,%i,%i,%s,%i,%i,%i,%.3f,%i,%i,%.2f,%s,%.2f,%.2f,%.2f,%s,%.2f,%.2f,%.2f,%i,%.3f,%s\n' % (expInfo['dateStr'],
                                                                                                                         expInfo['psychopy_version'],
                                                                                                                         expInfo['exp_version'],
                                                                                                                         expInfo['face_version'],
                                                                                                                         expInfo['house_version'],
                                                                                                                         expInfo['left'],
                                                                                                                         expInfo['subject'],
                                                                                                                         block,
                                                                                                                         trial,
                                                                                                                         total_rt,
                                                                                                                         response,
                                                                                                                         correct_response,
                                                                                                                         summed_val_total,
                                                                                                                         Rand_Stimuli.face[image_index[trial]],
                                                                                                                         face_val_base,
                                                                                                                         face_mult,
                                                                                                                         face_val_total,
                                                                                                                         Rand_Stimuli.house[image_index[trial]],
                                                                                                                         house_val_base,
                                                                                                                         house_mult,
                                                                                                                         house_val_total,
                                                                                                                         fix_num,
                                                                                                                         rt_array[fix_num],
                                                                                                                         fix_array[fix_num]))

            if trial_type == 'task':
                for fix_num in range(len(fix_array)):
                    dataFile.write(
                        '%s,%s,%s,%i,%i,%s,%i,%i,%i,%.3f,%i,%i,%.2f,%s,%.2f,%.2f,%.2f,%s,%.2f,%.2f,%.2f,%i,%.3f,%s\n' % (expInfo['dateStr'],
                                                                                                                         expInfo['psychopy_version'],
                                                                                                                         expInfo['exp_version'],
                                                                                                                         expInfo['face_version'],
                                                                                                                         expInfo['house_version'],
                                                                                                                         expInfo['left'],
                                                                                                                         expInfo['subject'],
                                                                                                                         block,
                                                                                                                         trial,
                                                                                                                         total_rt,
                                                                                                                         response,
                                                                                                                         correct_response,
                                                                                                                         summed_val_total,
                                                                                                                         Rand_Stimuli.face[trial],
                                                                                                                         face_val_base,
                                                                                                                         face_mult,
                                                                                                                         face_val_total,
                                                                                                                         Rand_Stimuli.house[trial],
                                                                                                                         house_val_base,
                                                                                                                         house_mult,
                                                                                                                         house_val_total,
                                                                                                                         fix_num,
                                                                                                                         rt_array[fix_num],
                                                                                                                         fix_array[fix_num]))

            if trial_type == 'practice':
                # report actual summed val
                visual.TextStim(win, units='norm', pos=[0, 0.5], height=0.1, color=(-1),
                                text='Actual value was: ${:.2f}'.format(summed_val_total)).draw()

                # randomize position that these are shown (which first) by resetting pos after...
                house_val_text = visual.TextStim(win, units='norm', pos=[0, -0.07], height=0.08,
                                           text='House Value: {} x ${:.2f} = ${:.2f}'.format(house_mult, house_val_base, house_val_total)).draw()
                face_val_text = visual.TextStim(win, units='norm', pos=[0,0.07], height=0.08,
                                           text='Face Value: {} x ${:.2f} = ${:.2f}'.format(face_mult, face_val_base, face_val_total)).draw()
            if trial_type == 'task':
                # report actual summed val
                visual.TextStim(win, units='norm', pos=[0, 0], height=0.12, color=(-1),
                                text='Actual value: ${:.2f}'.format(summed_val_total)).draw()
                # report actual summed val
                visual.Rect(win, units='norm', pos=[0,-0.5], width=0.7, height=0.15,
                            color=-1).draw()
                visual.TextStim(win, units='norm', pos=[0, -0.5], height=0.12, color=(1),
                                text='Earnings: ${:.2f}'.format(earnings)).draw()

            bottom_bar.draw()
            visual.TextStim(win, units='norm', pos=[0, -.95], height=0.06,
                            text='Press SPACE to continue...').draw()
            win.flip()
            event.waitKeys(keyList=['space'])
        # end of block
        if trial_type == 'practice': # this assumes only 1 block in practice
            continue
        else:
            visual.TextStim(win, units='norm', pos=[0, 0], height=0.1, color=(-1),
                            text='You have finished block {}.\n\nWhen you are ready press SPACE to continue.'.format(block+1)).draw()

            win.flip()
            event.waitKeys(keyList=['space'])
    dataFile.close()
    return(earnings)

####################
# 3.1 INSTRUCTIONS #
####################


def instructions_3(win, task_trial_num, blocks):
    """
    Function to show instructions for "3 TASK" section of MADE.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    task_trial_num -- int, total number of trials in experiment
    blocks -- int, number of blocks in experiment
    """
    #################### CONGRATS ####################
    # practice_complete
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='Well done, you have completed your practice.\n\nIf you have ANY QUESTIONS please ask the RA for help NOW, before starting the task.').draw()
    show(win)
    #################### THE TASK ####################
    # section
    visual.TextStim(win, pos=[0, 0], units='norm', height=0.28, color=-1,
                    text='Section 3').draw()
    show(win)
    # start_task
    visual.TextStim(win, pos=[0, 0], height=1.5,
                    text='You will now start the task.').draw()
    show(win)
    # winning_money1
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='Remember, you have the chance to win or lose REAL money on each trial.\n\nThe more accurate you are, the more money you can earn.').draw()
    show(win)
    # winning_money2
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='We will keep track of your total earnings.\n\nYou will be paid at the end at the rate of 1:10.\n\nThis means if you earned $150, for example, you would be paid $15.').draw()
    show(win)

    # winning_money3
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='Top players have earned up to $20.').draw()
    show(win)

    # trials1
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='There will be {} blocks of trials, with {} trials in each block.\n\nYou will have the chance to take a short break after each block.'.format(blocks, int(task_trial_num/blocks))).draw()
    show(win)
    # start1
    visual.TextStim(win, pos=[0, 0], height=1.2,
                    text='When you are ready to START press any key to begin...').draw()
    show(win)

"""
Better to do cumulative amount or "levels"?
    (change screen color and show bonus amount)
    0.55 or below = $0 bonus
    0.55-0.65 = $2 bonus
    0.65-0.75 = $5 bonus
    0.75-0.85 = $9 bonus
    0.85-0.95 = $14 bonus
    0.95-1 = $20 bonus

    This also makes all tries of equal 'value'...
"""

############
# 3.2 TASK #
############

# See 2.2 task (same for both)


####################
# 4.1 INSTRUCTIONS #
####################


def instructions_4(win, earnings, recall_trial_num):
    """
    Function to show instructions for "3 TASK" section of MADE.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    earnings -- float, amount earned in task section
    """
    #################### CONGRATS ####################
    # complete
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='Well done, you have completed the main task!\n\nYou earned a total of ${:.2f}/10 = ${:.2f}.'.format(earnings, earnings/10)).draw()
    show(win)
    #################### RECALL ####################
    # section
    visual.TextStim(win, pos=[0, 0], units='norm', height=0.28, color=-1,
                    text='Section 4').draw()
    show(win)
    # explain1
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='In the next section we want to see how well you learned the values of the houses and faces.').draw()
    show(win)
    # explain2
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='You will be asked to estimate the values of houses and faces like you did in Section 1.\n\nOnly this time with no feedback.').draw()
    show(win)
    # winning_money2
    visual.TextStim(win, pos=[0, 0], height=0.9,
                    text='Ready to begin?').draw()
    show(win)


##############
# 4.2 RECALL #
##############

# See 1.2 Stim Val Test (same for both)


####################
# 5.1 INSTRUCTIONS #
####################

def instructions_5(win):
    """
    Function to show instructions upon completion of MADE.

    Keyword arguments:
    win -- these are the window/monitor parameters that were set up earlier
    earnings -- float, amount earned in task section
    """
    #################### CONGRATS ####################
    # practice_complete
    visual.TextStim(win, pos=[0, 0], height=1.3, color=-1,
                    text='Well done, you have completed the experiment!').draw()
    show(win)
    visual.TextStim(win, pos=[0, 0], height=1.3, color=-1,
                    text='You will now complete a few questionnaires.\n\nPlease let the research assistant know that you are ready to continue.').draw()
    win.flip()
