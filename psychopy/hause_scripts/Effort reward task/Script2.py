'''
Function to move newly created files to a specific folder. By default, it's the Data folder. To use this script, just run it (like how you run any other PsychoPy experiment script), and enter the participant number accordingly. The participants' files will be moved into the Data folder.

Click green runinng man icon to begin.
'''

import pandas as pd
import numpy as np
import random, os, time
import zipfile

def moveFiles(outputDir = 'Data'):
    # info = {}
    # info['participant'] = ''
    # dlg = gui.DlgFromDict(info) # create a dialogue box (function gui)
    # if not dlg.OK: # if dialogue response is NOT OK, quit
    #     core.quit()

    # list files in current directory
    files = os.listdir(os.getcwd())

    # info['participant'] = "{:03}".format(int(info['participant']))

    # create new directory (or the directory to move files to)
    dataDirectory = os.getcwd() + os.path.sep + outputDir
    if not os.path.exists(dataDirectory):
        os.mkdir(dataDirectory)
        os.mkdir(dataDirectory + os.path.sep + 'Archive')

    for filename in files:
        if filename.endswith("csv") or filename.endswith("npy"):
            # print filename
            newNameAndDir = dataDirectory + os.path.sep + filename
            # print newNameAndDir
            os.rename(filename, newNameAndDir)

moveFiles(outputDir='Data') # move csv into Data directory

def zipStuff():
    # list files in current directory
    os.chdir('Data')
    # print os.getcwd()

    files = os.listdir(os.getcwd())
    # print files

    zipFileName = "_UPLOAD_ONLY_THIS_FILE_" + str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())) + '.zip'

    # remove existing zip files
    for filename in files:
        if filename.endswith("zip"):
            # print filename
            os.remove(filename)

    # zip file
    with zipfile.ZipFile(zipFileName, 'w') as myzip:
        for filename in files:
            if filename.endswith("csv"):
                # print filename
                myzip.write(filename)

zipStuff() # remove existing zip files and zip all csv files in the Data directory
