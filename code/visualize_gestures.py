'''
	This file helps you visualize the 26 gestures that are present in the dataset.

	Change the variables in the 'Initialization' block.

	The acronym, SD refers to Semantic Descriptor/Description. Note that, the list of command names and the list of SD names present in the MAT_FNAME should be in a cell array format and NOT in the character array format. It is in a char array format: run pmat_to_mmat.m file which does this conversion.

	INPUT:
		* 'sd_data_mturk2.mat'
			- 'full_bin_sd_mat' variable in the mat file.
			- 'full_sd_names' variable in the mat file.
	OUTPUT: None
'''

import sys
import time
from os.path import basename, dirname, join
from glob import glob
import argparse
from scipy.io import loadmat
import json
import numpy as np
import cv2

# Custom library
from helpers import *

######################
### Initialization ###
######################
GESTURE_ID = 1 # Gesture ID. Varies from 1 to 26.
VIDEO_FOLDER_PATH = r'..\media' # Path pointing to directory containing gesture videos
DATA_FOLDER_PATH = r'..\data' # Path pointing to the directory containing SD data ('sd_data_mturk.mat' or 'sd_data_mturk2.mat')
MAT_FNAME = r'sd_data_mturk.mat' # Name of the mat file containing SD data
BIN_SD_MAT_FNAME = 'full_bin_sd_mat' # A variable in the .mat file that contains full binary SD matrix (26 x 64).
SD_LABELS_MAT_FNAME = 'full_sd_names' # A variable in the .mat file that contains a list of SD names (64 x 1).
######################

#######################
### Argument Parser ###
#######################
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gesture_id", default=GESTURE_ID, help=("Gesture ID ranges from 1 to 26 "))
args = vars(parser.parse_args())
GESTURE_ID = int(args['gesture_id'])
######################

## Load the video
vid_fpath = glob(join(VIDEO_FOLDER_PATH, 'G'+str(GESTURE_ID)+'_*.mp4'))[0]
cap = cv2.VideoCapture(vid_fpath)
print('Gesture: ', basename(vid_fpath))

## Obtain corresponding gesture descriptors
data = loadmat(join(DATA_FOLDER_PATH, MAT_FNAME))
bin_sd = data[BIN_SD_MAT_FNAME]
# full_sd_names = np.array(cell_to_lstr(data[SD_LABELS_MAT_FNAME]))
full_sd_names = np.array(data[SD_LABELS_MAT_FNAME])
full_sd_names = np.array([temp.strip() for temp in full_sd_names])
sd_names = full_sd_names[bin_sd[GESTURE_ID-1, :] == 1]
print(json.dumps(sd_names.tolist()))

## Loop over the gesture.
while(cap.isOpened()):
	ret, frame = cap.read()
	if(not ret): # Loop over the gesture
		time.sleep(0.4)
		cap.release()
		cap = cv2.VideoCapture(vid_fpath)
		continue
	cv2.imshow(basename(vid_fpath)[:-4], frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

# create_sample_json(VIDEO_FOLDER_PATH, DATA_FOLDER_PATH)
# sys.exit()
