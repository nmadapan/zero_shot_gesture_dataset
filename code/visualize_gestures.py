'''
	This file helps you visualize the 26 gestures present in the dataset.
	Change the variables in the 'Initialization' block.
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
DATA_FOLDER_PATH = r'..\data' # Path pointing to the directory containing semantic descriptor data
MAT_FNAME = r'data_original.mat' # Name of the mat file containing semant descriptor data
BIN_SD_MAT_FNAME = 'full_bin_sd_mat'
SD_LABELS_MAT_FNAME = 'full_sd_names'
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
full_sd_names = np.array(cell_to_lstr(data[SD_LABELS_MAT_FNAME]))
sd_names = full_sd_names[bin_sd[GESTURE_ID-1, :] == 1]
print(json.dumps(sd_names.tolist()))

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
