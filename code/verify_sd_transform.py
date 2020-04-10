'''
	The semantic descriptors (SD) assigned to each gesture category using MTURK annotations are inconsistent in some cases. So, I manually inspected each gesture (by running visualize_gestures.py) and its descriptors, and manually corrected them. This information is present in 'sd_transform.json'.

	This file reads the file, 'sd_transform.json' and conducts sanity checks to make sure that there are no mistakes (typos) in the new descriptors.

	Note that, the list of command names and the list of SD names present in the MAT_FNAME should be in a cell array format and NOT in the character array format. It is in a char array format: run pmat_to_mmat.m file which does this conversion.

	INPUT:
		* 'sd_data_mturk2.mat'
			- 'full_sd_names' variable in the mat file.
			- 'full_cmd_names' variable in the mat file.
		* 'sd_transform.json'
	OUTPUT: None
'''

import sys
import numpy as np
import time
from os.path import basename, dirname, join

from glob import glob
from scipy.io import loadmat
import json
import cv2

# Custom modules
from helpers import *

########################
# Test 1: When modified flag is set to False, old and new variables should be equivalent. Vice versa otherwise.
# Test 2: Contents of old and new variables should be present in full_sd_names.
# Test 3: Print what SD's that are added, modified and removed.
########################

########################
#### Initialization ####
########################
DATA_FOLDER = r'..\data'
MAT_FNAME = r'sd_data_mturk2.mat'
SD_TRANSFORM_PATH = join(DATA_FOLDER, 'sd_transform.json')
SD_LABELS_MAT_FNAME = 'full_sd_names'
CLASS_LABELS_MAT_FNAME = 'full_cmd_names'
########################

## Read sd_transform.json
full_sd_dict = json_to_dict(SD_TRANSFORM_PATH)

## Read data.mat to get the SD matrix
data = loadmat(join(DATA_FOLDER, MAT_FNAME))
full_sd_names = np.array(cell_to_lstr(data[SD_LABELS_MAT_FNAME]))
class_labels = np.array(cell_to_lstr(data[CLASS_LABELS_MAT_FNAME]))

for idx, cname in enumerate(class_labels):
	sd_cname_dict = full_sd_dict[cname]
	sym = sd_cname_dict['sym']
	modified = sd_cname_dict['modified']
	old_sd_names = np.array(sd_cname_dict['old'])
	new_sd_names = np.array(sd_cname_dict['new'])

	## Test 1: When modified flag is set to False, old and new variables should be equivalent. Vice versa otherwise.
	flag = True
	if(not modified):
		flag = flag and (old_sd_names == new_sd_names).all()
	if(modified):
		if(len(old_sd_names) == len(new_sd_names)):
			flag = flag and (not (old_sd_names == new_sd_names).all())
	if(not flag): print('Test 1 Failed: ', cname)

	## Test 2: Contents of old and new variables should be present in full_sd_names.
	flag = True
	flag = flag and ((full_sd_names == old_sd_names[:, np.newaxis]).flatten().sum() == len(old_sd_names))
	flag = flag and ((full_sd_names == new_sd_names[:, np.newaxis]).flatten().sum() == len(new_sd_names))
	if(not flag): print('Test 2 Failed: ', cname)

print('If nothing is printed before this, tests are successfull :)) ')

## Test 3: Print what SD's are added, modified and removed.
for idx, cname in enumerate(class_labels):
	print('Gesture: %s'%(cname))
	sd_cname_dict = full_sd_dict[cname]
	sym = sd_cname_dict['sym']
	modified = sd_cname_dict['modified']
	old_sd_names = set(sd_cname_dict['old'])
	new_sd_names = set(sd_cname_dict['new'])

	# print('\tOld: ', old_sd_names)
	# print('\tNew: ', new_sd_names)
	# print('\tCommon: ', old_sd_names.intersection(new_sd_names))
	print('\tRemoved: ', old_sd_names.difference(new_sd_names))
	print('\tAdded: ', new_sd_names.difference(old_sd_names))
