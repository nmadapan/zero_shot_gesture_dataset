'''
	Given the sd_transform.json file, this file will create a new mat file containing the updated semantic descriptors (SD).

	This file will create 'sd_data_fixed.mat' containing the binary and continuous SD matrices for all gesture categories.

	Note that, the list of command names and the list of SD names present in the MAT_FNAME should be in a cell array format and NOT in the character array format. It is in a char array format: run pmat_to_mmat.m file which does this conversion.

	INPUT:
		* 'sd_data_mturk2.mat'
			- 'full_sd_names' variable in the mat file.
			- 'full_cmd_names' variable in the mat file.
			- 'full_con_sd_mat' variable in the mat file.
			- 'sd_names' variable in the mat file.
		* 'sd_transform.json'
	OUTPUT:
		* 'sd_data_fixed.mat'
			- 'full_sd_names' (copied)
			- 'full_cmd_names' (copied)
			- 'sd_names' (copied)
			- 'full_bin_sd_mat'
			- 'full_con_sd_mat'
			- 'bin_sd'
			- 'con_sd'
'''

import sys
import time
from os.path import basename, dirname, join
import json
from glob import glob
import numpy as np
from scipy.io import loadmat, savemat
import cv2

# Custom modules
from helpers import *

######################
### Initialization ###
######################
DATA_FOLDER = r'..\data'
MAT_FNAME = r'sd_data_mturk2.mat'
OUT_FPATH = join(DATA_FOLDER, 'sd_data_fixed.mat')
SD_TRANSFORM_PATH = join(r'..\data', 'sd_transform.json')
SD_LABELS_MAT_FNAME = 'full_sd_names'
CLASS_LABELS_MAT_FNAME = 'full_cmd_names'
FULL_CON_MAT_FNAME = 'full_con_sd_mat'
SD_NAMES_MAT_FNAME = 'sd_names'
######################

## Read sd_transform.json
sd_dict = json_to_dict(SD_TRANSFORM_PATH)

## Read MAT_FNAME to get the semantic description matrix
data = loadmat(join(DATA_FOLDER, MAT_FNAME))
print('Keys in data.mat: \n',data.keys())
full_sd_names = np.array(cell_to_lstr(data[SD_LABELS_MAT_FNAME]))
class_labels = np.array(cell_to_lstr(data[CLASS_LABELS_MAT_FNAME]))

############################
### New binary SD matrix ###
############################

full_bin_sd_mat = np.zeros((len(class_labels), len(full_sd_names)))
for idx, cname in enumerate(class_labels):
	t_sd_list = sd_dict[cname]['new']
	t_sd_ids = [np.where(full_sd_names == t_sd)[0][0] for t_sd in t_sd_list if t_sd in full_sd_names]
	full_bin_sd_mat[idx, t_sd_ids] = 1

temp = full_bin_sd_mat.sum(axis = 0) == 0.0
print('INACTIVE Descriptors: #', len(full_sd_names[temp]))

temp = full_bin_sd_mat.sum(axis = 0) == 1.0
print('Descriptors that are active ONLY ONCE: #', len(full_sd_names[temp]))

################################
### Saving to a new mat file ###
################################

## full_bin_sd_mat: Change the type from double to int8
full_bin_sd_mat = full_bin_sd_mat.astype(np.int8)

## full_sd_names
full_sd_names = full_sd_names

## class_labels
class_labels = class_labels

## full_con_sd_mat
# Correct the mistakes in full_con_sd_mat. If the value in full_con_sd_mat is
# < 0.5 and the corresponding entry in full_bin_sd_mat is > 0.5, then change
# the value of this entry to 0.75. Likewise, if the value in full_con_sd_mat
# is > 0.5 and the corresponding entry in full_bin_sd_mat is < 0.5, then
# change the value of this entry to 0.25.
full_con_sd_mat = np.copy(data[FULL_CON_MAT_FNAME])
mask_bin_sd = (full_bin_sd_mat == 1.0)
mask_con_sd = (full_con_sd_mat < 0.5)
full_con_sd_mat[np.logical_and(mask_bin_sd, mask_con_sd)] = 0.75
full_con_sd_mat[np.logical_and(np.logical_not(mask_bin_sd), np.logical_not(mask_con_sd))] = 0.25

## reduced_sd_names
reduced_sd_names = np.array(cell_to_lstr(data[SD_NAMES_MAT_FNAME]))
reduced_sd_ids = [np.where(full_sd_names == t_sd)[0][0] for t_sd in reduced_sd_names if t_sd in full_sd_names]

## bin_sd: Change the type from double to int8.
bin_sd = full_bin_sd_mat[:, reduced_sd_ids].astype(np.int8)

## con_sd
con_sd = full_con_sd_mat[:, reduced_sd_ids]

## Save into a new mat file
result = {'con_sd': con_sd, 'sd_names': reduced_sd_names, 'full_sd_names': full_sd_names, 'full_bin_sd_mat': full_bin_sd_mat, 'full_con_sd_mat': full_con_sd_mat, 'bin_sd': bin_sd, CLASS_LABELS_MAT_FNAME: class_labels}
if(len(OUT_FPATH) !=  0): savemat(OUT_FPATH, result)
