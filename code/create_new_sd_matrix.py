'''
	Given the sd_transform.json file, this file will create a new mat file containing the updating semantic descriptors.
	This file will create 'new_data.mat' containing the binary and continuous semantic description matrices for all gesture categories.
'''

import sys
import numpy as np
import cv2
from glob import glob
from os.path import basename, dirname, join
import time
from scipy.io import loadmat, savemat
from helpers import *
import json

######################
### Initialization ###
######################
DATA_FOLDER = r'..\data'
MAT_FNAME = r'data_original.mat'
OUT_FPATH = join(DATA_FOLDER, 'data_fixed.mat')
SD_TRANSFORM_PATH = join(r'..\backup', 'sd_transform.json')
SD_LABELS_MAT_FNAME = 'full_sd_names'
CLASS_LABELS_MAT_FNAME = 'class_labels'
######################

## Read sd_transform.json
sd_dict = json_to_dict(SD_TRANSFORM_PATH)

## Read data.mat to get the semantic description matrix
data = loadmat(join(DATA_FOLDER, MAT_FNAME))
print('Keys in data.mat: \n',data.keys())
full_sd_names = np.array(cell_to_lstr(data[SD_LABELS_MAT_FNAME]))
class_labels = np.array(cell_to_lstr(data[CLASS_LABELS_MAT_FNAME]))

## New binary SD matrix
full_bin_sd_mat = np.zeros((len(class_labels), len(full_sd_names)))

for idx, cname in enumerate(class_labels):
	t_sd_list = sd_dict[cname]['new']
	t_sd_ids = [np.where(full_sd_names == t_sd)[0][0] for t_sd in t_sd_list if t_sd in full_sd_names]
	full_bin_sd_mat[idx, t_sd_ids] = 1

# print(full_bin_sd_mat.sum(axis = 0))

temp = full_bin_sd_mat.sum(axis = 0) == 0.0
print('INACTIVE Descriptors: #', len(full_sd_names[temp]))
# print(full_sd_names[temp])

temp = full_bin_sd_mat.sum(axis = 0) == 1.0
print('Descriptors that are active ONLY ONCE: #', len(full_sd_names[temp]))
# print(full_sd_names[temp])

## Saving to a new mat file

# full_bin_sd_mat
full_bin_sd_mat = full_bin_sd_mat.astype(np.int8)

# full_sd_names
full_sd_names = full_sd_names

# class_labels
class_labels = class_labels

# full_con_sd_mat
full_con_sd_mat = np.copy(data['full_con_sd_mat'])
mask_bin_sd = (full_bin_sd_mat == 1.0)
mask_con_sd = (full_con_sd_mat < 0.5)
full_con_sd_mat[np.logical_and(mask_bin_sd, mask_con_sd)] = 0.75
full_con_sd_mat[np.logical_and(np.logical_not(mask_bin_sd), np.logical_not(mask_con_sd))] = 0.25

# reduced_sd_names
reduced_sd_names = np.array(cell_to_lstr(data['reduced_sd_names']))
reduced_sd_ids = [np.where(full_sd_names == t_sd)[0][0] for t_sd in reduced_sd_names if t_sd in full_sd_names]

# bin_sd
bin_sd = full_bin_sd_mat[:, reduced_sd_ids].astype(np.int8)

# con_sd
con_sd = full_con_sd_mat[:, reduced_sd_ids]

## Save into a new mat file
result = {'con_sd': con_sd, 'reduced_sd_names': reduced_sd_names, 'full_sd_names': full_sd_names, 'full_bin_sd_mat': full_bin_sd_mat, 'full_con_sd_mat': full_con_sd_mat, 'bin_sd': bin_sd, 'class_labels': class_labels}
if(len(OUT_FPATH) !=  0): savemat(OUT_FPATH, result)
