import sys
from glob import glob
from os.path import basename, dirname, join
import time
from copy import deepcopy
import json

import numpy as np
from scipy.io import loadmat, savemat
import cv2

## Custom modules
from helpers import *
from SD_Transformer import SD_Transformer

## Initialization
DATA_FOLDER = r'..\data'
MAT_FNAME = r'sd_data_fixed.mat'
BIN_DATA_PREFIX = 'full_bin_sd_mat'
CON_DATA_PREFIX = 'full_con_sd_mat'
OUT_FPATH = join(DATA_FOLDER, 'new_sd_data.mat')

## Read data.mat to get the semantic description matrix
data = loadmat(join(DATA_FOLDER, MAT_FNAME))

## Instantiate SD_Transformer object
sid = SD_Transformer(bin_sd_mat = data[BIN_DATA_PREFIX])

## Print sid.sd_info ##
print('#### Printing sd_info variable ####')
sid.print_sd_data()
print('###################\n')

## Print sd_names_list ##
print('#### Printing sd_names_list ####')
print(sid.sd_names_list)
print('No. of descriptors: ', len(sid.sd_names_list))
print('No. of descriptors: ', sid.num_sds)
print('###################\n')

## Verify get_ids ##
print('#### Verify get_ids() ####')
print('LHM: ', sid.get_ids(['LHM']))
print('RHP: ', sid.get_ids(['RHP']))
print('LHOF: ', sid.get_ids(['LHOF']))
print("['RHM', 'LHD', 'POB_Head', 'GP_BelowChest']", sid.get_ids(['RHM', 'LHD', 'POB_Head', 'GP_BelowChest']))
print('Left IDs', sid.get_left_ids())
print('Right IDs', sid.get_right_ids())
print('Finger sensitive IDs', sid.get_fing_sensitive_ids())
print('Plane IDs', sid.get_plane_ids())
print('Orientation IDs', sid.get_orientation_ids())
print('Hand configuration IDs', sid.get_hand_configuration_ids())
print('Overall IDs', sid.get_overall_ids())
print('Empty IDs', sid.get_empty_ids(K = 0))
print('Empty IDs: same: ', sid.get_empty_ids(K = 0, sd_mat = data[BIN_DATA_PREFIX]))
print('Empty IDs with symmetry: ', sid.get_empty_ids(K = 0, use_symmetry = True))
print('###########################')

print('#### Full IDs ####')
K = 10
print(np.sum(sid.bin_sd_mat.sum(axis = 0) >= (26-K)))
print('Full IDs: ', sid.get_full_ids(K = K))
print('Full IDs: Same:', sid.get_full_ids(K = K, sd_mat = data[BIN_DATA_PREFIX]))
print('##################')
K = 10
print('#### Remove empty and full ids: ####')
print('Remove empty: ', len(sid.remove_empty(K = K)))
print('Remove full: ', len(sid.remove_full(K = K)))
print('Remove empty full: ', len(sid.remove_empty_full(K = K)))

