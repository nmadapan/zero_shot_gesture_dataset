import sys
import numpy as np
import cv2
from glob import glob
from os.path import basename, dirname, join
import time
from scipy.io import loadmat
import json

def cell_to_lstr(cell_array):
	'''
		Convert a 1D cell array of strings from .mat file to list of strings in python3.
		Input:
			cell_array - 1D cell array of strings from .mat file.
		Output: list of strings
	'''
	temp = []
	for idx in range(len(cell_array)):
		temp.append(str(cell_array[idx][0][0]))
	return temp

def create_sample_json(video_folder, data_folder):
	mat_fname = 'data.mat'
	result = {}

	print('{')
	for vid_id in range(1, 27):
		vid_fpath = glob(join(video_folder, 'G'+str(vid_id)+'_*.mp4'))[0]
		vid_fname = basename(vid_fpath)

		data = loadmat(join(data_folder, mat_fname))
		bin_sd = data['full_bin_sd_mat']
		full_sd_names = np.array(cell_to_lstr(data['full_sd_names']))
		sd_names = full_sd_names[bin_sd[vid_id-1, :] == 1]

		print('\t"{0}":'.format(basename(vid_fpath)[:-4]))
		print('\t{')
		print('\t\t"sym": ', 'true', ',')
		print('\t\t"modified": ', 'false', ',')
		print('\t\t"old": ', json.dumps(sd_names.tolist()), ',')
		print('\t\t"new": ', json.dumps(sd_names.tolist()))
		print('\t},')
	print('}')

	return result

