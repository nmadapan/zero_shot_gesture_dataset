'''
	This script implements a class 'SID_Transformer' which helps manipulate the semantic description matrix. For instance, you can switch the left and right hand, remove some group of descriptors, etc.
'''

import sys
import numpy as np
import cv2
from glob import glob
from os.path import basename, dirname, join
import time
from copy import deepcopy
from scipy.io import loadmat, savemat
from helpers import *
import json

class SID_Transformer(object):
	def __init__(self, data_folder = r'..\data', sd_fname = 'sd.json', bin_sd_mat = None):
		'''
			data_folder: path to a folder containing sd.json file
			sd_fname: sd.json file containing semantic descriptors in an organized manner
			bin_sd_mat: A 2D np.ndarray of size (num_classes x num_descriptors). This must be a matrix with only 0 and 1s.
		'''
		# Path to the sd.json file
		self.sd_json_path = join(data_folder, sd_fname)
		self.sd_info = json_to_dict(self.sd_json_path)
		self.bin_sd_mat = bin_sd_mat
		self.check_mat() # Check if bin_sd_mat has only zeros and ones.

		## Prefix of semantic descriptors.
		self.sd_prefixes = ['LHM', 'RHM', 'OM', 'LHD', 'RHD', 'LHOF', 'RHOF', 'LHP', 'RHP', 'POB', 'GP']

		## Remove 'None' descriptors from sd_info and re-indexes the ids.
		# This call modifies the self.sd_info variable.
		self.remove_none_descriptors()

		## list of names of SDs
		self.sd_names_list = np.array(self.gen_sd_names())
		self.num_sds = len(self.sd_names_list)

	def check_mat(self):
		if(self.bin_sd_mat is not None):
			zeroes = (self.bin_sd_mat == 0.0).flatten().sum()
			ones = (self.bin_sd_mat == 1.0).flatten().sum()
			if(zeroes + ones != self.bin_sd_mat.size):
				raise Exception('Matrix is not binary. It should have only zero and ones. ')

	def gen_sd_names(self):
		'''
			Returns a list of SD names. ['LHM_Left', 'LHM_Right', ...]. Its length is equal to self.num_sds
		'''
		one_dict = {}
		for key, value in self.sd_info.items():
			for _key, _value in value.items():
				one_dict[_value] = key + '_' + _key
		return list(one_dict.values())

	def remove_none_descriptors(self):
		'''
			Removes the 'None' descriptors from self.sd_info dictionary and re-indexes the IDs given to each descriptor.
			NOTE: Call to this function modifies the self.sd_info variable.
		'''
		for idx, key in enumerate(self.sd_prefixes):
			if('None' in self.sd_info[key]): self.sd_info[key].pop('None')
			for _key, _value in self.sd_info[key].items():
				self.sd_info[key][_key] = _value - idx

	def print_sd_data(self):
		## Pretty printing of sd_info variable.
		for key in self.sd_prefixes:
			print(key, ':')
			# Interchange keys and values in each sub-dictionary
			ndict = {y: x for x, y in self.sd_info[key].items()}
			# Sort the SD IDs
			sorted_values = sorted(ndict.keys())
			for _key in sorted_values:
				print('\t', ndict[_key], ':', _key)

	def get_ids(self, strs):
		## Given a list of prefix strings or sd names, return the SD IDs.
		ids = []
		for val in strs:
			if('_' in val): # If there is '_', it means that it is a full descriptor.
				temp = np.where(self.sd_names_list == val)[0][0]
				ids.append(temp)
			else: #  Otherwise, it is a prefix.
				ids += self.get_prefix_ids(val)
		return np.unique(ids).tolist()

	def get_prefix_ids(self, strs, only_prefix = True):
		## Given the prefix strings, return the SD IDs.
		ids = []
		for key, value in self.sd_info.items():
			for _key, _value in value.items():
				if(key in strs): ids.append(_value)
		return sorted(ids)

	def get_left_ids(self):
		## Return the SD IDs of left hand.
		left_strs = ['LHM', 'LHD', 'LHOF', 'LHP']
		return self.get_ids(left_strs)

	def get_right_ids(self):
		## Return the SD IDs of right hand.
		right_strs = ['RHM', 'RHD', 'RHOF', 'RHP']
		return self.get_ids(right_strs)

	def get_fing_sensitive_ids(self):
		## Return the SD IDs of orientation and configuration of left hand and right hands.
		strs = ['LHD', 'RHD', 'LHOF', 'RHOF']
		return self.get_ids(strs)

	def get_plane_ids(self):
		## Return the SD IDs of plane information of left hand and right hands.
		strs = ['LHP', 'RHP']
		return self.get_ids(strs)

	def get_orientation_ids(self):
		## Return the SD IDs of orientation of left hand and right hands.
		strs = ['LHD', 'RHD']
		return self.get_ids(strs)

	def get_hand_configuration_ids(self):
		## Return the SD IDs of configuration of left hand and right hands.
		strs = ['LHOF', 'RHOF']
		return self.get_ids(strs)

	def get_overall_ids(self):
		## Return the SD IDs related to overall motion, general position and part of the body.
		strs = ['OM', 'POB', 'GP']
		return self.get_ids(strs)

	def get_empty_ids(self, K = 0, use_symmetry = False):
		## Any descriptor that is present at most K number of times.
		row_sum = np.sum(self.bin_sd_mat, axis=0)
		if(use_symmetry):
			left_ids = self.get_left_ids()
			right_ids = self.get_right_ids()
			temp = np.maximum(row_sum[left_ids], row_sum[right_ids])
			row_sum[left_ids], row_sum[right_ids] = temp, temp
		empty_ids = np.nonzero(row_sum <= K)[0].tolist()
		empty_ids = np.unique(empty_ids)
		sd_names = np.copy(self.sd_names_list)[empty_ids]
		return empty_ids.tolist(), sd_names

	def switch(self, x_ids, y_ids, sd_mat = None):
		## Exchange the semantic descriptor data between x_ids and y_ids
		# Returns new sd_matrix and new sd names
		assert self.bin_sd_mat is not None, 'SD matrix should be a numpy array. It can not be none'
		assert isinstance(self.bin_sd_mat, np.ndarray), 'SD matrix should be a numpy array.'
		assert self.bin_sd_mat.shape[1] == len(self.sd_names_list), 'No. of columns in bin_sd_mat should be equal to the number of descriptors'
		assert len(x_ids) == len(y_ids), 'For switching: length of x_ids and y_ids should be same'
		if(sd_mat is None): sd_mat = np.copy(self.bin_sd_mat)
		sd_names = np.copy(self.sd_names_list)
		sd_mat[:, x_ids], sd_mat[:, y_ids] = sd_mat[:, y_ids], sd_mat[:, x_ids]
		sd_names[x_ids], sd_names[y_ids] = sd_names[y_ids], sd_names[x_ids]
		return sd_mat, sd_names

	def remove(self, x_ids, sd_mat = None):
		## Removes the ids present x_ids from the matrix.
		# Returns new sd_matrix and new sd names
		assert self.bin_sd_mat is not None, 'SD matrix should be a numpy array. It can not be none'
		assert isinstance(self.bin_sd_mat, np.ndarray), 'SD matrix should be a numpy array.'
		assert self.bin_sd_mat.shape[1] == len(self.sd_names_list), 'No. of columns in bin_sd_mat should be equal to the number of descriptors'
		all_ids = set(range(len(self.sd_names_list)))
		keep_ids = list(all_ids.difference(set(x_ids)))
		if(sd_mat is None): sd_mat = np.copy(self.bin_sd_mat)
		sd_mat = sd_mat[:, keep_ids]
		sd_names = np.copy(self.sd_names_list)[keep_ids]
		return sd_mat, sd_names

	def switch_left_right(self, sd_mat = None):
		## Exchange the semantic descriptor data between left_ids and right_ids
		# Returns new sd_matrix and new sd names
		left_ids = self.get_left_ids()
		right_ids = self.get_right_ids()
		return self.switch(left_ids, right_ids, sd_mat)

	def remove_left(self, sd_mat = None):
		## Removes the left ids from the matrix.
		# Returns new sd_matrix and new sd names
		return self.remove(self.get_left_ids(), sd_mat)

	def remove_right(self, sd_mat = None):
		## Removes the right ids from the matrix.
		# Returns new sd_matrix and new sd names
		return self.remove(self.get_right_ids(), sd_mat)

	def remove_plane(self, sd_mat = None):
		## Removes the plane ids from the matrix.
		# Returns new sd_matrix and new sd names
		return self.remove(self.get_plane_ids(), sd_mat)

	def remove_overall_motion(self, sd_mat = None):
		## Removes the overall motion ids from the matrix.
		# Returns new sd_matrix and new sd names
		return self.remove(self.get_ids(['OM']), sd_mat)

	def remove_empty(self, sd_mat = None):
		## Removes the empty ids from the matrix.
		# Returns new sd_matrix and new sd names
		return self.remove(self.get_empty_ids()[0], sd_mat)

	def combine_left_right(self, sd_mat = None):
		## Combines the values corresponding to left and right SD IDs by taking a max of both the values
		left_ids = self.get_left_ids()
		right_ids = self.get_right_ids()
		if(sd_mat is None): sd_mat = np.copy(self.bin_sd_mat)
		sd_mat[:, left_ids] = np.maximum(sd_mat[:, left_ids], sd_mat[:, right_ids])
		return self.remove_right(sd_mat)

if __name__ == '__main__':
	## Initialization
	DATA_FOLDER = r'..\data'
	MAT_FNAME = r'data_fixed.mat'
	BIN_DATA_PREFIX = 'full_bin_sd_mat'
	CON_DATA_PREFIX = 'full_con_sd_mat'
	OUT_FPATH = join(DATA_FOLDER, 'new_data.mat')

	## Read data.mat to get the semantic description matrix
	data = loadmat(join(DATA_FOLDER, MAT_FNAME))

	## Instantiate SID_Transformer object
	sid = SID_Transformer(bin_sd_mat = data[BIN_DATA_PREFIX])

	################################
	## Manipulate the matrix here ##
	################################
	# For example, let us say we want to remove empty ids with K = 1, 'OM' related IDs and finger sensitive ids
	# rem_ids = sid.get_empty_ids(K = 1)[0] + sid.get_ids(['OM']) + sid.get_fing_sensitive_ids()

	# For example, let us say we want to remove empty ids with K = 1 and finger sensitive ids
	# rem_ids = sid.get_empty_ids(K = 1)[0] + sid.get_fing_sensitive_ids()

	# For example, let us say we want to remove ONLY empty ids with K = 3
	# rem_ids = sid.get_empty_ids(K = 3)[0]

	# For example, let us say we want to remove ONLY empty ids with K = 3 by using the symmetry between the left and right ids.
	rem_ids = sid.get_empty_ids(K = 3, use_symmetry = True)[0]

	rem_ids = np.unique(rem_ids)
	bin_sd, reduced_sd_names = sid.remove(rem_ids)
	con_sd, _ = sid.remove(rem_ids, data[CON_DATA_PREFIX])
	new_data = deepcopy(data)
	new_data['bin_sd'] = bin_sd
	new_data['con_sd'] = con_sd
	new_data['reduced_sd_names'] = reduced_sd_names
	if(len(OUT_FPATH) !=  0): savemat(OUT_FPATH, new_data)

# Save a new mat file containing the same set of variables.
# Debug switch and remove again.
# Debug new functions properly until you are confident.
# Proofread everything that you have so far. And wrap things nicely. Updating the README.md and stuff.
