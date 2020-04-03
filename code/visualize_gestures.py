
import sys
import numpy as np
import cv2
from glob import glob
from os.path import basename, dirname, join
import time
from scipy.io import loadmat
from helpers import *
import json

vid_id = 26

video_folder = r'..\media'
data_folder = r'..\data'
mat_fname = r'data.mat'

# create_sample_json(video_folder, data_folder)
# sys.exit()

vid_fpath = glob(join(video_folder, 'G'+str(vid_id)+'_*.mp4'))[0]

print('Gesture: ', basename(vid_fpath))

cap = cv2.VideoCapture(vid_fpath)
data = loadmat(join(data_folder, mat_fname))
bin_sd = data['full_bin_sd_mat']
full_sd_names = np.array(cell_to_lstr(data['full_sd_names']))
sd_names = full_sd_names[bin_sd[vid_id-1, :] == 1]
print(json.dumps(sd_names.tolist()))

while(cap.isOpened()):
	ret, frame = cap.read()
	if(not ret):
		time.sleep(0.4)
		cap.release()
		cap = cv2.VideoCapture(vid_fpath)
		continue
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
