# Code to generate new SDs

## Description
The overall pipeline to generate the semantic description matrices is given as follows.

* **Gesture Datasets (*Repository: ZSGL_Dataset*)**
	* It has two gesture datasets: 1. MSRC-12 and 2. CGD 2013. Eight out of 12 gestures from MSRC-12 and 20 out of 20 gestures from CGD 2013 are considered. So, there are a total of 28 gesture classes. These videos are present in 'ZSGL_Dataset/media/gesture_videos/final_videos_v1' repository. These are the vidoes that are originally used on the MTurk platform.
	* However, once the analysis was done, we realized that 'LEFT' and 'RIGHT' tags are confusing and MTurk workers made some errors related to the 'LEFT' and 'RIGHT' directions. These mistakes are manually fixed. And, the 'LEFT' and 'RIGHT' tags are switched and the annotations are changed accordingly. A new set of videos are created (refer to 'ZSGL_Dataset/media/gesture_videos/morphed_videos_v1').

* **Gesture Attributes (*Repository: ZSGL_Dataset*)**
	* We came up with 75 gesture attributes (refer to sd.json file present in ''). These attributes are belong to 11 high-level categories: ['LHM', 'RHM', 'OM', 'LHD', 'RHD', 'LHOF', 'RHOF', 'LHP', 'RHP', 'POB', 'GP'].
	* Each of these categories have an attribute called 'None' which is removed later on in order to get a list of 64 attributes. Refer to 'ZSGL_Dataset' repository.

* **AMT Interface (*Repository: ZSGL_Dataset*)**
	* We developed a web interface that was floated on MTurk platform in order to allow AMT workers to annotate the 28 gesture categories w.r.t the 75 attributes. Each gesture class was annotated by 15-20 workers.
	* The data was obtained from two sources: 1. In the form of a database obtained from the Ubuntu computer. This database consisted of the descriptor annotations, time taken by the workers to finish the annotations, confirmation IDs etc. 2. AMT Platform gives data related to worker IDs, start and completion times, confirmation IDs, etc.
	* This data was processed using the code present in 'ZSGL_Dataset/misc' repository. The three main scripts were 'DescriptorPreprocess.py', 'MturkPreprocess.py' and 'Preprocess.py'.
	* This code gets the data from both the sources, combines them to generate the following: 1. binary SD matrix (28 x 64), 2. continuous SD matrix (28 x 64), 3. reduced binary SD matrix (28 x 34), 4. reduced continuous SD matrix (28 x 34), 5. full list of SDs (64 x 1), 6. reduced list of SDs (34 x 1) and 7. Full list of commands (28, 1). This data is stored in the *sd_data_mturk.mat* file using *scipy.io.savemat*.

* **Issue with scipy.io Library (*Repository: zero_shot_gesture_dataset*)**
	* There is an issue with the *scipy.io.savemat* function. This function stores the list of strings (python) as a character array in the .mat file instead of storing it as a cell array of strings.
	* One option is to read *sd_data_mturk.mat* using *scipy.io.loadmat* and manually strip the strings in the form of character array using the function, *strip()*
	* So, I created a Matlab script *pmat_to_mmat.m* that fixes this issue i.e. list of strings is stored as a character array. This script takes *sd_data_mturk.mat* as an input and outputs *sd_data_mturk2.mat* in which this isse is fixed.

* **Mistakes in SD Data (*Repository: zero_shot_gesture_dataset*)**
	* Once the SD matrices are obtained from the MTurk data, we performed a manual verification of data. It was found that subjects performed following mistakes.
	* Confusion between the LEFT and RIGHT directions. This issue was fixed manually by modifying the descriptors in *sd_transform.json* file. Also, a new set of videos were created to represent these changes (refer to 'ZSGL_Dataset/media/gesture_videos/morphed_videos_v1'). Now, when you annotate the data, user is expected to pretend as if he/she was performing the gesture and annotate the gesture accordingly.
	* Unintended upward motion that happens in the beginning of the gesture was considered as an upward motion. Now, this issue was fixed setting the respective descriptors (*LHM_Up, RHM_Up*) to zero unless the hand crosses the lower chest.
	* When large circular motions occur, they were not annotated for left, right, up, and downward motions. For some gestures, the clockwise and counterclockwise motions are not included. Now, this issue was fixed by adding those missing descriptors.
	* The gesture descriptor related to the 'part of the body referred to ' *(POB)* was confusing and sparse across the gesture categories. Hence, this issue was fixed by setting the values related to *POB_Eyes, POB_Mouth, POB_Nose* to zero and setting the value of *POB_Head* to one when other body parts are referred in the gesture. So, in essence, it is equivalent to removing those descriptors (*POB_Eyes, POB_Mouth, POB_Nose*) as they are always set to zero.
	* For few gestures, the descriptors related to the general position (*GP*) were wrong and they were fixed now.

* **SD Matrix Correction (*Repository: zero_shot_gesture_dataset*)**
	* Run ``` python3 visualize_gestures.py -g 1``` to visualize the gesture 1. This will print the current semantic descriptors that are associated with this gesture. Now, manually verify if those descriptors are correct. If they are wrong, manually edit them in *sd_transform.json* file. In addition to *old* and *new* descriptors, the keys: *sym* (true if the gesture is symmetrical) and *modified* (true if descriptors are corrected) are also added in the *sd_transform.json* file. Repeat this procedure for all the gesture ids. Once this is done, you have a complete *sd_transform.json* file. This script takes *sd_data_mturk2.mat* as an input and displays the video using OpenCV.
	* Run ```python3 verify_sd_transform.py``` to conduct automatic sanity checks on *sd_transform.json* file. These tests make sure that there are no typos or inconsistencies in the json file. This script prints the inconsistencies so you can go and manually fix them. Next, it also prints the descriptors that are removed/added/modified with respect to each gesture for manual verification. This script takes *sd_data_mturk2.mat* as an input to read 'full_sd_names' and 'full_cmd_names' variables.
	* Run ```python3 create_new_sd_matrix.py``` to generate a *sd_data_fixed.mat* file containing the binary and continuous description matrices of the gesture categories. This script takes *sd_data_mturk2.mat* as an input and modifies its variables ('full_bin_sd_mat', 'full_con_sd_mat', 'bin_sd' and
	'con_sd'), and saves it in *sd_data_fixed.mat* file. Rest of the unmodified variables are saved as they are.
	* Run ```python3 SD_Transformer.py``` to manipulate the SD matrices depending on what semantic descriptors we want to use. For instance, we can use this script to eliminate the descriptors that are always zero, always one, combine left and right ids, and so on. This script takes *sd_data_fixed.mat* file as an input and generates *new_sd_data.mat* file that containes required semantic description matrices.

* **Overall Pipeline**
	1. **MTurk Data Collection** &rarr; **Python:** `DescriptorPreprocess.py, MturkPreprocess.py, Preprocess.py` &rarr; `sd_data_mturk.mat` # To create the initial SD data from MTurk data (*Repository: ZSGL_Dataset*).
	2. `sd_data_mturk.mat` &rarr; **Matlab:** `pmat_to_mmat.m` &rarr; `sd_data_mturk2.mat` # To convert the list of strings from char arrays to cell arrays (*Repository: zero_shot_gesture_dataset*).
	3. `sd_data_mturk2.mat` &rarr; **Python:** `visualize_gestures.py, verify_sd_transform.py, create_new_sd_matrix.py` &rarr; `sd_data_fixed.mat` # To fix the human errors in the MTurk annotations and the SD matrices (*Repository: zero_shot_gesture_dataset*).
	4. `sd_data_fixed.mat` &rarr; **Python:** `SD_Transformer.py` &rarr; `new_sd_data.mat` # To manipulate the SD matrices in a desired manner (*Repository: zero_shot_gesture_dataset*).
