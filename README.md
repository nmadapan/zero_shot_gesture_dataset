# Dataset for Zero-Shot Gesture Learning (ZSGL)

## Description
The main goal of this project is to facilitate zero-shot gesture learning (ZSGL). Zero-shot learning (ZSL) is a transfer learning paradigm in which the training and testing classes are disjoint i.e. the classes encountered during training and testing are different. ZSL relies on the meta-information available either in the form of semantic descriptors or embedding vectors in order to transfer the knowledge gained from the training classes to the classes that were never seen before [1].

ZSL was extensively studied in the domains of object recognition [1], neural decoding [2], scene understanding [3]. However, the problem of ZSGL has been hardly studied in the computer vision research due to unavailability of data standard attribute-based datasets for gestures. Hence, in this project, we developed the first ZSGL dataset consisting of gesture categories from MSRC-12 [4] and CGD 2013 [5] datasets.

## Experimental Protocol

First, we relied on the human studies, and literature in gestures, semantics and computational linguistics to develop a list of representative attributes of gesture categories. The idea is to represent each gesture category as a binary vector where zero/one indicates the absence/presence of the respective attribute. If there are *K* attributes, then each category is represented as a *K* dimensional vector.

Next, Amazon Mechanical Turk (AMT) platform was used to create gesture annotations for the gesture categories present in MSRC-12 and CGD 2013 datasets. Each category was annotated by 18-20 AMT workers with respect to the attributes in the database. These annotations are automatically processed to generate the gesture representations. Overall, this dataset consists of 64 gesture attributes, 26 classes (18 categories from CGD 2013 and 8 categories from MSRC-12).

## Gesture Categories

### MSRC-12 Dataset
[MSRC-12](https://www.microsoft.com/en-us/download/details.aspx?id=52283) aka kinect gesture dataset has 12 categories consisting of gestures with full body motions. The data is available only in the form the skeleton information i.e. the RGB-D videos are not available. Overall, each category has 594 examples. In this dataset, the gestures corresponding to the leg motion are eliminated.

The following gesture IDs are eliminated: 2 (duck / crouch or hide), 7 (bow / take a bow to end music session), 9 (had enough / protest the music) and 12 (kick). Refer to the [data description](https://nanopdf.com/download/this-document-microsoft-research_pdf) of MSRC-12 dataset for more information. Rest of the 8 gesture categories are considered. Note that these IDs are removed and the gestures are re-indexed from 1 to 8.

| Gesture ID | Name | Label |
|:----------:|:----:|:-----:|
|      1     | Shoot 		| G1_K_Shoot 		|
|      2     |Throw 		| G2_K_Throw 		|
|      3     |Change Weapon | G3_K_ChangeWeapon |
|      4     | Goggles 		| G4_K_Goggles 		|
|      5     |Start 		| G5_K_Start 		|
|      6     |Next 			| G6_K_Next 		|
|      7     |Wind Up 		| G7_K_WindUp 		|
|      8     |Tempo 		| G8_K_Tempo 		|

### CGD 2013 Dataset
[CGD 2013](http://gesture.chalearn.org/2013-multi-modal-challenge/data-2013-challenge) aka italian gesture dataset has 20 italian gesture categories. This dataset consists of both the RGB-D videos and the 3D skeletal data. There are approximately 400 examples for each class. In our database, two gestures related to the leg motion are eliminated. The eliminated gestures include: *ok* and *messidaccordo*. The original dataset has the class names in italics (refer the column, '**Name**'). However, they are translated to English and the labels are consutructed as given in column, '**Label**'. Rest of the 18 gesture categories are considered.


| Gesture ID | Name | Label || Gesture ID | Name | Label |
|:----------:|:----:|:-----:||:----------:|:----:|:-----:|
| 9  | vieniqui 		| G9_C_ComeHere       || 18 | cheduepalle 	| G18_C_ThatTwoBalls      |
| 10 | prendere 		| G10_C_Take          || 19 | cosatifarei 	| G19_C_WhatWouldIDoToYou |
| 11 | sonostufo 		| G11_C_Tired         || 20 | fame 			| G20_C_Hunger            |
| 12 | chevuoi 			| G12_C_WhatDoYouWant || 21 | noncenepiu 	| G21_C_NoMore            |
| 13 | daccordo 		| G13_C_Agree         || 22 | furbo 		| G22_C_Clever            |
| 14 | perfetto 		| G14_C_Perfect       || 23 | combinato 	| G23_C_Combined          |
| 15 | vattene 			| G15_C_GetOut        || 24 | freganiente 	| G24_C_DoNotWorry        |
| 16 | basta 			| G16_C_Just          || 25 | seipazzo 		| G25_C_Crazy             |
| 17 | buonissimo 		| G17_C_VeryGood      || 26 | tantotempo 	| G26_C_ALongTime         |


## Gesture Attributes
Gesture attributes are key to zero shot learning. Given below are a list of 64 gesture attributes. The approach taken to obtain this list of attributes is described in detail the in the paper. Now, every gesture category is represented as a 64-dimensional binary vector.

| Descriptor | Descriptor | Descriptor | Descriptor | Descriptor | Descriptor | Descriptor | Descriptor |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| LHM_Left | LHM_Iterative | RHM_Backward | OM_Inward | RHD_Left | LHOF_2 | RHOF_4 | POB_Eyes |
| LHM_Right | LHM_Circular | RHM_Clockwise | OM_Outward | RHD_Right | LHOF_3 | RHOF_5 | POB_Nose |
| LHM_Up | LHM_Rectangular | RHM_CounterClockwise | LHD_Left | RHD_Up | LHOF_4 | LHP_Coronal | POB_Mouth |
| LHM_Down | RHM_Left | RHM_Iterative | LHD_Right | RHD_Down | LHOF_5 | LHP_Sagittal | POB_Ear |
| LHM_Forward | RHM_Right | RHM_Circular | LHD_Up | RHD_Forward | RHOF_0 | LHP_Transverse | POB_Head |
| LHM_Backward | RHM_Up | RHM_Rectangular | LHD_Down | RHD_Backward | RHOF_1 | RHP_Coronal | GP_AboveEyes |
| LHM_Clockwise | RHM_Down | OM_Circular | LHD_Forward | LHOF_0 | RHOF_2 | RHP_Sagittal | GP_BetweenEyesAndChest |
| LHM_CounterClockwise | RHM_Forward | OM_Rectangular | LHD_Backward | LHOF_1 | RHOF_3 | RHP_Transverse | GP_BelowChest |

The nomenclature used to describe the gesture attributes is explained here.

| Acronym | Description |
|:-:|:-:|
|LHM|Left hand motion trajectory|
|RHM|Right hand motion trajectory|
|OM|Overall motion trajectory|
|LHD|Left hand direction or orientation|
|RHD|Right hand direction or orientation|
|LHP|Left hand motion plane|
|RHP|Right hand motion plane|
|LHOF|Configuration of left hand (No. of open fingers)|
|RHOF|Configuration of right hand (No. of open fingers)|
|POB|Part of the body referred to|
|GP|General position of gesture|

## Representing Dataset as a Heatmap
Each of the 26 categories are represented as 64 dimensional vectors. In addition to binary values, we also computed continuous values i.e. a value of the descriptor can vary between 0 and 1, where, 0 implies absent, 1 implies present, and any other value indicates that partial presence of an attribute. The binary and continuous semantic description matrices (28 x 64) are visualized in the figures given below.

![Binary gesture description matrix](../master/figures/binary-sd-heatmap.PNG)

![Continuous gesture description matrix](../master/figures/continuous-sd-heatmap.PNG)

## Repository Contents
This section briefly explains the contents of this repository.
* **class_labels.txt:** this file consists of class labels or command names.
* **full_descriptor_names.txt:** this file consists of the labels of the full list of gesture attributes.
* **full_binary_description_matrix.csv:** this file contains the binary gesture description matrix (28 x 64 dimensional tensor). The values are either zeroes or ones.
* **full_continuous_description_matrix.csv:** this file contains the continuous gesture description matrix (28 x 64 dimensional tensor). Each value ranges from 0 to 1.

In our experiments, we used a reduced list of 34 attributes i.e. the descriptors related to motion plane and fingers are removed. The reduced set of attributes and their values are given below.

* **reduced_descriptor_names.txt:** this file consists of the labels of the reduced list of gesture attributes.
* **reduced_binary_description_matrix.csv:** this file contains the binary gesture description matrix (28 x 34 dimensional tensor). The values are either zeroes or ones.
* **reduced_continuous_description_matrix.csv:** this file contains the continuous gesture description matrix (28 x 34 dimensional tensor). Each value ranges from 0 to 1.
* **sd_data.mat:** this file contains the matlab variables related to the class labels, gesture description matrices, descriptor labels, etc.


## How to cite ?

Please cite this article if you use this dataset.

**N. Madapana** and J. Wachs, "Database of Gesture Attributes: Zero Shot Learning for Gesture Recognition," 2019 14th *IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019)*, Lille, France, 2019, pp. 1-8.

## References
1. Lampert, Christoph H., Hannes Nickisch, and Stefan Harmeling. "Attribute-based classification for zero-shot visual object categorization." IEEE transactions on pattern analysis and machine intelligence 36, no. 3 (2013): 453-465.
2. Socher, Richard, Milind Ganjoo, Christopher D. Manning, and Andrew Ng. "Zero-shot learning through cross-modal transfer." In Advances in neural information processing systems, pp. 935-943. 2013.
3. Patterson, Genevieve, and James Hays. "Sun attribute database: Discovering, annotating, and recognizing scene attributes." In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pp. 2751-2758. IEEE, 2012.
4. Fothergill, Simon, Helena Mentis, Pushmeet Kohli, and Sebastian Nowozin. "Instructing people for training gestural interactive systems." In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pp. 1737-1746. 2012.
5. Escalera, Sergio, Jordi Gonzàlez, Xavier Baró, Miguel Reyes, Oscar Lopes, Isabelle Guyon, Vassilis Athitsos, and Hugo Escalante. "Multi-modal gesture recognition challenge 2013: Dataset and results." In Proceedings of the 15th ACM on International conference on multimodal interaction, pp. 445-452. 2013.
