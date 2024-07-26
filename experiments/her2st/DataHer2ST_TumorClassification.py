# DataHer2ST_TumorClassification.py

"""
This script imports Her2 H&E (Hematoxylin and Eosin) image data and expression data as numpy arrays.
The data is used for tumor classification tasks in the Her2 dataset analysis.

Detailed Steps:
1. Define data directories for H&E images and labeled coordinates.
2. Load and preprocess image data, including resizing and converting to numpy arrays.
3. Load corresponding expression data and merge with image data.
4. Filter and sort image files to remove blank images.
5. Prepare the final dataset for model training and evaluation.

Imports:
- os: For directory and file path operations.
- numpy: For numerical operations and array manipulations.
- pandas: For data manipulation and analysis.
- sklearn: For data splitting and preprocessing.
- PIL (Pillow): For image processing.

Global Variables:
- data_dir_deepspace: Directory for deep space image data.
- data_dir: Directory for labeled coordinates and other related data.
- code_dir: Directory for code and scripts.
- image_size: Size to which images are resized (default is 72x72).
- cancer_type: Type of cancer being analyzed (default is 'invasive cancer').

Functions:
- The script contains loops and operations to read image files, preprocess them, and match them with expression data based on coordinates.

Example Usage:
To run this script, ensure that the directories and paths are correctly set, and that the required libraries are installed.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np
import ensembl #from st-net
import pandas as pd


data_dir_deepspace = '/projects/activities/deepspace/team/yue/data/her2st-data/'
#data_set = 'V1_Adult_Mouse_Brain_Coronal_Section_2'
data_dir = "/projects/li-lab/Yue/DataPool/Spatial/her2st/data"  #SC2200236_EuE-31
code_dir = '/projects/li-lab/Yue/SpatialAnalysis/'

import glob
import os

X = []
Y = []
voxel_ids = []

cancer_type = 'invasive cancer'

image_size = 72
(img_height, img_width) = (image_size, image_size)
image_sizes = (img_height, img_width)
input_shape = (img_height, img_width, 3)

list_of_reps = glob.glob(data_dir+ '/ST-pat/lbl/*.tsv')  #only the ones with annotations

for i in range(len(list_of_reps)):
    data_set = os.path.basename(list_of_reps[i])[0:2]
    
    list_of_files = glob.glob(data_dir_deepspace+data_set+"/voxel_pics/*.png")
    list_of_files = sorted(list_of_files,
                            key =  lambda x: os.stat(x).st_size)
    ####remove smallest 50 images due since they r blank
    #filtered_files = list_of_files[50:]
    filtered_files = list_of_files


    df_Y_origin = pd.read_csv(data_dir+ '/ST-pat/lbl/'+data_set+'_labeled_coordinates.tsv', index_col = 0, sep = '\t')
    print(data_set)
    df_Y = df_Y_origin.loc[:,'label']
    df_Y.index = df_Y_origin.x.round().astype('int').astype('str')+'x'+df_Y_origin.y.round().astype('int').astype('str')

    print(df_Y)


    for f in filtered_files:
        #print(f)
        #im = tf.keras.preprocessing.image.load_img(f, target_size=image_sizes)
        #im_array = tf.keras.preprocessing.image.img_to_array(im)
        im_array = np.asarray(Image.open(f).convert('RGB').resize(image_sizes))
        #print(im_array.shape)

        ar = f.split('/')
        
        voxel_id = ar[-1].replace('.png', '')
        print(voxel_id)
        if voxel_id not in df_Y.index:
            y = 0
        else:
            y = df_Y.loc[voxel_id]

        X.append(f)
        Y.append(y)
        voxel_ids.append(data_set+'_'+voxel_id)

Y = np.array(Y)
print(Y)

Y[Y!=cancer_type] = 0
Y[Y==cancer_type] = 1
Y_filtered = Y

Y = Y.astype(int)
print(Y)