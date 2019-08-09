#!/usr/bin/python
#
# Fruits part 2. Using transfer learning with MobileNet
# Inpired by the code from Francois, Chollet. 
#                     "Deep learning with Python." Manning (2018) 362 pages.
# And 
# https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/
###############################################################################
# Global import                                                               #
###############################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications import nasnet
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras.callbacks import *
from keras import backend as K

###############################################################################
# HELPER FUNCTIONS                                                            #
###############################################################################
def number_of_files(dirname):
    cpt = sum([len(files) for r, d, files in os.walk(dirname)])
    return cpt

################################################################################ 
# DEFINITION OF INPUT DATA                                                     #
################################################################################
test_data_dir='new_fruits'
epochs = 50    # Number of iteration over the dataset
batch_size = 5 # Number of images processed at the same time
nb_test_samples=number_of_files(test_data_dir)
img_width, img_height = 224, 224 # Needed dimensions for most model

################################################################################ 
# ResNet, MobileNet or other                                                   #
################################################################################
#See available models at https://keras.io/applications/#documentation-for-individual-models
model_nasnet = nasnet.NASNetMobile(weights="imagenet",input_shape=(img_width, img_height, 3))

test_generator = ImageDataGenerator(preprocessing_function = nasnet.preprocess_input).flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
	shuffle=False,
	)

predicted=model_nasnet.predict_generator(test_generator,steps = nb_test_samples // batch_size)

y_pred=nasnet.decode_predictions(predicted, top=1)
results=pd.DataFrame(np.concatenate(y_pred), columns=['imagenet_id','predict_class','percent']) 
results["filename"]=test_generator.filenames
print(results)
results.to_csv("predicted.csv", sep=";")

################################################################################
# TO DO                                                                        #
################################################################################
#
# 1. Train for the apples
# 2. Test other models 

