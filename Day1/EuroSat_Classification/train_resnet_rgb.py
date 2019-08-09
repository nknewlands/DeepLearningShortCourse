#!/usr/bin/python
# ==============================================================================
# This demonstrate how to use transfer learning using a ResNet50 network.
# Original data from https://arxiv.org/abs/1709.00029
# Etienne Lord - 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
from pathlib import Path
import os

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def number_of_files(dirname):
	cpt = sum([len(files) for r, d, files in os.walk(dirname)])
	return cpt

################################################################################ 
# DATASETS DEFINITION                                                          #
################################################################################
train_data_dir = 'EuroSatRGB_training'
validation_data_dir = 'EuroSatRGB_validation'
test_data_dir = 'EuroSatRGB_test'
nb_train_samples=number_of_files(train_data_dir)
nb_validation_samples=number_of_files(validation_data_dir)
nb_test_samples=number_of_files(test_data_dir)
# Training image dimensions
img_width, img_height = 64, 64

epochs_pre = 10      # Pre-training epoch 
epochs_last = 20     # Complete model epoch
batch_size = 64      # Batch size (adjust according to your avail. memory)

################################################################################ 
# MODEL DEFINITION                                                             #
################################################################################
base_model = ResNet50(weights='imagenet', include_top=False) #Load the ResNet model

x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)
# and a logistic layer with 10 classes (in our dataset)
predictions = Dense(10, activation='softmax')(x)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Model definitions
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

################################################################################ 
# IMAGES LOADING                                                               #
################################################################################

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator() 

# Note, we could use data augmentation, 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
	shuffle = True,
    class_mode='categorical') # Note: the class_mode is categorical

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
	shuffle = True,
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                             patience=5, min_lr=0.001)

tensor=TensorBoard(log_dir='.',histogram_freq=1,embeddings_freq=1,)
csv_logger = CSVLogger('resnet50_rgb_pre_log.csv', append=True, separator=';')

################################################################################ 
# RUN MODEL  (Part 1)                                                          #
################################################################################

# Start the pretraining 
original_hist=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs_pre,
    verbose=1,
    callbacks=[csv_logger],
    validation_data=validation_generator,
    validation_steps= (nb_validation_samples // batch_size))

model.save('resnet50_rgb_first.hdf5')
# At this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

################################################################################ 
# RUN MODEL (Part 2)                                                           #
################################################################################

for layer in model.layers:
   layer.trainable = True

model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

csv_logger = CSVLogger('resnet50_rgb_last_log.csv', append=True, separator=';')
checkpointer = ModelCheckpoint(filepath='resnet50_rgb_weights.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)
original_hist2=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs_last,
    verbose=1,
    callbacks=[csv_logger,checkpointer],
    validation_data=validation_generator,
    validation_steps= (nb_validation_samples // batch_size))

model.save("resnet50_rgb_end.h5")

################################################################################ 
# SAVE MODEL                                                                   #
################################################################################
model.save("resnet50_rgb_final.hdf5")
#
# Note: To load model:
# from keras.models import load_model 
# model=load_model("final_model.hdf5")

################################################################################ 
# FINAL NOTES                                                                  #
################################################################################
#
# 1. This demonstrate how to use transfer learning to train. However, we only use 
#    a very small part of the dataset. Using the full dataset, we can achiever 
#    > 95% accuracy. 
