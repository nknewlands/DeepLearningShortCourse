#==============================================================================
# This demonstrate how to use transfer learning using a ResNet50 network.
# Original data from https://arxiv.org/abs/1709.00029
# Etienne Lord - 2019
# Note: this required EuroSatRGB images
#
# A subset is found in : EuroSatRGB_very_small.zip
#
# Citation: Helber P, Bischke B, Dengel A, Borth D. Eurosat: A novel dataset and 
# deep learning benchmark for land use and land cover classification. arXiv 
# preprint arXiv:1709.00029. 2017 Aug 31.
#==============================================================================

###############################################################################
# GLOBAL IMPORT                                                               #
###############################################################################
library(keras)
library(ggplot2)
################################################################################ 
# HELPER FUNCTIONS                                                             #
################################################################################
number_of_files<-function(dirname) {
    return (length(list.files(dirname, recursive=TRUE)))
}
################################################################################ 
# DATASETS DEFINITION                                                          #
################################################################################
train_data_dir <- 'EuroSatRGB_training'
validation_data_dir <- 'EuroSatRGB_validation'
test_data_dir <- 'EuroSatRGB_test'
nb_train_samples <- number_of_files(train_data_dir)
nb_validation_samples <- number_of_files(validation_data_dir)
nb_test_samples <-number_of_files(test_data_dir)
# Training image dimensions
img_width  <- 64
img_height <- 64

epochs_pre  <- 10      # Pre-training epoch 
epochs_last <- 20      # Complete model epoch
batch_size  <- 64      # Batch size (adjust according to your avail. memory)

################################################################################ 
# MODEL DEFINITION                                                             #
################################################################################
# See: https://cran.rstudio.com/web/packages/keras/vignettes/applications.html
base_model <- application_resnet50(weights='imagenet', include_top=FALSE) #Load the ResNet model

predictions <- base_model$output %>% 
     layer_global_average_pooling_2d() %>%
     layer_dense(units = 1024, activation = "relu") %>%
     layer_dropout(rate = 0.25) %>%
     layer_dense(units = 1, activation = "sigmoid") %>%
     layer_dense(units=10, activation='softmax')

# first: train only the top layers (which were randomly initialized)
for (layer in base_model$layers) {
  layer$trainable = FALSE
}

# Model definitions
model <-  keras_model(inputs=base_model$input, outputs=predictions)
summary(model)

# Compile the model (should be done *after* setting layers to non-trainable)
model %>% compile(optimizer = optimizer_rmsprop(lr = 0.001), 
                  loss = 'categorical_crossentropy', metrics=c('acc'))

################################################################################ 
# IMAGES LOADING                                                               #
################################################################################

train_datagen <- image_data_generator(
  rescale = 1/255.0,
  shear_range=0.2,
  zoom_range=0.2,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  horizontal_flip=TRUE,
  fill_mode='nearest',
)

validation_datagen <- image_data_generator(rescale = 1/255) # No augmentation on validation data

train_generator <- flow_images_from_directory(
  train_data_dir,  
  train_datagen,  
  target_size = c(img_width, img_height),
  batch_size = batch_size,  
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  validation_data_dir,  
  validation_datagen,  
  target_size = c(img_width, img_height),
  batch_size = batch_size,  
  class_mode = "categorical"
)

################################################################################ 
# RUN MODEL  (Part 1)                                                          #
################################################################################

original_hist <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = floor(nb_train_samples/batch_size)+1,
  epochs = epochs_pre,
  validation_data = validation_generator,
  validation_steps = floor(nb_validation_samples / batch_size)+1,
  verbose=1,
  #callbacks = list(cp_callback)
)
save_model_hdf5("resnet50_rgb_end.h5")('resnet50_rgb_first.hdf5')

# At this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

################################################################################ 
# RUN MODEL (Part 2)                                                           #
################################################################################

for (layer in model$layers) {
  layer$trainable = TRUE
}
#Note: we could also use unfreeze_weights(model)
model %>% compile(optimizer = optimizer_rmsprop(lr = 0.0001), 
                  loss = 'categorical_crossentropy', metrics=c('acc'))

dir.create("logs", showWarnings = FALSE)
cp_lists = list(
  # save best model after every epoch
  callback_model_checkpoint("checkpoint.hdf5", save_best_only = TRUE),
  # only needed for visualising with TensorBoard
  callback_tensorboard(log_dir = "logs")
)

original_hist2 <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = floor(nb_train_samples/batch_size)+1,
  epochs = epochs_last,
  validation_data = validation_generator,
  validation_steps = floor(nb_validation_samples / batch_size)+1,
  callbacks = cp_lists,
  verbose=1
)

save_model_hdf5("resnet50_rgb_final.hdf5")

