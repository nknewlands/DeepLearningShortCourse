# Fruits part 1. R Code
# Inpired by the code from Francois Chollet. 
#                     "Deep learning with Python." Manning (2018) 362 pages.
#                     and
#                     François Chollet with J. J. Allaire
#                     "Deep Learning with R." Manning (2018) 360 pages.
# 
# From François Chollet's books:
# "Convnets are the best type of machine-learning models for computer-vision tasks. 
#  It’s possible to train one from scratch even on a very small dataset, with decent results.
#  On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way,
#  to fight overfitting when you’re working with image data.
#  It’s easy to reuse an existing convnet on a new dataset via feature extraction. 
#  This is a valuable technique for working with small image datasets."
#
# Preparation: Unzip the file fruits_200.zip containing oranges and pinapples pictures 
#              in the same directory as this file. Note, this is a VERY VERY small dataset
#              created using picture from the ImageNet dataset and used for demonstration.
# 
#              Orange: http://image-net.org/synset?wnid=n07747607
#              Pineapple: http://image-net.org/synset?wnid=n07753275
#
#               fruits_200.zip
#               ---------------
#               Training  : 140 (70%)
#               Validation: 30 (15%)
#               Test      : 30 (15%)
#               Training:   orange (72), pineapple (68)
#               Validation: orange (19), pineapple (11)
#               Test :      orange (15), pineapple (15)
#
#               fruits_300.zip
#               ---------------
#               Training  : 210 (70%)
#               Validation: 45 (15%)
#               Test      : 45
#
#               fruits_400.zip
#               ---------------
#               Training  : 280 (70%)
#               Validation: 60  (15%)
#               Test      : 60
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
# DEFINITION OF INPUT DATA                                                     #
################################################################################
train_data_dir <- 'fruits_training'
validation_data_dir <- 'fruits_validation'
test_data_dir <- 'fruits_test'
epochs <- 1    # Number of iteration over the dataset
batch_size <- 24 # Number of images processed at the same time
nb_train_samples <- number_of_files(train_data_dir)
nb_validation_samples <- number_of_files(validation_data_dir)
nb_test_samples <- number_of_files(test_data_dir)
img_width  <- 150 # Image size
img_height <- 150 

################################################################################ 
# CNN                                                                          #
################################################################################

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(img_width, img_height, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>%                       # Last layers
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")
  
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-3, rho=0.9),
  metrics = c("acc")
)
################################################################################ 
# OPTIMIZATION                                                                 #
################################################################################
summary(model) # Display a summary of the model

################################################################################ 
# DATA AUGMENTATION                                                            #
################################################################################

train_datagen <- image_data_generator(
   rescale = 1/255.0,
   shear_range=0.2,
   zoom_range=0.2,
   rotation_range=40,
   width_shift_range=0.2,
   height_shift_range=0.2,
   horizontal_flip=TRUE,
   fill_mode='nearest'
)

validation_datagen <- image_data_generator(rescale = 1/255) # No augmentation on validation data

train_generator <- flow_images_from_directory(
  train_data_dir,  
  train_datagen,  
  target_size = c(img_width, img_height),
  batch_size = batch_size,  
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_data_dir,  
  validation_datagen,  
  target_size = c(img_width, img_height),
  batch_size = batch_size,  
  class_mode = "binary"
)

test_generator <- flow_images_from_directory(
  test_data_dir,  
  validation_datagen,  
  target_size = c(img_width, img_height),
  batch_size = batch_size,  
  class_mode = "binary"
)


# Create checkpoint callback
checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  monitor="val_loss",
  verbose = 1
)
model <- load_model_hdf5("final_model.hdf5")
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = floor(nb_train_samples/batch_size)+1,
  epochs = epochs,
  validation_data = validation_generator,
  validation_steps = floor(nb_validation_samples / batch_size)+1,
  verbose=1,
  callbacks = list(cp_callback)
)
str(history) 

################################################################################ 
# SAVE MODEL                                                                   #
################################################################################
model %>% save_model_hdf5("final_model.hdf5")

# Note: To load model use:
# model <- load_model_hdf5("final_model.hdf5")
################################################################################ 
# VISUALIZE THE TRAINING RESULTS                                               #
################################################################################
png("accuracy.png") # Note: this will display both validation loss and acc.
plot(history,  method ="ggplot2")
dev.off()

################################################################################
# RUN THE MODEL ON THE TEST DATA                                               #
################################################################################
filenames = test_generator$filenames
nb_samples = length(filenames)
y_true_labels = test_generator$classes
y_indices=test_generator$class_indices

# Evaluate all the test sample
evalu = model %>% evaluate_generator(test_generator,steps = floor(nb_samples /batch_size)+1)
cat("Total samples: ",nb_test_samples," Final accuracy: ",evalu$acc,"\n")
test_generator$reset()
predicted = model %>% predict_generator(test_generator,
                                        steps = floor(nb_samples /batch_size)+1,
                                        verbose=1)

y_pred=round(predicted)
labels = c('orange','pineapple')
predictions=labels[y_pred+1] # Note, we add 1 since array start at position one in R
results=data.frame(filenames=filenames,values=round(predicted,2),prediction=predictions)
write.table(results,"predicted.csv", sep=",")
################################################################################
# TO DO                                                                        #
################################################################################
#
# 1. Try to adjust the batch size and the number of epoch
# 2. Add mode layers to the CNN
# 3. Try to increase the number of samples (see fruits_300.zip and fruits_400.zip)
# 4. Part 2. Use pre-trained neural networks
