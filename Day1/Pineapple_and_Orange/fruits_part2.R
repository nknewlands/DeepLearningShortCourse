#
# Fruits part 2. Using transfer learning with MobileNet
# Inpired by the code from Francois, Chollet. 
#                     "Deep learning with Python." Manning (2018) 362 pages.
#                     and
#                     François Chollet with J. J. Allaire
#                     "Deep Learning with R." Manning (2018) 360 pages.
# https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/
#
# Before running this, unzip the file new_fruits.zip
# 
###############################################################################
# Global import                                                               #
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
test_data_dir <- 'new_fruits'
epochs <- 50    # Number of iteration over the dataset
batch_size <- 5 # Number of images processed at the same time
nb_test_samples=number_of_files(test_data_dir)
img_width <- 224 # Needed dimensions for most model
img_height <- 224 

################################################################################ 
# ResNet, MobileNet or other                                                   #
################################################################################
#See available models at https://cran.rstudio.com/web/packages/keras/vignettes/applications.html
resnet <- application_resnet50(weights="imagenet")

image_generator <- image_data_generator(preprocessing_function = imagenet_preprocess_input)

test_generator <- flow_images_from_directory(
     test_data_dir,
     image_generator,
     target_size=c(img_width, img_height),
     batch_size=batch_size,
     shuffle=FALSE
 )

predicted=resnet %>% predict_generator(test_generator,steps = floor(nb_test_samples / batch_size)+1)

y_pred <- imagenet_decode_predictions(predicted, top=1)

results=do.call(rbind.data.frame,y_pred) #Generate a data.frame
results["filename"]=test_generator$filenames
print(results)
write.table(results,"predicted.csv", sep=",")
################################################################################
# TO DO                                                                        #
################################################################################
#
# 1. Train for the apples
# 2. Test other models

