# DeepLearningShortCourse
Short Course on Deep Learning for the 62nd International Statistical Institute (ISI) World Statistics Congress (https://www.isi2019.org/)


Directories:

Installations: Instructions for installing Keras on R or Python. There are two test files to test your installation of keras (and tensorflow) using the MNIST dataset.

Pineapple_and_Orange: Convolution Neural Network (CNN) showing use of keras in classifying pineapple and orange (images from ImageNet archive), with validating using the MobileNet pretrained network. Accuracy is low because it uses a small training dataset on github, but demonstrates basic concepts - code has been adapted from Cat and Dog training tutorial.

EuroSat_Classification: CNN classification based on ResNet, using limited dataset to demonstrate modeling and to obtain a faster solution.

Detecting DNA sites: This demonstrates the use of a 1D CNN or Long Short-Term Memory (LSTM) model to detect cleavage site/binding site in DNA. Code is adapted from Zou et al. 2018. A primer on deep learning in genomics. Nat. Genet. 2019 Jan(1): 12-18.

TimeSeries_Anomoly: This demonstrates use of a Recurrent Neural Network (RNN) (LSTM or GRU) to detect anomolies in time-series based on quantile prediction.

