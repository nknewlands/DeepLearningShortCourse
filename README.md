# DeepLearningShortCourse
Short Course on Deep Learning for the 62nd International Statistical Institute (ISI) World Statistics Congress (https://www.isi2019.org/)

**Day1:**

Installations: *(See this first)* Instructions for installing Keras on R or Python. There are two test files to test your installation of Keras (and Tensorflow) using the MNIST dataset.

Pineapple_and_Orange: Convolution Neural Network (CNN) showing use of keras in classifying pineapple and orange (images from ImageNet archive), with validation using the MobileNet/ResNet pretrained network. Accuracy is low because it uses a small training dataset on github, but demonstrates basic concepts - code has been adapted from the keras Cat and Dog tutorial.

EuroSat_Classification: CNN classification based on ResNet, using limited dataset, to demonstrate transfer learning in modeling to obtain a faster solution.

**Day2:**

Detecting DNA sites: This demonstrates the use of a 1D CNN or Long Short-Term Memory (LSTM) model to detect cleavage site/binding site in DNA. Code is adapted from Zou et al. 2018. A primer on deep learning in genomics. Nat. Genet. 2019 Jan(1): 12-18.

TimeSeries Anomaly: This demonstrates the use of a Recurrent Neural Network (RNN) (LSTM or GRU) to detect TimeSeries anomalies in time-series based on quantile prediction.
