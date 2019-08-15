# DeepLearningShortCourse
Short Course on Deep Learning for the 62nd International Statistical Institute (ISI) World Statistics Congress (https://www.isi2019.org/)

**Tutorials** (Not necesarily in order of presentation)

**Installations**: *(See this first)* Instructions for installing Keras on R or Python. There are two test files to test your installation of Keras (and Tensorflow) using the MNIST dataset.

**3 layer NN** Create a simple 3-layer neural network and show how overfitting and how accuracy varies with the number of hidden layers 

**EuroSat_Classification**: CNN classification based on ResNet, using limited dataset, to demonstrate transfer learning in modeling to obtain a faster solution (code in R and python).

**Pineapple_and_Orange**: This demonstrates image classification as well as using a pretrained network for initializing model weights. This example uses a convolution Neural Network (CNN) showing use of keras in classifying pineapple and orange (images from ImageNet archive), with validation using the MobileNet/ResNet pretrained network. Accuracy is low because it uses a small training dataset on github, but demonstrates basic concepts - code has been adapted from the keras Cat and Dog tutorial (code in R and python).

**Fruit Recognition (CNN in R)** Shows use of keras sequential model (linear stack of layers). A simple sequential convolutional neural net (CNN) with 2 convolutional layers, 1 pooling layer, 1 dense layer is trained and used to predict fruit class/type. 

**Detecting DNA sites**: This demonstrates the use of a 1D CNN or Long Short-Term Memory (LSTM) model to detect cleavage site/binding site in protein sequences (code in R and python). 

**Weather forecasting using a RNN model** This demonstrates a Recurrent Neural Network (RNN) model to forecast the weather

**Detecting Anomolies in TimeSeries using LSTM and GRU models**: This demonstrates the use of a Recurrent Neural Network (RNN) (Long-short term Memory (LSTM) or Gated Recurrent GRU) to detect TimeSeries anomalies in time-series based on quantile prediction (code in R and python).

**Predicting wine-quality**  Predicts wine quality using a deep neural network (DNN) with weights initialized by DBN (Deep Belief Network).

**Hyperparameter optimization (Bayesian)** This example demonstrates the method of Bayesian optimization. 

**DBN** Ths Demonstrates a Deep Belief Network (DBN) model on the MNINT dataset. THe MNIST (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.

**DQN** This example demonstrates deep reinforcement learning for the 'Cartpole' balancing game 
