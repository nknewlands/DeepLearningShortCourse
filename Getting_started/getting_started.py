

#need to install pandas by "conda install pandas" at Anaconda prompt

#importing the required libraries for the MLP model
import keras
from keras.models import Sequential
import numpy as np

#loading the MNIST dataset from keras
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshaping the x_train, y_train, x_test and y_test to conform to MLP input and output dimensions
x_train=np.reshape(x_train,(x_train.shape[0],-1))/255
x_test=np.reshape(x_test,(x_test.shape[0],-1))/255

import pandas as pd
y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)

#performing one-hot encoding on target variables for train and test
y_train=np.array(y_train)
y_test=np.array(y_test)

#defining model with one input layer[784 neurons], 1 hidden layer[784 neurons] with dropout rate 0.4 and 1 output layer [10 #neurons]
model=Sequential()

from keras.layers import Dense

model.add(Dense(784, input_dim=784, activation='relu'))
keras.layers.core.Dropout(rate=0.4)
model.add(Dense(10,input_dim=784,activation='softmax'))

# compiling model using adam optimiser and accuracy as metric
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# fitting model and performing validation

model.fit(x_train,y_train,epochs=50,batch_size=128,validation_data=(x_test,y_test))