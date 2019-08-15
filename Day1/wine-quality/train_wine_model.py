from keras.models import Sequential
from keras.layers import Dense
import numpy

dataset = numpy.loadtxt("wine-data.csv", delimiter=";")

input = dataset[:, 0:11]
output = dataset[:, 11]

# since the data comes in a scale of 0 to 10, this is needed to we get a simple true or false
output = [(round(each / 10)) for each in output]

model = Sequential()
model.add(Dense(20, input_dim=11, init='uniform', activation='relu'))
model.add(Dense(12, init='uniform', activation='relu'))
model.add(Dense(4, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(input, output, epochs=5000, batch_size=50, verbose=2)

model.save('wine-model.h5')
