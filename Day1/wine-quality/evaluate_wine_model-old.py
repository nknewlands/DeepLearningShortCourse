from keras.models import load_model

import numpy

dataset = numpy.loadtxt("wine-data.csv", delimiter=";")
input = dataset[:, 0:11]
output = dataset[:, 11]

# since the data comes in a scale of 0 to 10, this is needed to we get a simple true or false
output = [(round(each / 10)) for each in output]

model = load_model('wine-model.h5')

scores = model.evaluate(input, output)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))