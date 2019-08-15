from keras.models import load_model

import numpy

model = load_model('wine-model.h5')

predict_me = numpy.array([
    [6.6, 0.16, 0.4, 1.5, 0.044, 48, 143, 0.9912, 3.54, 0.52, 12.4], # Good
    [5.2, 0.405, 0.15, 1.45, 0.038, 10, 44, 0.99125, 3.52, 0.4, 11.6] # Bad
])

predictions = model.predict(predict_me)

rounded = [round(output[0]) for output in predictions]

print(rounded)
