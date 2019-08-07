# Test MNIST
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels) #Converts a class vector (integers) to binary class matrix. (utils)
test_labels = to_categorical(test_labels)

# create the deep learning network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compile the network and run deep learning 
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

original_hist=network.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images,test_labels))

# Compute a discrete value for accuracy 
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Validation accuracy on test set:', test_acc)

acc = original_hist.history['acc']
val_acc = original_hist.history['val_acc']
loss = original_hist.history['loss']
val_loss = original_hist.history['val_loss']
epochs = range(1, len(acc) + 1)

# Plot figures
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(5, 5))
fig.subplots_adjust(hspace=0.5)

ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


ax2.plot(epochs, acc, 'bo', label='Training acc.')
ax2.plot(epochs, val_acc, 'b', label='Validation acc.')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

