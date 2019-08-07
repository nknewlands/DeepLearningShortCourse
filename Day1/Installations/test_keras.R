library(keras)

# Load dataset
mnist <- dataset_mnist()

# Rescale images and normalize
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# Create the model
model <- keras_model_sequential() %>%
layer_dense(units=512, activation='relu',input_shape=c(28 * 28)) %>%
layer_dense(units=10, activation='softmax')
# Compile the model 
model %>% compile(
	optimizer = "rmsprop",
	loss = "categorical_crossentropy",
	metrics = c("accuracy")
)
# Run the model
histo=model %>% fit(
	train_images, train_labels,
	epochs = 5, batch_size=128,
	validation_data=list(test_images,test_labels)
)
# Alternative computation of accuracy 
results <- model %>% evaluate(test_images, test_labels)
results