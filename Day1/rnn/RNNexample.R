
library(tibble)
library(readr)
library(ggplot2)
library(keras)
library(tensorflow)

library(base)
library(rstudioapi)

get_directory <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    file <- "--file="
    rstudio <- "RStudio"
    
    match <- grep(rstudio, args)
    if (length(match) > 0) {
        return(dirname(rstudioapi::getSourceEditorContext()$path))
    } else {
        match <- grep(file, args)
        if (length(match) > 0) {
            return(dirname(normalizePath(sub(file, "", args[match]))))
        } else {
            return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
        }
    }
}

setwd(get_directory())
filename <- "jena_climate_2009_2016.csv"
data <- read_csv(filename)

glimpse(data)

ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()

ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()


data <- data.matrix(data[,-1])

train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

#data - The original array of floating-point data, which you normalized in listing 6.32.
#lookback - How many timesteps back the input data should go.
#delay - How many timesteps in the future the target should be.
#min_index and max_index - Indices in the data array that delimit which timesteps to draw from. 
#shuffle - Whether to shuffle the samples or draw them in chronological order.
#batch_size - The number of samples per batch.
#step - The period, in timesteps, at which you sample data. Step=6 for one data point every hour

#A generator function is a special type of function that you call repeatedly to obtain a sequence of values 

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1, 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}

# build three generators: one for training, one for validation, and one for testing

# forecasting window parameters 

# lookback = 1440 - Observations will go back 10 days
# steps = 6 - Observations will be sampled at one data point per hour
# delay = 144 - Targets will be 24 hours in the future

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps <- (300000 - 200001 - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data) - 300001 - lookback) / batch_size


#baseline/benchmark to compare to Deep Learning (DL)


evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method()


# Keras Model composed of a linear stack of layers 
# Long Short-Term Memory (LSTM) Layer
# this approach flattens the time series, removing the notion of time from the input data
# MAE of 0.29

model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae")

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 2,
# can set epochs to higher value (20)
  validation_data = val_gen,
  validation_steps = val_steps,
  callbacks = callback_tensorboard(log_dir = "logs/run_a")
)

# callback_tensorboard(log_dir = "logs", histogram_freq = 0,
# write_graph = TRUE, write_images = FALSE, embeddings_freq = 0,
# embeddings_layer_names = NULL, embeddings_metadata = NULL)

# set global default to never show metrics
#options(keras.view_metrics = FALSE)

plot(history)

# launch TensorBoard (data won't show up until after the first epoch)
tensorboard("logs/run_a")

#
# Recurrent-sequence processing model to exploit temporal ordering of data points
# Gated recurrent unit (GRU) layer (Chung et al. in 2014)
# Trades-off computational expensiveness and representational power
# MAE of ~0.265 
# training and validation curves that the model is overfitting: 
# the training and validation losses start to diverge considerably after a few epochs


model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 2,
  # can set epochs to higher value (20)
  validation_data = val_gen,
  validation_steps = val_steps
)

plot(history)
tensorboard(c("logs/run_a", "logs/run_b"))


# USING RECURRENT DROPOUT TO FIGHT OVERFITTING
# employ "dropout" which randomly zeros out input units of a layer 
# in order to break happenstance correlations in the training data
# will no longer be overfitting during the first 20 epochs.

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 2,
  # can set epochs to higher value (40)
  validation_data = val_gen,
  validation_steps = val_steps
)

# STACKING RECURRENT LAYERS
# hit a performance bottleneck, so increase the capacity of the network. 
# (increasing number of units in the layers or adding more layers)
# To stack recurrent layers on top of each other in Keras, all intermediate 
# layers should return their full sequence of outputs (a 3D tensor) rather 
# than their output at the last timestep. 
# Specify: return_sequences = TRUE

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 2,
  # can set epochs to higher value (40)
  validation_data = val_gen,
  validation_steps = val_steps
)





