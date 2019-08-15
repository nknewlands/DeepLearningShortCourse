#
# Annomaly detection in time series using LSTM
# Code and data adapted from: https://github.com/cerlymarco/MEDIUM_NoteBook/blob/master/Anomaly_Detection_LSTM/Anomaly_Detection_LSTM.ipynb
# See also: https://towardsdatascience.com/deep-quantile-regression-c85481548b5a
# Modifications: Etienne Lord 2019
# Added GRU and other options
#
# Dataset represent the number of NY taxi trip each 30 minutes
#
###############################################################################
# GLOBAL IMPORT                                                               #
###############################################################################
library(keras)
library(ggplot2)

################################################################################ 
# LOAD  DATA                                                                   #
################################################################################
orders = read.table('nyc_taxi.csv', sep=",",
                    stringsAsFactors = FALSE, 
                    col.names = c("timestamp","value"), 
                    colClasses = c("character","integer"),
                    header = T
                    )

#'timestamp','value','yr','mt','d',''
orders$timestamp=as.POSIXlt(orders$timestamp, format="%Y-%m-%d %H:%M:%S")
orders$H <- orders$timestamp$hour        # hour
orders$d <- orders$timestamp$wday        # weekday
orders$mt <- orders$timestamp$mon+1     # month of year (zero-indexed)
orders$yr <- orders$timestamp$year+1900  # years since 1900
orders$timestamp=as.POSIXct(orders$timestamp)
df=orders #remove first row
head(df)

# Create a new column called 'weekday_hour' with the concatenation 
# of the weekday and the our
df$weekday_hour <- paste0(df$d,"-",df$H)

#We could do better, but this explain what we need to do...
mean_value<-function(df) {
 m_value=c()
 for (i in 1:nrow(df)) {
   w_h=df[i,]$weekday_hour
   value=mean(df[df$weekday_hour==w_h,]$value)
   m_value=c(m_value,value)
   
 }
  return (m_value)
}

df$m_weekday <- mean_value(df) # Create a column with the mean value for 1 hour


################################################################################ 
# LOAD  DATA                                                                   #
################################################################################
### CREATE GENERATOR FOR LSTM ###
sequence_length <- 48 # We learn 48 points at a time 

# Generate a series of values to be learn
# We essentially want to learn to associate 
# a series of m_weekday to the "normal" New Yor Taxi activity value

cnt<-list()

index<-1
for (position in 1:(nrow(df)-sequence_length)) {
  val=c(log(df$value[position:(position+sequence_length-1)]))
  mean=c(log(df$m_weekday[position:(position+sequence_length-1)]))
  cnt[[index]]<-list(val-mean)
  index<-index+1
}

# We start at sequence_length to the end 
labels=c()
inits=c()
for (position in sequence_length:(nrow(df)-1)) {
  val=c(log(df$value[position]))
  mean=c(log(df$m_weekday[position]))
  inits=c(inits,mean)
  labels=c(labels,val-mean)
}
cnt=array_reshape(unlist(cnt),c(length(labels),sequence_length,1))
    
x_train=array_reshape(cnt[1:5000,,],c(5000,sequence_length,1))
x_test=array_reshape(cnt[5000:nrow(cnt),,],c(nrow(cnt)-4999,sequence_length,1))
y_train=labels[1:5000]
y_test=labels[5000:length(labels)]


### DEFINE QUANTILE LOSS ###
### See: https://towardsdatascience.com/deep-quantile-regression-c85481548b5a
### And https://github.com/rstudio/keras/issues/451
q_loss <- function(q, y, f) {
  e <- y - f
  k_mean(k_maximum(q * e, (q - 1) * e), axis = 2)
}

losses=list(
  function(y_true, y_pred) q_loss(0.1, y_true, y_pred),
  function(y_true, y_pred) q_loss(0.5, y_true, y_pred),
  function(y_true, y_pred) q_loss(0.9, y_true, y_pred)
)


################################################################################ 
# CREATE MODEL 1 - LSTM                                                        #
################################################################################

input <- layer_input(shape=c(sequence_length,1))
base_model <- input %>%
  layer_lstm(64,return_sequences = TRUE, dropout = 0.3, trainable = TRUE) %>%
  layer_lstm(16,return_sequences = FALSE, dropout = 0.3, trainable = TRUE) %>%
  layer_dense(50) 

# add outputs
out10 <- base_model %>% 
  layer_dense(units = 1, name="out10") 

out50 <- base_model %>% 
  layer_dense(units = 1, name="out50") 

out90 <- base_model %>% 
  layer_dense(units = 1, name="out90") 

model<-keras_model(input, list(out10,out50,out90))

model %>% compile(
  optimizer = "adam",
  loss = losses,
  loss_weights = c(0.3,0.3,0.3)
)

summary(model)

################################################################################ 
# CREATE MODEL 2 - GRU                                                         #
################################################################################
# 
# base_model2 <- input %>%
#   layer_gru(64,return_sequences = TRUE, dropout = 0.3, trainable = TRUE) %>%
#   layer_gru(16,return_sequences = FALSE, dropout = 0.3, trainable = TRUE) %>%
#   layer_dense(50) 
# 
# # add outputs
# out10_2 <- base_model2 %>% 
#   layer_dense(units = 1, name="out10") 
# 
# out50_2 <- base_model2 %>% 
#   layer_dense(units = 1, name="out50") 
# 
# out90_2 <- base_model2 %>% 
#   layer_dense(units = 1, name="out90") 
# 
# model2<-keras_model(input, list(out10_2,out50_2,out90_2))
# 
# model2 %>% compile(
#   optimizer = "adam",
#   loss = losses,
#   loss_weights = c(0.3,0.3,0.3)
# )
# 
# summary(model2)

################################################################################ 
# RUN BOTH MODELS                                                              #
################################################################################

epochs <- 10 # Number of training steps (should be ~100 or more)
history = model %>% fit(
  x_train, 
 list(y_train,y_train,y_train), 
  epochs=epochs, 
  batch_size=256, 
  verbose=1, 
  shuffle=TRUE
)

#history2 = model2.fit(X_train, [y_train,y_train,y_train], epochs=epochs, batch_size=256, verbose=1, shuffle=True)


################################################################################ 
# VISUALIZE THE COMPUTED QUANTILE                                              #
################################################################################
### QUANTILEs BOOTSTRAPPING ###


## Note: change model to model2 here for GRU, and this is the exact translation of the python code
##       however, we can also use the predict method which return the 3 outputs.
#NN = k_function(list(model$layers[[1]]$input,k_learning_phase()), 
#                list(model$layers[[5]]$output,model$layers[[6]]$output,model$layers[[7]]$output))


## Do some prediction for x epochs
pred_10=c()
pred_50=c()
pred_90=c()
for (i in 1:epochs) {
    #predd = NN(list(x_test)) # Similar to the python code
    predd <- model %>% predict(x_test)
    pred_10=c(pred_10,predd[[1]])
    pred_50=c(pred_50,predd[[2]])
    pred_90=c(pred_90,predd[[3]])
}    

pred_10 <- array_reshape(pred_10,c(epochs,nrow(x_test)))
pred_50 <- array_reshape(pred_50,c(epochs,nrow(x_test)))
pred_90 <- array_reshape(pred_90,c(epochs,nrow(x_test)))

pred_10_m = exp(apply(pred_10,2,mean,probs=0.1)+ inits[5000:length(inits)])
pred_50_m = exp(apply(pred_50,2,mean,probs=0.5)+ inits[5000:length(inits)])
pred_90_m = exp(apply(pred_90,2,mean,probs=0.9)+ inits[5000:length(inits)])

### PLOT QUANTILE PREDICTIONS ###
qplot((1:length(pred_10_m)), (pred_90_m-pred_10_m),  geom = "path", xlab="Time", ylab="Number of travels")


################################################################################
# TO DO                                                                        #
################################################################################
#
# 1. Try to adjust the batch size and the number of epoch.
# 2. Try to add more LSTM layers or other memory cells.
# 3. Try to train on more data.