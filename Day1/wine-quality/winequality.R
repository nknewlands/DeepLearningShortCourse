

library(caret)
library(deepnet)

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

# Download dataset, if it does not exist.
fileName <- 'winequality-red.csv';
if (!file.exists(fileName)) {
  download.file(paste0('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/', fileName), fileName, method="curl")
}
fileName <- 'winequality-white.csv';
if (!file.exists(fileName)) {
  download.file(paste0('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/', fileName), fileName, method="curl")
}

data <- read.csv('winequality-white.csv', sep=';')
data <- rbind(data, read.csv('winequality-red.csv', sep=';'))

partition <- createDataPartition(data$quality, p = 0.75)[[1]]
train <- data[partition,]
test <- data[-partition,]

train <-as.matrix(train)
y<-data$quality


# Training a Deep neural network with weights initialized by DBN

fit <- dbn.dnn.train(train, y,  
                         #hidden=c(64),
                         #learningrate = 0.1, 
                         cd=0.00005,
                         #,initW = NULL, initB = NULL,
                         activationfun="tanh", hidden=c(5,5), learningrate=0.5, momentum=0.5, 
                         learningrate_scale=1, 
                         output="sigm"
                         , 
                         numepochs=10
                         , batchsize=100
                         #, hidden_dropout=0, visible_dropout=0
)

results <- nn.predict(fit, test)



