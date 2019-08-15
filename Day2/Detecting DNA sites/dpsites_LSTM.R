#!/usr/bin/python
#
# Original code from:  
# Zou, J., Huss, M., Abid, A., Mohammadi, P., Torkamani, A., & Telenti, A. (2018). A primer on deep learning in genomics. 
# Nat Genet. 2019 Jan;51(1):12-18. doi: 10.1038/s41588-018-0295-5.
#
# Modification: Etienne Lord 2019
#
# See also for review: Yue, T., & Wang, H. (2018). Deep learning for genomics: A concise overview. arXiv preprint arXiv:1802.00810.
# https://arxiv.org/abs/1802.00810
#
# Identify some potential Plum pox virus DNA cleaving sites.
# File: sequences.txt (Cleavage site)
#       labels.txt    (Information about each site: 0 negative, 1 positive)
#       test.fasta    (Sample fasta file)
#
###############################################################################
# GLOBAL IMPORT                                                               #
###############################################################################
library(keras)
library(ggplot2)
library(seqinr)

################################################################################ 
# HELPER FUNCTION   
# See: https://stackoverflow.com/questions/38620424/label-encoder-functionality-in-r
################################################################################

label_encoder = function(vec){
  levels = sort(unique(as.factor(vec)))
  function(x){
    match(x, levels)
  }
}


################################################################################ 
# PROCESS THE SITES INFORMATIONS                                               #
################################################################################

sequence <- read.table("sequences.txt", col.names = c("Sequences","Sites"), stringsAsFactors = FALSE)

################################################################################ 
# ENCODE THE DNA INFORMATION USING ONE HOT ENCODING                            #
################################################################################
# The LabelEncoder encodes a sequence of bases as a sequence of integers.

label_encoder_prot=label_encoder(unlist(strsplit(sequence$Sites,""))) #Create encoder

seq_array<-to_categorical(label_encoder_prot(unlist(strsplit(sequence$Sites,""))))

sequence$integer_encoded <- array_reshape(seq_array,c(nrow(sequence),nchar(sequence$Sites[1]),22))

################################################################################ 
# LOAD LABELS FOR EACH SITES AND ENCODE                                        #
################################################################################
label_file <- read.table("labels.txt", col.names=c("Sequences","Labels"))
label_file$input_labels <- to_categorical(label_file$Labels)

train_features=sequence$integer_encoded
train_labels=label_file$input_labels


################################################################################ 
# MODEL DEFINITION (LSTM, one layer)                                            #
################################################################################

input <- layer_input(shape=c(sequence_length,1))
model <- keras_model_sequential() %>% 
         layer_lstm(32,return_sequences = TRUE, 
                    dropout = 0.1, 
                    trainable = TRUE, 
                    input_shape=c(nchar(sequence$Sites[1]), 22)) %>%
         layer_flatten() %>%                       
         layer_dense(units = 16, activation = "relu") %>% 
         layer_dense(units = 2, activation = "softmax") # Sites

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("binary_accuracy")
)

summary(model)

################################################################################ 
# RUN MODEL                                                                    #
################################################################################
history = model%>% fit(train_features, train_labels,epochs=50, verbose=1, validation_split=0.25)

          
################################################################################ 
# TEST ON REAL DATA (test.fasta)                                               #
################################################################################

proteome=read.fasta("test.fasta",as.string=T,forceDNAtolower=F)

## Predict for each sequence of len 9 bp
for (s in proteome) {
  prot=as.character(unlist(getSequence(s,as.string=T))) #Vraiment laid
  name=unlist(getAnnot(s))
  seqlist <- c()
  index <- 1
  for (j in seq(from=1,to=nchar(prot)-9,by=1)) {
    to_test=substring(prot,j,(j+8))
    seqlist <- c(seqlist,to_test)
  }
  seq_array<-to_categorical(label_encoder_prot(unlist(strsplit(seqlist,""))))
  seq_features <- array_reshape(seq_array,c(length(seqlist),9,22))
  predicted_labels <- model %>% predict(seq_features)
  predicted_labels = predicted_labels[,2] # we want the positive sites
  print(qplot(seq(1,length(predicted_labels)),predicted_labels, main=name, ylab="prob. of cleavages site (%)",xlab="relative position", geom=c("line")))
}

################################################################################ 
# FINAL NOTES                                                                  #
################################################################################
#
# 1. This demonstrate how to use LSTM to find some interesting sites in proteic
#    sequences. 
# 2. TO DO, transform this model to a GRU.
# 3. Adjust dropout to avoid overfitting.
# 4. What if we really have DNA sequences? What should be changed?
