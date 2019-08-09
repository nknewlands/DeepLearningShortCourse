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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
import itertools


################################################################################ 
# PROCESS THE SITES INFORMATIONS                                               #
################################################################################

sequence_file = open("sequences.txt", "r")
sequences = sequence_file.read().split('\n')
sequences = list(filter(None, sequences))  # This removes empty sequences.
for i in range(len(sequences)):    
    sequences[i]=sequences[i].split('\t')[1]

pd.DataFrame(sequences, index=np.arange(1, len(sequences)+1),columns=['Sequences']).head()

################################################################################ 
# ENCODE THE DNA INFORMATION USING ONE HOT ENCODING                            #
################################################################################
# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(categories=[range(40)])   
input_features = []
for sequence in sequences:
  integer_encoded = integer_encoder.fit_transform(list(sequence))
  integer_encoded = np.array(integer_encoded).reshape(-1, 1)
  one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
  input_features.append(one_hot_encoded.toarray())

input_features = np.stack(input_features)

################################################################################ 
# LOAD LABELS FOR EACH SITES AND ENCODE                                        #
################################################################################
label_file = open("labels.txt", "r")

labels = label_file.read().split('\n')
labels = list(filter(None, labels))  # removes empty sequences
for i in range(len(labels)):    
    labels[i]=labels[i].split('\t')[1]

one_hot_encoder2 = OneHotEncoder(categories=[range(2)])
labels = np.array(labels).reshape(-1, 1)
input_labels = one_hot_encoder2.fit_transform(labels).toarray()

## SPLIT THE TRAINING DATA 75% TRAINING / 25% VALIDATION
train_features, test_features, train_labels, test_labels = train_test_split(input_features, input_labels, test_size=0.25, random_state=42)


################################################################################ 
# MODEL DEFINITION (1D CNN)                                                    #
################################################################################
# Model
model = Sequential() # Adjust the conv size
model.add(Conv1D(filters=32, kernel_size=5,input_shape=(train_features.shape[1], 40)))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['binary_accuracy'])
model.summary()

################################################################################ 
# RUN MODEL                                                                    #
################################################################################
history = model.fit(train_features, train_labels,epochs=50, verbose=1, validation_split=0.25)

print('Confusion matrix:\n',cm)

predicted_labels = model.predict(np.stack(test_features))
cm = confusion_matrix(np.argmax(test_labels, axis=1), 
                      np.argmax(predicted_labels, axis=1))

          
################################################################################ 
# TEST ON REAL DATA (test.fasta)                                               #
################################################################################

def load_fasta(fa_file):
    # Variable data 
    data = []
    # Open the sequences file
    for record in SeqIO.parse(fa_file, "fasta"):
            data.append([record.id, record.seq.upper(), None])
    # Return data
    return data

proteome=load_fasta("test.fasta") # Load fasta files
df=pd.DataFrame(columns=['name','sequence','probs','max_score'])
file = open("results.txt","w")  # Results file

## Predict for each sequence of len 9 bp
for i in range(len(proteome)):
    seqname=proteome[i][0]
    print('[ %d/%d %s]'%(i,len(proteome),seqname))
    seq=str(proteome[i][1])
    seqlist=list()
    by=2
    for j in range(0,len(seq),by):
        to_test=seq[j:(j+9)]
        seqlist.append(to_test)       
    seq_features = []
    for sequence in seqlist:            
            if len(sequence)==9:
                integer_encoded = integer_encoder.fit_transform(list(sequence))
                integer_encoded = np.array(integer_encoded).reshape(-1, 1)
                one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
                seq_features.append(one_hot_encoded.toarray())  
    if len(seq_features)>1:
        predicted_labels = model.predict(np.stack(seq_features))    
        file.write("%s\t%f\t%s\t%s\n"%(seqname,np.max(predicted_labels[:,1]),seq,','.join(map(str, predicted_labels[:,1]))))
     
file.close()
################################################################################ 
# FINAL NOTES                                                                  #
################################################################################
#
# 1. This demonstrate how to use 1D CNN to find some interesting sites in DNA
#    sequences. 
# 2. See the file dpsites_gru.py for the use of RNN.