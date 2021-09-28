# General imports
import pandas as pd
import numpy as np
from collections import Counter

# Defined functions
import preprocessing_functions as preproc
#import preprocessing_functions_for_final_runs as preproc_final

# For data preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For the CNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Activation, Dropout, Dense
from tensorflow.keras.layers import MaxPooling1D, concatenate, Flatten
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras import regularizers

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Getting the data files
qa_data_train = pd.read_csv('Data/Final_QA_datasets/qa_data_train.csv', sep='\t', index_col=0)
qa_data_tune = pd.read_csv('Data/Final_QA_datasets/qa_data_tune.csv', sep='\t', index_col=0)
#qa_data_dev = pd.read_csv('Data/Final_QA_datasets/qa_data_dev.csv', sep='\t', index_col=0)
#qa_data_test = pd.read_csv('Data/Final_QA_datasets/qa_data_test.csv', sep='\t', index_col=0)

number_of_labels = 4
maxlen = 200

train_sentences = preproc.exclude_labels(qa_data_train,[5,6],"Goldstandard")
tune_sentences = preproc.exclude_labels(qa_data_tune,[5,6],"Goldstandard")
#dev_sentences = preproc.exclude_labels(qa_data_dev,[5,6],"Goldstandard")
#test_sentences = exclude_labels(qa_data_test,[5,6],"Goldstandard")

#Concatenating train and tune to use for training
train_sentences = pd.concat([train_sentences, tune_sentences]).reset_index()

Y_train = np.array([int(i) for i in train_sentences["Goldstandard"]])
Y_train = np.delete(Y_train, 4678, axis=0)  #removing the stupid index, INDEX IS MISSING ON Q+A AND A
Y_train = preproc.one_hot_encoder(Y_train, number_of_labels)

#Y_dev = np.array([int(i) for i in dev_sentences["Goldstandard"]])
#Y_dev = preproc.one_hot_encoder(Y_dev, number_of_labels)
#Y_test = np.array([int(i) for i in test_sentences["Goldstandard"]])

#Loading the Bert embeddings
X_train_Q = np.load("Bert_embs/friends_Q_bert_embs_train.npy", mmap_mode="r")
X_train_Q = np.delete(X_train_Q, 4678, axis=0)  #removing the stupid index, INDEX IS MISSING ON Q+A AND A, so have to remove from Q as well
#X_dev_Q = np.load("Bert_embs/friends_Q_bert_embs_dev.npy", mmap_mode="r")
#X_test_Q = np.load("Bert_embs/friends_Q_bert_embs_test.npy", mmap_mode="r")

X_train_A = np.load("Bert_embs/friends_A_bert_embs_train.npy", mmap_mode="r")
#X_dev_A = np.load("Bert_embs/friends_A_bert_embs_dev.npy", mmap_mode="r")
#X_test_A = np.load("Bert_embs/friends_A_bert_embs_test.npy", mmap_mode="r")


# Function for making convolutions
def convolutions(emb_layer, maxlen, act, num_filters, k_size, dropoutrate, num_conv):
    
    all_convs = []
    
    for i in range(num_conv):
        conv = Conv1D(num_filters, kernel_size=k_size[i], activation=act)(emb_layer)
        maxpool = MaxPooling1D(pool_size=maxlen-k_size[i]+1)(conv)
        flatten = Flatten()(maxpool)
        dropout = Dropout(dropoutrate)(flatten)
        
        all_convs.append(dropout)
    
    concat_model = concatenate([i for i in all_convs], axis=-1)
    
    return concat_model


# single input - Single channel
def CNN(num_filters, k_size, batch_s, num_conv, reg=0.5, opp='adam', act='relu', dropoutrate=0.5):
    # Model for Answers 
    # Defining the Embedding layer and the input 
    input_Q = Input(shape=(maxlen,768))

    # Obtaining the concatenated convolutional layers 
    concat_model_Q = convolutions(input_Q, maxlen, act, num_filters, k_size, dropoutrate, num_conv[0])

    # Model for Answers 
    # Defining the Embedding layer and the input 
    input_A = Input(shape=(maxlen,768))

    # Concatenating the concatenated convolutional layers
    concat_model_A = convolutions(input_A, maxlen, act, num_filters, k_size, dropoutrate, num_conv[1])

    # Concatenating Q and A 
    concat_model_QA = concatenate([concat_model_Q, concat_model_A], axis=-1) 

    # Making the final model 
    output = Dense(number_of_labels, activation='softmax', kernel_regularizer=regularizers.l2(reg))(concat_model_QA)    
    Final_model = Model(inputs=[input_Q, input_A], outputs=output)

    # Compiling the model 
    Final_model.compile(optimizer=opp, loss='categorical_crossentropy', metrics=['acc'])

    return Final_model


#path = "SISC/Friends/"
path = ""


# Defining the model parameters
k_size = [2, 3, 4, 5, 6, 7, 8, 9, 10]
num_filters = 100
batch_s = 25
epochs = 21
num_conv = [3,6]


# Training the model: 
Final_model = CNN(num_filters, k_size, batch_s, num_conv, reg=0.5, opp='adam', act='relu', dropoutrate=0.5)
history = Final_model.fit([X_train_Q, X_train_A], Y_train, batch_size=batch_s, epochs=epochs, verbose=0)
