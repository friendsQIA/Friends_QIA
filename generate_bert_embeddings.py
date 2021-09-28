# General imports
import pandas as pd
import numpy as np

import tensorflow as tf
from npy_append_array import NpyAppendArray
from transformers import BertTokenizer, TFBertModel

# Own functions
from data_preparation import preprocessing_functions as preproc

def get_data_friends(input_def = 'c'):
    labeled_data_train = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_train.csv', sep='\t', index_col=0)
    labeled_data_tune = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_tune.csv', sep='\t', index_col=0)
    labeled_data_dev = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_dev.csv', sep='\t', index_col=0)
    labeled_data_test = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_test.csv', sep='\t', index_col=0)

    train_sentences = preproc.exclude_labels(labeled_data_train,[5,6],"Goldstandard")
    tune_sentences = preproc.exclude_labels(labeled_data_tune,[5,6],"Goldstandard")
    dev_sentences = preproc.exclude_labels(labeled_data_dev,[5,6],"Goldstandard")
    test_sentences = preproc.exclude_labels(labeled_data_test,[5,6],"Goldstandard")

    #Concatenating train and tune to use for training
    train_sentences = pd.concat([train_sentences, tune_sentences]).reset_index()
    # Removing instance number 4678
    train_sentences = train_sentences.drop(4678)
    train_sentences = train_sentences.reset_index()
    
    y_train = np.array([int(i) for i in train_sentences["Goldstandard"]])
    y_dev = np.array([int(i) for i in dev_sentences["Goldstandard"]]) 
    y_test = np.array([int(i) for i in test_sentences["Goldstandard"]])

    if input_def == 'c':
        train_sentences = train_sentences["Q_modified"]+ " _ " + train_sentences["A_modified"]
        dev_sentences = dev_sentences["Q_modified"]+ " _ " + dev_sentences["A_modified"]
        test_sentences = test_sentences["Q_modified"]+ " _ " + test_sentences["A_modified"]
    
    elif input_def == 'a':
        train_sentences = train_sentences["A_modified"]
        dev_sentences = dev_sentences["A_modified"]
        test_sentences = test_sentences["A_modified"]

    elif input_def == 'q':
        train_sentences = train_sentences["Q_modified"]
        dev_sentences = dev_sentences["Q_modified"]
        test_sentences = test_sentences["Q_modified"]

    return train_sentences, y_train, dev_sentences, y_dev, test_sentences, y_test

### Function to save the Bert embeddings in a file
def save_bert_embs(dataset, filename, model):
    npaa = NpyAppendArray(filename)
    
    for qa in dataset:
        encoded_input = tokenizer(qa, return_tensors='tf')
        output = model(encoded_input)

        tmp = []

        for t in output[0][0]:
            tmp.append(t)

        #Adding padding for maxlen 200
        while len(tmp) != 200:
            tmp.append(np.zeros(768))
    
        tmp = np.asarray(tmp)

        #Saving in a file
        i = tmp.reshape(1,200,768)
        npaa.append(i)

# Extracting the BERT embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained("bert-base-cased")
tz = BertTokenizer.from_pretrained("bert-base-cased")

###Saving the embeddings for FRIENDS

## FRIENDS: Q+A
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = get_data_friends('c')

save_bert_embs(X_train, 'Bert_embs/friends_QA_bert_embs_train.npy', model)
save_bert_embs(X_dev, 'Bert_embs/friends_QA_bert_embs_dev.npy', model)
save_bert_embs(X_test, 'Bert_embs/friends_QA_bert_embs_test.npy', model)

## FRIENDS: Q
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = get_data_friends('q')

save_bert_embs(X_train, 'Bert_embs/friends_Q_bert_embs_train.npy', model)
save_bert_embs(X_dev, 'Bert_embs/friends_Q_bert_embs_dev.npy', model)
save_bert_embs(X_test, 'Bert_embs/friends_Q_bert_embs_test.npy', model)

## FRIENDS: A
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = get_data_friends('a')

save_bert_embs(X_train, 'Bert_embs/friends_A_bert_embs_train.npy', model)
save_bert_embs(X_dev, 'Bert_embs/friends_A_bert_embs_dev.npy', model)
save_bert_embs(X_test, 'Bert_embs/friends_A_bert_embs_test.npy', model)
