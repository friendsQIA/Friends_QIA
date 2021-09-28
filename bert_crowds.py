import crowds_implementation as cr
import pandas as pd
import preprocessing_functions as prep
import preprocessing_functions_for_final_runs as preproc_final
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Activation, Dropout, Dense
from tensorflow.keras.layers import MaxPooling1D, concatenate, Flatten
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras import regularizers

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

friends_train = pd.read_csv("Data/Final_QA_datasets/qa_data_train.csv", delimiter="\t", index_col=0)
friends_tune = pd.read_csv("Data/Final_QA_datasets/qa_data_tune.csv", delimiter="\t", index_col=0)
friends_dev = pd.read_csv("Data/Final_QA_datasets/qa_data_dev.csv", delimiter="\t", index_col=0)
friends_test = pd.read_csv("Data/Final_QA_datasets/qa_data_test.csv", delimiter="\t", index_col=0)

# Excluding the labels (needed for multiple annotations)
friends_train = prep.exclude_labels(friends_train, [5,6], 'Goldstandard')
friends_tune = prep.exclude_labels(friends_tune, [5,6], 'Goldstandard')
friends_dev = prep.exclude_labels(friends_dev, [5,6], 'Goldstandard')
        
X_train, Y_train, X_dev, Y_dev, Y_dev_original, X_test, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels = preproc_final.preprocessing(friends_train, friends_tune, friends_dev, friends_test, exclude=[5,6], multi_input=False, input_def='c')

friends_train = pd.concat([friends_train, friends_tune])

friends_train_ann1 = np.delete(np.array([i-1 for i in friends_train['Annotation_1']]), 4678, axis=0)
friends_train_ann2 = np.delete(np.array([i-1 for i in friends_train['Annotation_2']]), 4678, axis=0)
friends_train_ann3 = np.delete(np.array([i-1 for i in friends_train['Annotation_3']]), 4678, axis=0)

friends_train_ann1_onehot = prep.one_hot_encoder_with_missing_labels(friends_train_ann1, number_of_labels)
friends_train_ann2_onehot = prep.one_hot_encoder_with_missing_labels(friends_train_ann2, number_of_labels)
friends_train_ann3_onehot = prep.one_hot_encoder_with_missing_labels(friends_train_ann3, number_of_labels)

friends_anns_train = list([friends_train_ann1_onehot, friends_train_ann2_onehot, friends_train_ann3_onehot])
friends_num_ann = len(friends_anns_train)
friends_anns_train = np.asarray(friends_anns_train)
friends_anns_train = np.transpose(friends_anns_train,(1,2,0))
friends_data_size = len(friends_train)

# Obtaining the embedding matrix
data = np.load("friends_QA_bert_embs_train.npy", mmap_mode="r")
data_dev = np.load("friends_QA_bert_embs_dev.npy", mmap_mode="r")
data_test = np.load("friends_QA_bert_embs_test.npy", mmap_mode="r")

num_filters = 128
k_size = [2, 3, 4, 5, 6, 7, 8, 9, 10]
batch_s = 50
num_conv = 9
maxlen = 200
number_of_labels = 4
epochs = 67

def load_model(file_json, file_h5):
    json_file = open(file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(file_h5)

    return model

base_model = load_model("base_model/model_1_QA.json","base_model/model_1_QA.h5")

model_crowds = cr.CrowdsClassification(number_of_labels, friends_num_ann)(base_model.output)
model_crowds = Model(inputs=base_model.input, outputs=model_crowds)
model_crowds.compile(optimizer='adam', loss=loss)

model_crowds.fit(data, friends_anns_train, batch_size=bs, epochs=epochs, verbose=1)

# Save the model's weights
model_crowds_weights = model_crowds.layers[-1].get_weights()
# Remove the crowd layer to make predictions
model_crowds = Model(inputs=model_crowds.input, outputs=model_crowds.layers[-2].output)       

