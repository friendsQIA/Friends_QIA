# General imports
import pandas as pd

# Defined functions
import data_preparation.preprocessing_functions as preproc
import data_preparation.preprocessing_functions_for_final_runs as preproc_final

# For the CNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dropout, Dense
from tensorflow.keras.layers import MaxPooling1D, concatenate, Flatten
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras import regularizers


# Getting the data files

# Circa
# qa_data_train = pd.read_csv('Data/Circa_data/circa_data_train.csv', sep='\t', index_col=0)
# qa_data_tune = pd.read_csv('Data/Circa_data/circa_data_tune.csv', sep='\t', index_col=0)
# qa_data_dev = pd.read_csv('Data/Circa_data/circa_data_dev.csv', sep='\t', index_col=0)
# qa_data_test = pd.read_csv('Data/Circa_data/circa_data_test.csv', sep='\t', index_col=0)

# Friends
qa_data_train = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_train.csv', sep='\t', index_col=0)
qa_data_tune = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_tune.csv', sep='\t', index_col=0)
qa_data_dev = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_dev.csv', sep='\t', index_col=0)
qa_data_test = pd.read_csv('Data/Friends_data/Final_QA_datasets/qa_data_test.csv', sep='\t', index_col=0)


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


## Multi input - Single Channel CNN function
def CNN(num_filters, k_size, batch_s, num_conv, reg=0.5, opp='adam', act='relu', dropoutrate=0.5):
    # Model for Answers 
    # Defining the Embedding layer and the input 
    input_Q = Input(shape=(maxlen,))
    embedding_layer_Q = Embedding(vocab_size, emb_size, weights=[glove_embedding_matrix], input_length=maxlen, trainable=True, mask_zero=True)(input_Q)
    
    # Obtaining the concatenated convolutional layers 
    concat_model_Q = convolutions(embedding_layer_Q, maxlen, act, num_filters, k_size, dropoutrate, num_conv[0])

    # Model for Answers 
    # Defining the Embedding layer and the input 
    input_A = Input(shape=(maxlen,))
    embedding_layer_A = Embedding(vocab_size, emb_size, weights=[glove_embedding_matrix2], input_length=maxlen, trainable=True, mask_zero=True)(input_A)

    # Concatenating the concatenated convolutional layers
    concat_model_A = convolutions(embedding_layer_A, maxlen, act, num_filters, k_size, dropoutrate, num_conv[1])

    # Concatenating Q and A 
    concat_model_QA = concatenate([concat_model_Q, concat_model_A], axis=-1) 

    # Making the final model 
    output = Dense(number_of_labels, activation='softmax', kernel_regularizer=regularizers.l2(reg))(concat_model_QA)    
    Final_model = Model(inputs=[input_Q, input_A], outputs=output)

    # Compiling the model 
    Final_model.compile(optimizer=opp, loss='categorical_crossentropy', metrics=['acc'])

    return Final_model


# Multi input - Circa
# X_train_Q, X_train_A, Y_train, X_dev_Q, X_dev_A, Y_dev, Y_dev_original, X_test_Q, X_test_A, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels = preproc_final.preprocessing_circa(qa_data_train, qa_data_tune, qa_data_dev, qa_data_test, exclude=[5,6], multi_input=True)

# Multi input - Friends
X_train_Q, X_train_A, Y_train, X_dev_Q, X_dev_A, Y_dev, Y_dev_original, X_test_Q, X_test_A, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels = preproc_final.preprocessing(qa_data_train, qa_data_tune, qa_data_dev, qa_data_test, exclude=[5,6], multi_input=True)


# Obtaining the embedding matrix
emb_file = 'glove.6B.100d.txt'   ##REMEMBER to change this path
emb_size = 100
glove_embedding_matrix = preproc.get_embeddings(vocab, emb_file, tokenizer, emb_size)
glove_embedding_matrix2 = glove_embedding_matrix   #Copy for the multi input


# Defining the model parameters
num_filters = 50
k_size = [2, 4, 6, 8, 10, 12, 14, 16, 18]
batch_s = 25
epochs = 5
num_conv = [9,9]


# # Training the model: 
Final_model = CNN(num_filters, k_size, batch_s, num_conv, reg=0.5, opp='adam', act='relu', dropoutrate=0.5)
history = Final_model.fit([X_train_Q, X_train_A], Y_train, batch_size=batch_s, epochs=epochs, verbose=1)
