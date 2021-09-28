from collections import Counter
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import re

#For data preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Apply tokenization to the data
def tokenize(data):
	d = []
	for i in data:
		i = i.lower()
		token = re.compile("[\w]+(?=n['’‘]t)|n['’‘]t\\b|['’‘]re\\b|['’‘]d\\b|['’‘]m\\b|['’‘]ll\\b|['’‘]s\\b|['’‘]ve\\b|[\w]+|[.?!;_,\-()—:'’‘]")
		text = token.findall(i)
		d.append(text)

	return d

#Joining each sentence to one string so we can use nltk's tokenizer
def join(data):
    l = []
    for senc in data:
        temp = []
        for word in senc:
            temp.append(word)
        temp = (" ".join(temp))
        temp = temp.replace("  ", " ")
        l.append(temp)
    
    return l


#Excludig labels 
def exclude_labels(data, list_to_exclude, column):
    for i in list_to_exclude:
        data = data[data[column] != i]
    
    # finding the kept labels and making the new labels that run from 0 to len(kept_labels)
    kept_labels = sorted(list(set(data[column])))
    new_labels = [i for i in range(len(kept_labels))]
    
    # changing the labels to the new labels, using index
    for data_idx, original_label in data.iterrows():
        for kept_label_idx, kept_label in enumerate(kept_labels):
            if kept_label == original_label[column]:
                data[column][data_idx] = new_labels[kept_label_idx]

    return data


#Making one-hot encoding
def one_hot_encoder(Y_data, number_of_labels):
    Y_encoded = np.zeros((len(Y_data), number_of_labels))
    for idx, i in enumerate(Y_data):
        Y_encoded[idx][i] = 1
    return Y_encoded

# Function for the crowds study
def one_hot_encoder_with_missing_labels(Y_data, number_of_labels):
    Y_encoded = np.zeros((len(Y_data), number_of_labels))
    for idx, i in enumerate(Y_data):
        if i > number_of_labels-1:
            Y_encoded[idx][:] = -1
        else:
            Y_encoded[idx][i] = 1
    return Y_encoded

#Obtaining the embedding matrix
def get_embeddings(vocab, emb_file, tokenizer, emb_size):
	#Computing the vocab size
	vocab_size = len(vocab) + 1

	#Creating dictionary with words as keys and their corresponding embeddings as values
	embeddings_dictionary = dict()
	embeddings_file = open(emb_file, encoding="utf8")

	for line in embeddings_file:
	    records = line.split()
	    word = records[0]
	    vector_dimensions = np.asarray(records[1:], dtype='float32')
	    embeddings_dictionary[word] = vector_dimensions
	embeddings_file.close()

	glove_embedding_matrix = np.zeros((vocab_size, emb_size))
	for word, index in tokenizer.word_index.items():
	    embedding_vector = embeddings_dictionary.get(word)
	    if word in vocab and embedding_vector is not None:
	    	glove_embedding_matrix[index] = embedding_vector

	return glove_embedding_matrix


def preprocessing(train_data, tune_data, dev_data, exclude=[], multi_input=False, NB=False, input_def='c'):

	#THIS FUNCTION DOES NOT HANDLE THE TEST SET

	#Input definitions
	# 'q' = only question
	# 'a' = only answer
	# 'c' = concatenated question and answer


	#Excluding the appropriate labels
	train_data = exclude_labels(train_data, exclude, 'Goldstandard')
	tune_data = exclude_labels(tune_data, exclude, 'Goldstandard')
	dev_data = exclude_labels(dev_data, exclude, 'Goldstandard')

	#Getting the number of labels
	number_of_labels = len(set(train_data["Goldstandard"]))

	#Obtaining the labels
	Y_train = np.array([int(i) for i in train_data["Goldstandard"]])
	Y_tune = np.array([int(i) for i in tune_data["Goldstandard"]])
	Y_dev = np.array([int(i) for i in dev_data["Goldstandard"]])

	#Keeping the original format of the labels to compute predictions outside the CNN model
	Y_tune_original = Y_tune
	Y_dev_original = Y_dev

	#Converting the y data to one-hot encoding to use inside the CNN model
	Y_train = one_hot_encoder(Y_train, number_of_labels)
	Y_tune = one_hot_encoder(Y_tune, number_of_labels)


	if multi_input == False:

		if input_def == 'q':
			#Using only the questions for single input
			X_train = list(train_data['Q_modified'])
			X_tune = list(tune_data['Q_modified'])
			X_dev = list(dev_data['Q_modified'])

		elif input_def == 'a':
			#Using only the answers for single input
			X_train = list(train_data['A_modified'])
			X_tune = list(tune_data['A_modified'])
			X_dev = list(dev_data['A_modified'])

		elif input_def == 'c':
			#Get the QA data, concatenate Q and A for single input
			X_train = list(train_data['Q_modified'] + " _ " + train_data['A_modified'])
			X_tune = list(tune_data['Q_modified'] + " _ " + tune_data['A_modified'])
			X_dev = list(dev_data['Q_modified'] + " _ " + dev_data['A_modified'])

		#Tokenize the data
		X_train = tokenize(X_train)
		X_tune = tokenize(X_tune)
		X_dev = tokenize(X_dev)

		#Join the data again after tokenization
		X_train = join(X_train)
		X_tune = join(X_tune)
		X_dev = join(X_dev)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train)

		#Converging the words to integers
		X_train = tokenizer.texts_to_sequences(X_train)
		X_tune = tokenizer.texts_to_sequences(X_tune)
		X_dev = tokenizer.texts_to_sequences(X_dev)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		if NB == False:
			#Padding the sentences with zeroes (this should not be done for Naive Bayes)
			X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
			X_tune = pad_sequences(X_tune, padding='post', maxlen=maxlen)
			X_dev = pad_sequences(X_dev, padding='post', maxlen=maxlen)

		return X_train, Y_train, X_tune, Y_tune, Y_tune_original, X_dev, Y_dev, Y_dev_original, vocab, vocab_size, maxlen, tokenizer, number_of_labels


	if multi_input == True:

		#Making Question and Answer inputs 

		#Concatenate Q and A to be used for tokenization on the combined data
		X_train_QA = list(train_data['Q_modified'] + " _ " + train_data['A_modified'])

		X_train_Q = list(train_data['Q_modified'])
		X_tune_Q = list(tune_data['Q_modified'])
		X_dev_Q = list(dev_data['Q_modified'])

		X_train_A = list(train_data['A_modified'])
		X_tune_A = list(tune_data['A_modified'])
		X_dev_A = list(dev_data['A_modified'])

		#Tokenizing Question and answer input
		X_train_QA = tokenize(X_train_QA)

		X_train_Q = tokenize(X_train_Q)
		X_tune_Q = tokenize(X_tune_Q)
		X_dev_Q = tokenize(X_dev_Q)

		X_train_A = tokenize(X_train_A)
		X_tune_A = tokenize(X_tune_A)
		X_dev_A = tokenize(X_dev_A)

		#Joining the data again after tokenization
		X_train_QA = join(X_train_QA)

		X_train_Q = join(X_train_Q)
		X_tune_Q = join(X_tune_Q)
		X_dev_Q = join(X_dev_Q)

		X_train_A = join(X_train_A)
		X_tune_A = join(X_tune_A)
		X_dev_A = join(X_dev_A)	

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train_QA)

		#Converging the words to integers
		X_train_Q = tokenizer.texts_to_sequences(X_train_Q)
		X_tune_Q = tokenizer.texts_to_sequences(X_tune_Q)
		X_dev_Q = tokenizer.texts_to_sequences(X_dev_Q)

		X_train_A = tokenizer.texts_to_sequences(X_train_A)
		X_tune_A = tokenizer.texts_to_sequences(X_tune_A)
		X_dev_A = tokenizer.texts_to_sequences(X_dev_A)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA on the Friends data

		#Padding the sentences with zeroes 
		X_train_Q = pad_sequences(X_train_Q, padding='post', maxlen=maxlen)
		X_tune_Q = pad_sequences(X_tune_Q, padding='post', maxlen=maxlen)
		X_dev_Q = pad_sequences(X_dev_Q, padding='post', maxlen=maxlen)

		X_train_A = pad_sequences(X_train_A, padding='post', maxlen=maxlen)
		X_tune_A = pad_sequences(X_tune_A, padding='post', maxlen=maxlen)
		X_dev_A = pad_sequences(X_dev_A, padding='post', maxlen=maxlen)

		return X_train_Q, X_train_A, Y_train, X_tune_Q, X_tune_A, Y_tune, Y_tune_original, X_dev_Q, X_dev_A, Y_dev, Y_dev_original, vocab, vocab_size, maxlen, tokenizer, number_of_labels


def preprocessing_circa(train_data, tune_data, dev_data, exclude=[], multi_input=False, NB=False, input_def='c'):
    
    #THIS FUNCTION DOES NOT HANDLE THE TEST SET

	#Input definitions
	# 'q' = only question
	# 'a' = only answer
	# 'c' = concatenated question and answer


	#Excluding the appropriate labels
	train_data = exclude_labels(train_data, exclude, 'Label')
	tune_data = exclude_labels(tune_data, exclude, 'Label')
	dev_data = exclude_labels(dev_data, exclude, 'Label')

	#Getting the number of labels
	number_of_labels = len(set(train_data["Label"]))

	#Obtaining the labels
	Y_train = np.array([int(i) for i in train_data["Label"]])
	Y_tune = np.array([int(i) for i in tune_data["Label"]])
	Y_dev = np.array([int(i) for i in dev_data["Label"]])

	#Keeping the original format of the labels to compute predictions outside the CNN model
	Y_tune_original = Y_tune
	Y_dev_original = Y_dev

	#Converting the y data to one-hot encoding to use inside the CNN model
	Y_train = one_hot_encoder(Y_train, number_of_labels)
	Y_tune = one_hot_encoder(Y_tune, number_of_labels)


	if multi_input == False:

		if input_def == 'q':
			#Using only the questions for single input
			X_train = list(train_data['question-X'])
			X_tune = list(tune_data['question-X'])
			X_dev = list(dev_data['question-X'])

		elif input_def == 'a':
			#Using only the answers for single input
			X_train = list(train_data['answer-Y'])
			X_tune = list(tune_data['answer-Y'])
			X_dev = list(dev_data['answer-Y'])

		elif input_def == 'c':
			#Get the QA data, concatenate Q and A for single input
			X_train = list(train_data['question-X'] + " _ " + train_data['answer-Y'])
			X_tune = list(tune_data['question-X'] + " _ " + tune_data['answer-Y'])
			X_dev = list(dev_data['question-X'] + " _ " + dev_data['answer-Y'])

		#Tokenize the data
		X_train = tokenize(X_train)
		X_tune = tokenize(X_tune)
		X_dev = tokenize(X_dev)

		#Join the data again after tokenization
		X_train = join(X_train)
		X_tune = join(X_tune)
		X_dev = join(X_dev)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train)

		#Converging the words to integers
		X_train = tokenizer.texts_to_sequences(X_train)
		X_tune = tokenizer.texts_to_sequences(X_tune)
		X_dev = tokenizer.texts_to_sequences(X_dev)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		if NB == False:
			#Padding the sentences with zeroes (this should not be done for Naive Bayes)
			X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
			X_tune = pad_sequences(X_tune, padding='post', maxlen=maxlen)
			X_dev = pad_sequences(X_dev, padding='post', maxlen=maxlen)

		return X_train, Y_train, X_tune, Y_tune, Y_tune_original, X_dev, Y_dev, Y_dev_original, vocab, vocab_size, maxlen, tokenizer, number_of_labels


	if multi_input == True:

		#Making Question and Answer inputs

		#Concatenate Q and A to be used for tokenization on the combined data
		X_train_QA = list(train_data['question-X'] + " _ " + train_data['answer-Y'])

		X_train_Q = list(train_data['question-X'])
		X_tune_Q = list(tune_data['question-X'])
		X_dev_Q = list(dev_data['question-X'])

		X_train_A = list(train_data['answer-Y'])
		X_tune_A = list(tune_data['answer-Y'])
		X_dev_A = list(dev_data['answer-Y'])

		#Tokenizing Question and answer input
		X_train_QA = tokenize(X_train_QA)

		X_train_Q = tokenize(X_train_Q)
		X_tune_Q = tokenize(X_tune_Q)
		X_dev_Q = tokenize(X_dev_Q)

		X_train_A = tokenize(X_train_A)
		X_tune_A = tokenize(X_tune_A)
		X_dev_A = tokenize(X_dev_A)

		#Joining the data again after tokenization
		X_train_QA = join(X_train_QA)

		X_train_Q = join(X_train_Q)
		X_tune_Q = join(X_tune_Q)
		X_dev_Q = join(X_dev_Q)

		X_train_A = join(X_train_A)
		X_tune_A = join(X_tune_A)
		X_dev_A = join(X_dev_A)	

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train_QA)

		#Converging the words to integers
		X_train_Q = tokenizer.texts_to_sequences(X_train_Q)
		X_tune_Q = tokenizer.texts_to_sequences(X_tune_Q)
		X_dev_Q = tokenizer.texts_to_sequences(X_dev_Q)

		X_train_A = tokenizer.texts_to_sequences(X_train_A)
		X_tune_A = tokenizer.texts_to_sequences(X_tune_A)
		X_dev_A = tokenizer.texts_to_sequences(X_dev_A)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA on the Friends data

		#Padding the sentences with zeroes 
		X_train_Q = pad_sequences(X_train_Q, padding='post', maxlen=maxlen)
		X_tune_Q = pad_sequences(X_tune_Q, padding='post', maxlen=maxlen)
		X_dev_Q = pad_sequences(X_dev_Q, padding='post', maxlen=maxlen)

		X_train_A = pad_sequences(X_train_A, padding='post', maxlen=maxlen)
		X_tune_A = pad_sequences(X_tune_A, padding='post', maxlen=maxlen)
		X_dev_A = pad_sequences(X_dev_A, padding='post', maxlen=maxlen)

		return X_train_Q, X_train_A, Y_train, X_tune_Q, X_tune_A, Y_tune, Y_tune_original, X_dev_Q, X_dev_A, Y_dev, Y_dev_original, vocab, vocab_size, maxlen, tokenizer, number_of_labels


