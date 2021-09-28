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
    

#Obtaining the embedding matrix
def get_embeddings(vocab, emb_file, tokenizer, emb_size):
	#Computing the vocab size
	vocab_size = len(vocab) + 1

	#Creating dictionary with words as keys and their corresponding embeddings as values
	embeddings_dictionary = dict()
	embeddings_file = open(emb_file)#, encoding="utf8")

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


def preprocessing(train_data, tune_data, dev_data, test_data, exclude=[], multi_input=False, input_def='c'):

	#Input definitions
	# 'q' = only question
	# 'a' = only answer
	# 'c' = concatenated question and answer


	#Excluding the appropriate labels
	train_data = exclude_labels(train_data, exclude, 'Goldstandard')
	tune_data = exclude_labels(tune_data, exclude, 'Goldstandard')
	dev_data = exclude_labels(dev_data, exclude, 'Goldstandard')
	test_data = exclude_labels(test_data, exclude, 'Goldstandard')

	#Concatenating train and tune to use for training
	train_data = pd.concat([train_data, tune_data]).reset_index()

	#Getting the number of labels
	number_of_labels = len(set(train_data["Goldstandard"]))

	#Obtaining the labels
	Y_train = np.array([int(i) for i in train_data["Goldstandard"]])
	Y_dev = np.array([int(i) for i in dev_data["Goldstandard"]])
	Y_test = np.array([int(i) for i in test_data["Goldstandard"]])

	#Keeping the original format of the labels to compute predictions outside the CNN model
	Y_dev_original = Y_dev

	#Converting the y data to one-hot encoding to use inside the CNN model
	Y_train = one_hot_encoder(Y_train, number_of_labels)
	Y_dev = one_hot_encoder(Y_dev, number_of_labels)


	if multi_input == False:

		if input_def == 'q':
			#Using only the questions for single input
			X_train = list(train_data['Q_modified'])
			X_dev = list(dev_data['Q_modified'])
			X_test = list(test_data['Q_modified'])

		elif input_def == 'a':
			#Using only the answers for single input
			X_train = list(train_data['A_modified'])
			X_dev = list(dev_data['A_modified'])
			X_test = list(test_data['A_modified'])

		elif input_def == 'c':
			#Get the QA data, concatenate Q and A for single input
			X_train = list(train_data['Q_modified'] + " _ " + train_data['A_modified'])
			X_dev = list(dev_data['Q_modified'] + " _ " + dev_data['A_modified'])
			X_test = list(test_data['Q_modified'] + " _ " + test_data['A_modified'])

		#Tokenize the data
		X_train = tokenize(X_train)
		X_dev = tokenize(X_dev)
		X_test = tokenize(X_test)

		#Join the data again after tokenization
		X_train = join(X_train)
		X_dev = join(X_dev)
		X_test = join(X_test)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train)

		#Converging the words to integers
		X_train = tokenizer.texts_to_sequences(X_train)
		X_dev = tokenizer.texts_to_sequences(X_dev)
		X_test = tokenizer.texts_to_sequences(X_test)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		#Padding the sentences with zeroes 
		X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
		X_dev = pad_sequences(X_dev, padding='post', maxlen=maxlen)
		X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

		return X_train, Y_train, X_dev, Y_dev, Y_dev_original, X_test, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels


	if multi_input == True:

		#Making Question and Answer inputs 

		#Concatenate Q and A to be used for tokenization on the combined data
		X_train_QA = list(train_data['Q_modified'] + " _ " + train_data['A_modified'])

		X_train_Q = list(train_data['Q_modified'])
		X_dev_Q = list(dev_data['Q_modified'])
		X_test_Q = list(test_data['Q_modified'])

		X_train_A = list(train_data['A_modified'])
		X_dev_A = list(dev_data['A_modified'])
		X_test_A = list(test_data['A_modified'])

		#Tokenizing Question and answer input
		X_train_QA = tokenize(X_train_QA)

		X_train_Q = tokenize(X_train_Q)
		X_dev_Q = tokenize(X_dev_Q)
		X_test_Q = tokenize(X_test_Q)

		X_train_A = tokenize(X_train_A)
		X_dev_A = tokenize(X_dev_A)
		X_test_A = tokenize(X_test_A)

		#Joining the data again after tokenization
		X_train_QA = join(X_train_QA)

		X_train_Q = join(X_train_Q)
		X_dev_Q = join(X_dev_Q)
		X_test_Q = join(X_test_Q)

		X_train_A = join(X_train_A)
		X_dev_A = join(X_dev_A)	
		X_test_A = join(X_test_A)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train_QA)

		#Converging the words to integers
		X_train_Q = tokenizer.texts_to_sequences(X_train_Q)
		X_dev_Q = tokenizer.texts_to_sequences(X_dev_Q)
		X_test_Q = tokenizer.texts_to_sequences(X_test_Q)

		X_train_A = tokenizer.texts_to_sequences(X_train_A)
		X_dev_A = tokenizer.texts_to_sequences(X_dev_A)
		X_test_A = tokenizer.texts_to_sequences(X_test_A)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA on the Friends data

		#Padding the sentences with zeroes 
		X_train_Q = pad_sequences(X_train_Q, padding='post', maxlen=maxlen)
		X_dev_Q = pad_sequences(X_dev_Q, padding='post', maxlen=maxlen)
		X_test_Q = pad_sequences(X_test_Q, padding='post', maxlen=maxlen)

		X_train_A = pad_sequences(X_train_A, padding='post', maxlen=maxlen)
		X_dev_A = pad_sequences(X_dev_A, padding='post', maxlen=maxlen)
		X_test_A = pad_sequences(X_test_A, padding='post', maxlen=maxlen)

		return X_train_Q, X_train_A, Y_train, X_dev_Q, X_dev_A, Y_dev, Y_dev_original, X_test_Q, X_test_A, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels


def preprocessing_circa(train_data, tune_data, dev_data, test_data, exclude=[], multi_input=False, input_def='c'):
    
	#Excluding the appropriate labels
	train_data = exclude_labels(train_data, exclude, 'Label')
	tune_data = exclude_labels(tune_data, exclude, 'Label')
	dev_data = exclude_labels(dev_data, exclude, 'Label')
	test_data = exclude_labels(test_data, exclude, 'Label')

	#Concatenating train and tune to use for training
	train_data = pd.concat([train_data, tune_data]).reset_index()

	#Getting the number of labels
	number_of_labels = len(set(train_data["Label"]))

	#Obtaining the labels
	Y_train = np.array([int(i) for i in train_data["Label"]])
	Y_dev = np.array([int(i) for i in dev_data["Label"]])
	Y_test = np.array([int(i) for i in test_data["Label"]])

	#Keeping the original format of the labels to compute predictions outside the CNN model
	Y_dev_original = Y_dev

	#Converting the y data to one-hot encoding to use inside the CNN model
	Y_train = one_hot_encoder(Y_train, number_of_labels)
	Y_dev = one_hot_encoder(Y_dev, number_of_labels)


	if multi_input == False:

		if input_def == 'q':
			#Using only the questions for single input
			X_train = list(train_data['question-X'])
			X_dev = list(dev_data['question-X'])
			X_test = list(test_data['question-X'])

		elif input_def == 'a':
			#Using only the answers for single input
			X_train = list(train_data['answer-Y'])
			X_dev = list(dev_data['answer-Y'])
			X_test = list(test_data['answer-Y'])

		elif input_def == 'c':
			#Get the QA data, concatenate Q and A for single input
			X_train = list(train_data['question-X'] + " _ " + train_data['answer-Y'])
			X_dev = list(dev_data['question-X'] + " _ " + dev_data['answer-Y'])
			X_test = list(test_data['question-X'] + " _ " + test_data['answer-Y'])
		
		#Tokenize the data
		X_train = tokenize(X_train)
		X_dev = tokenize(X_dev)
		X_test = tokenize(X_test)

		#Join the data again after tokenization
		X_train = join(X_train)
		X_dev = join(X_dev)
		X_test = join(X_test)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train)

		#Converging the words to integers
		X_train = tokenizer.texts_to_sequences(X_train)
		X_dev = tokenizer.texts_to_sequences(X_dev)
		X_test = tokenizer.texts_to_sequences(X_test)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		#Padding the sentences with zeroes 
		X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
		X_dev = pad_sequences(X_dev, padding='post', maxlen=maxlen)
		X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

		return X_train, Y_train, X_dev, Y_dev, Y_dev_original, X_test, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels


	if multi_input == True:

		#Making Question and Answer inputs

		#Concatenate Q and A to be used for tokenization on the combined data
		X_train_QA = list(train_data['question-X'] + " _ " + train_data['answer-Y'])

		X_train_Q = list(train_data['question-X'])
		X_dev_Q = list(dev_data['question-X'])
		X_test_Q = list(test_data['question-X'])

		X_train_A = list(train_data['answer-Y'])
		X_dev_A = list(dev_data['answer-Y'])
		X_test_A = list(test_data['answer-Y'])

		#Tokenizing Question and answer input
		X_train_QA = tokenize(X_train_QA)

		X_train_Q = tokenize(X_train_Q)
		X_dev_Q = tokenize(X_dev_Q)
		X_test_Q = tokenize(X_test_Q)

		X_train_A = tokenize(X_train_A)
		X_dev_A = tokenize(X_dev_A)
		X_test_A = tokenize(X_test_A)

		#Joining the data again after tokenization
		X_train_QA = join(X_train_QA)

		X_train_Q = join(X_train_Q)
		X_dev_Q = join(X_dev_Q)
		X_test_Q = join(X_test_Q)

		X_train_A = join(X_train_A)
		X_dev_A = join(X_dev_A)	
		X_test_A = join(X_test_A)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train_QA)

		#Converging the words to integers
		X_train_Q = tokenizer.texts_to_sequences(X_train_Q)
		X_dev_Q = tokenizer.texts_to_sequences(X_dev_Q)
		X_test_Q = tokenizer.texts_to_sequences(X_test_Q)

		X_train_A = tokenizer.texts_to_sequences(X_train_A)
		X_dev_A = tokenizer.texts_to_sequences(X_dev_A)
		X_test_A = tokenizer.texts_to_sequences(X_test_A)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		#Padding the sentences with zeroes 
		X_train_Q = pad_sequences(X_train_Q, padding='post', maxlen=maxlen)
		X_dev_Q = pad_sequences(X_dev_Q, padding='post', maxlen=maxlen)
		X_test_Q = pad_sequences(X_test_Q, padding='post', maxlen=maxlen)

		X_train_A = pad_sequences(X_train_A, padding='post', maxlen=maxlen)
		X_dev_A = pad_sequences(X_dev_A, padding='post', maxlen=maxlen)
		X_test_A = pad_sequences(X_test_A, padding='post', maxlen=maxlen)

		return X_train_Q, X_train_A, Y_train, X_dev_Q, X_dev_A, Y_dev, Y_dev_original, X_test_Q, X_test_A, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels


def preprocessing_cross_domain_separate_data(train_circa, tune_circa, dev_friends, test_friends, exclude=[], multi_input=False, input_def='c'):
    
	#Excluding the appropriate labels
	train_circa = exclude_labels(train_circa, exclude, 'Label')
	tune_circa = exclude_labels(tune_circa, exclude, 'Label')
	dev_data = exclude_labels(dev_friends, exclude, 'Goldstandard')
	test_data = exclude_labels(test_friends, exclude, 'Goldstandard')

	#Concatenating train and tune to use for training
	train_data = pd.concat([train_circa, tune_circa]).reset_index()

	#Getting the number of labels
	number_of_labels = len(set(train_data["Label"]))

	#Obtaining the labels
	Y_train = np.array([int(i) for i in train_data["Label"]])
	Y_dev = np.array([int(i) for i in dev_data["Goldstandard"]])
	Y_test = np.array([int(i) for i in test_data["Goldstandard"]])

	if multi_input == False:

		if input_def == 'q':
			#Using only the questions for single input
			X_train = list(train_data['question-X'])
			X_dev = list(dev_data['Q_modified'])
			X_test = list(test_data['Q_modified'])

		elif input_def == 'a':
			#Using only the answers for single input
			X_train = list(train_data['answer-Y'])
			X_dev = list(dev_data['A_modified'])
			X_test = list(test_data['A_modified'])

		elif input_def == 'c':
			#Get the QA data, concatenate Q and A for single input
			X_train = list(train_data['question-X'] + " _ " + train_data['answer-Y'])
			X_dev = list(dev_data['Q_modified'] + " _ " + dev_data['A_modified'])
			X_test = list(test_data['Q_modified'] + " _ " + test_data['A_modified'])
		
		#Tokenize the data
		X_train = tokenize(X_train)
		X_dev = tokenize(X_dev)
		X_test = tokenize(X_test)

		#Join the data again after tokenization
		X_train = join(X_train)
		X_dev = join(X_dev)
		X_test = join(X_test)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train)

		#Converging the words to integers
		X_train = tokenizer.texts_to_sequences(X_train)
		X_dev = tokenizer.texts_to_sequences(X_dev)
		X_test = tokenizer.texts_to_sequences(X_test)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		#Padding the sentences with zeroes 
		X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
		X_dev = pad_sequences(X_dev, padding='post', maxlen=maxlen)
		X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

		return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels


	if multi_input == True:

		#Making Question and Answer inputs 

		#Concatenate Q and A to be used for tokenization on the combined data
		X_train_QA = list(train_data['question-X'] + " _ " + train_data['answer-Y'])

		X_train_Q = list(train_data['question-X'])
		X_dev_Q = list(dev_data['Q_modified'])
		X_test_Q = list(test_data['Q_modified'])

		X_train_A = list(train_data['answer-Y'])
		X_dev_A = list(dev_data['A_modified'])
		X_test_A = list(test_data['A_modified'])

		#Tokenizing Question and answer input
		X_train_QA = tokenize(X_train_QA)

		X_train_Q = tokenize(X_train_Q)
		X_dev_Q = tokenize(X_dev_Q)
		X_test_Q = tokenize(X_test_Q)

		X_train_A = tokenize(X_train_A)
		X_dev_A = tokenize(X_dev_A)
		X_test_A = tokenize(X_test_A)

		#Joining the data again after tokenization
		X_train_QA = join(X_train_QA)

		X_train_Q = join(X_train_Q)
		X_dev_Q = join(X_dev_Q)
		X_test_Q = join(X_test_Q)

		X_train_A = join(X_train_A)
		X_dev_A = join(X_dev_A)	
		X_test_A = join(X_test_A)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train_QA)

		#Converging the words to integers
		X_train_Q = tokenizer.texts_to_sequences(X_train_Q)
		X_dev_Q = tokenizer.texts_to_sequences(X_dev_Q)
		X_test_Q = tokenizer.texts_to_sequences(X_test_Q)

		X_train_A = tokenizer.texts_to_sequences(X_train_A)
		X_dev_A = tokenizer.texts_to_sequences(X_dev_A)
		X_test_A = tokenizer.texts_to_sequences(X_test_A)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		#Padding the sentences with zeroes 
		X_train_Q = pad_sequences(X_train_Q, padding='post', maxlen=maxlen)
		X_dev_Q = pad_sequences(X_dev_Q, padding='post', maxlen=maxlen)
		X_test_Q = pad_sequences(X_test_Q, padding='post', maxlen=maxlen)

		X_train_A = pad_sequences(X_train_A, padding='post', maxlen=maxlen)
		X_dev_A = pad_sequences(X_dev_A, padding='post', maxlen=maxlen)
		X_test_A = pad_sequences(X_test_A, padding='post', maxlen=maxlen)

		return X_train_Q, X_train_A, Y_train, X_dev_Q, X_dev_A, Y_dev, X_test_Q, X_test_A, Y_test, vocab, vocab_size, maxlen, tokenizer, number_of_labels


def preprocessing_cross_domain_concatenated_data(train_circa, tune_circa, dev_circa, test_circa, train_friends, tune_friends, dev_friends, test_friends, exclude=[], multi_input=False, input_def='c'):

	#Excluding the appropriate labels    
	train_circa = exclude_labels(train_circa, exclude, 'Label')
	tune_circa = exclude_labels(tune_circa, exclude, 'Label')
	dev_circa = exclude_labels(dev_circa, exclude, 'Label')
	test_circa = exclude_labels(test_circa, exclude, 'Label')

	train_friends = exclude_labels(train_friends, exclude, 'Goldstandard')
	tune_friends = exclude_labels(tune_friends, exclude, 'Goldstandard')
	dev_friends = exclude_labels(dev_friends, exclude, 'Goldstandard')
	test_friends = exclude_labels(test_friends, exclude, 'Goldstandard')

	#Droppinig the irrelevant columns
	c = ['context', 'canquestion-X', 'judgements', 'goldstandard1', 'goldstandard2', 'Annotation_1', 'Annotation_2', 'Annotation_3', 'Annotation_4', 'Annotation_5']
	train_circa.drop(c, inplace=True, axis=1)
	tune_circa.drop(c, inplace=True, axis=1)
	dev_circa.drop(c, inplace=True, axis=1)
	test_circa.drop(c, inplace=True, axis=1)

	c = ['Season', 'Episode', 'Category', 'Q_person', 'A_person', 'Q_original', 'A_original', 'Annotation_1', 'Annotation_2', 'Annotation_3']
	train_friends.drop(c, inplace=True, axis=1)
	tune_friends.drop(c, inplace=True, axis=1)
	dev_friends.drop(c, inplace=True, axis=1)
	test_friends.drop(c, inplace=True, axis=1)

	#Changing the column names
	train_circa = train_circa.rename(columns = {'question-X': 'Question', 'answer-Y': 'Answer'}, inplace = False)
	tune_circa = tune_circa.rename(columns = {'question-X': 'Question', 'answer-Y': 'Answer'}, inplace = False)
	dev_circa = dev_circa.rename(columns = {'question-X': 'Question', 'answer-Y': 'Answer'}, inplace = False)
	test_circa = test_circa.rename(columns = {'question-X': 'Question', 'answer-Y': 'Answer'}, inplace = False)

	train_friends = train_friends.rename(columns = {'Q_modified': 'Question', 'A_modified': 'Answer', 'Goldstandard': 'Label'}, inplace = False)
	tune_friends = tune_friends.rename(columns = {'Q_modified': 'Question', 'A_modified': 'Answer', 'Goldstandard': 'Label'}, inplace = False)
	dev_friends = dev_friends.rename(columns = {'Q_modified': 'Question', 'A_modified': 'Answer', 'Goldstandard': 'Label'}, inplace = False)
	test_friends = test_friends.rename(columns = {'Q_modified': 'Question', 'A_modified': 'Answer', 'Goldstandard': 'Label'}, inplace = False)

	#Concatenating train and tune to use for training (separately for Circa and Friends)
	train_circa = pd.concat([train_circa, tune_circa]).reset_index()
	train_circa.drop('index', inplace=True, axis=1)
	train_friends = pd.concat([train_friends, tune_friends]).reset_index()
	train_friends.drop('index', inplace=True, axis=1)

	#Concatenating the Friends and Circa data to use for training
	train_data = pd.concat([train_circa, train_friends]).reset_index()
	train_data.drop('index', inplace=True, axis=1)

	#Getting the number of labels
	number_of_labels = len(set(train_data["Label"]))

	#Obtaining the labels
	Y_train = np.array([int(i) for i in train_data["Label"]])
	Y_dev_circa = np.array([int(i) for i in dev_circa["Label"]])
	Y_test_circa = np.array([int(i) for i in test_circa["Label"]])
	Y_dev_friends = np.array([int(i) for i in dev_friends["Label"]])
	Y_test_friends = np.array([int(i) for i in test_friends["Label"]])

	#Converting the y data to one-hot encoding to use inside the CNN model
	Y_train = one_hot_encoder(Y_train, number_of_labels)

	if multi_input == False:

		if input_def == 'q':
			#Using only the questions for single input
			X_train = list(train_data['Question'])
			X_dev_circa = list(dev_circa['Question'])
			X_test_circa = list(test_circa['Question'])
			X_dev_friends = list(dev_friends['Question'])
			X_test_friends = list(test_friends['Question'])

		elif input_def == 'a':
			#Using only the answers for single input
			X_train = list(train_data['Answer'])
			X_dev_circa = list(dev_circa['Answer'])
			X_test_circa = list(test_circa['Answer'])
			X_dev_friends = list(dev_friends['Answer'])
			X_test_friends = list(test_friends['Answer'])

		elif input_def == 'c':
			#Get the QA data, concatenate Q and A for single input
			X_train = list(train_data['Question'] + " _ " + train_data['Answer'])
			X_dev_circa = list(dev_circa['Question'] + " _ " + dev_circa['Answer'])
			X_test_circa = list(test_circa['Question'] + " _ " + test_circa['Answer'])
			X_dev_friends = list(dev_friends['Question'] + " _ " + dev_friends['Answer'])
			X_test_friends = list(test_friends['Question'] + " _ " + test_friends['Answer'])

		#Tokenize the data
		X_train = tokenize(X_train)
		X_dev_circa = tokenize(X_dev_circa)
		X_test_circa = tokenize(X_test_circa)
		X_dev_friends = tokenize(X_dev_friends)
		X_test_friends = tokenize(X_test_friends)

		#Join the data again after tokenization
		X_train = join(X_train)
		X_dev_circa = join(X_dev_circa)
		X_test_circa = join(X_test_circa)
		X_dev_friends = join(X_dev_friends)
		X_test_friends = join(X_test_friends)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train)

		#Converging the words to integers
		X_train = tokenizer.texts_to_sequences(X_train)
		X_dev_circa = tokenizer.texts_to_sequences(X_dev_circa)
		X_test_circa = tokenizer.texts_to_sequences(X_test_circa)
		X_dev_friends = tokenizer.texts_to_sequences(X_dev_friends)
		X_test_friends = tokenizer.texts_to_sequences(X_test_friends)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		#Padding the sentences with zeroes 
		X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
		X_dev_circa = pad_sequences(X_dev_circa, padding='post', maxlen=maxlen)
		X_test_circa = pad_sequences(X_test_circa, padding='post', maxlen=maxlen)
		X_dev_friends = pad_sequences(X_dev_friends, padding='post', maxlen=maxlen)
		X_test_friends = pad_sequences(X_test_friends, padding='post', maxlen=maxlen)

		return X_train, Y_train, X_dev_circa, Y_dev_circa, X_dev_friends, Y_dev_friends, X_test_circa, Y_test_circa, X_test_friends, Y_test_friends, vocab, vocab_size, maxlen, tokenizer, number_of_labels


	if multi_input == True:

		#Making Question and Answer inputs 

		#Concatenate Q and A to be used for tokenization on the combined data
		X_train_QA = list(train_data['Question'] + " _ " + train_data['Answer'])

		X_train_Q = list(train_data['Question'])
		X_dev_Q_circa = list(dev_circa['Question'])
		X_test_Q_circa = list(test_circa['Question'])
		X_dev_Q_friends = list(dev_friends['Question'])
		X_test_Q_friends = list(test_friends['Question'])

		X_train_A = list(train_data['Answer'])
		X_dev_A_circa = list(dev_circa['Answer'])
		X_test_A_circa = list(test_circa['Answer'])
		X_dev_A_friends = list(dev_friends['Answer'])
		X_test_A_friends = list(test_friends['Answer'])

		#Tokenizing Question and answer input
		X_train_QA = tokenize(X_train_QA)

		X_train_Q = tokenize(X_train_Q)
		X_dev_Q_circa = tokenize(X_dev_Q_circa)
		X_test_Q_circa = tokenize(X_test_Q_circa)
		X_dev_Q_friends = tokenize(X_dev_Q_friends)
		X_test_Q_friends = tokenize(X_test_Q_friends)

		X_train_A = tokenize(X_train_A)
		X_dev_A_circa = tokenize(X_dev_A_circa)
		X_test_A_circa = tokenize(X_test_A_circa)
		X_dev_A_friends = tokenize(X_dev_A_friends)
		X_test_A_friends = tokenize(X_test_A_friends)

		#Joining the data again after tokenization
		X_train_QA = join(X_train_QA)

		X_train_Q = join(X_train_Q)
		X_dev_Q_circa = join(X_dev_Q_circa)
		X_test_Q_circa = join(X_test_Q_circa)
		X_dev_Q_friends = join(X_dev_Q_friends)
		X_test_Q_friends = join(X_test_Q_friends)

		X_train_A = join(X_train_A)
		X_dev_A_circa = join(X_dev_A_circa)
		X_test_A_circa = join(X_test_A_circa)
		X_dev_A_friends = join(X_dev_A_friends)
		X_test_A_friends = join(X_test_A_friends)

		#Using the keras tokenizer 
		tokenizer = Tokenizer(filters = " ", lower=False)
		tokenizer.fit_on_texts(X_train_QA)

		#Converging the words to integers
		X_train_Q = tokenizer.texts_to_sequences(X_train_Q)
		X_dev_Q_circa = tokenizer.texts_to_sequences(X_dev_Q_circa)
		X_test_Q_circa = tokenizer.texts_to_sequences(X_test_Q_circa)
		X_dev_Q_friends = tokenizer.texts_to_sequences(X_dev_Q_friends)
		X_test_Q_friends = tokenizer.texts_to_sequences(X_test_Q_friends)

		X_train_A = tokenizer.texts_to_sequences(X_train_A)
		X_dev_A_circa = tokenizer.texts_to_sequences(X_dev_A_circa)
		X_test_A_circa = tokenizer.texts_to_sequences(X_test_A_circa)
		X_dev_A_friends = tokenizer.texts_to_sequences(X_dev_A_friends)
		X_test_A_friends = tokenizer.texts_to_sequences(X_test_A_friends)

		#Creating the vocabulary using the tokenizer
		vocab = list(tokenizer.word_index.keys())

		#Padding the samples based on the max length of a QA
		vocab_size = len(vocab) + 1
		maxlen = 200   #number found earlier through EDA

		#Padding the sentences with zeroes 
		X_train_Q = pad_sequences(X_train_Q, padding='post', maxlen=maxlen)
		X_dev_Q_circa = pad_sequences(X_dev_Q_circa, padding='post', maxlen=maxlen)
		X_test_Q_circa = pad_sequences(X_test_Q_circa, padding='post', maxlen=maxlen)
		X_dev_Q_friends = pad_sequences(X_dev_Q_friends, padding='post', maxlen=maxlen)
		X_test_Q_friends = pad_sequences(X_test_Q_friends, padding='post', maxlen=maxlen)

		X_train_A = pad_sequences(X_train_A, padding='post', maxlen=maxlen)
		X_dev_A_circa = pad_sequences(X_dev_A_circa, padding='post', maxlen=maxlen)
		X_test_A_circa = pad_sequences(X_test_A_circa, padding='post', maxlen=maxlen)
		X_dev_A_friends = pad_sequences(X_dev_A_friends, padding='post', maxlen=maxlen)
		X_test_A_friends = pad_sequences(X_test_A_friends, padding='post', maxlen=maxlen)

		return X_train_Q, X_train_A, Y_train, X_dev_Q_circa, X_dev_A_circa, Y_dev_circa, X_dev_Q_friends, X_dev_A_friends, Y_dev_friends, X_test_Q_circa, X_test_A_circa, Y_test_circa, X_test_Q_friends, X_test_A_friends, Y_test_friends, vocab, vocab_size, maxlen, tokenizer, number_of_labels
