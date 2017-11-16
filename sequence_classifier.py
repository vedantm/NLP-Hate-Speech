# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 02:23:06 2017

@author: madha
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:03:25 2017

@author: madha
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from project_utils import Classifier

class SequenceClassifier(Classifier):
    def __init__(self,train_X, train_Y, test_X, test_Y, no_of_epochs,learning_rate,no_of_feats,sentence_length,embedding_dim):
        super(SequenceClassifier,self).__init__(train_X, train_Y, test_X, test_Y,no_of_epochs,learning_rate,no_of_feats)
        self.sentence_length = sentence_length
        self._tokenize()
        self.train_X = sequence.pad_sequences(self.train_X, maxlen=self.sentence_length)
        self.test_X = sequence.pad_sequences(self.test_X, maxlen=self.sentence_length)
        self.classifier = Sequential()
        self.classifier.add(Embedding(self.no_of_feats, embedding_dim, input_length=self.sentence_length))

    def _tokenize(self):
        # create the tokenizer
        t = Tokenizer(num_words=self.no_of_feats)
        # fit the tokenizer on the documents
        t.fit_on_texts(self.train_X)
        # summarize what was learned
       # print(t.word_counts)
        #print(t.document_count)
        print(t.word_index)
        #print(t.word_docs)
        # integer encode documents
        self.train_X = t.texts_to_sequences(self.train_X)
        self.test_X = t.texts_to_sequences(self.test_X)
#        print(list(self.train_X[0]))
        