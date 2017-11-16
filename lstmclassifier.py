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
from sequence_classifier import SequenceClassifier

class LSTMClassifier(SequenceClassifier):
    def __init__(self,train_X, train_Y, test_X, test_Y, no_of_epochs,learning_rate,no_of_feats,sentence_length,embedding_dim):
        super(LSTMClassifier,self).__init__(train_X, train_Y, test_X, test_Y,no_of_epochs,learning_rate,no_of_feats,sentence_length,embedding_dim)
        self.classifier.add(LSTM(100))
        self.classifier.add(Dense(1, activation='sigmoid'))
        adma_optimizer = optimizers.Adam(lr=self.leanrningRate)
        self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def train(self):
        print(self.classifier.summary())
        self.classifier.fit(self.train_X, self.train_Y, epochs=self.no_of_epochs, batch_size=64)
        
    def test(self):
        scores = self.classifier.evaluate(self.test_X, self.test_Y, verbose=0)
        return scores[1]