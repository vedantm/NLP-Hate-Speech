# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:54:02 2017

@author: madha
"""
import sys
import csv
import string
import re
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from sklearn.pipeline import Pipeline
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from project_utils import Classifier
from sklearn.metrics import accuracy_score

class SVMClassifier(Classifier):
    
    def __init__(self,train_X, train_Y, test_X, test_Y, no_of_epochs,learning_rate,no_of_feats):
        super(SVMClassifier,self).__init__(train_X, train_Y, test_X, test_Y,no_of_epochs,learning_rate,no_of_feats)
        self.classifier = Pipeline([('vect', CountVectorizer(analyzer='char_wb', ngram_range=(3,4),max_features=self.no_of_feats)),
                             ('tfidf', TfidfTransformer()),
                             ('clf', LinearSVC(multi_class="crammer_singer",max_iter=self.no_of_epochs,verbose=1)),
                             ])
        
    def train(self):
        global NO_OF_FEATURES
        self.classifier = self.classifier.fit(self.train_X,self.train_Y)
        
    def test(self):
        pred_y = self.classifier.predict(self.test_X)        
        return accuracy_score(self.test_Y,pred_y)

    