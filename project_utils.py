# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:09:10 2017

@author: madha
"""
import re
import string
from nltk.corpus import stopwords
from nltk.util import ngrams

def parse_label(text):
    if(text == 'The tweet is not offensive'):
        return 0
    elif(text == 'The tweet uses offensive language but not hate speech'):
        return 1
    elif(text == 'The tweet contains hate speech'):
        return 2

def pre_process(tweet):
#    removechars = set(string.punctuation)
#    word = word.lower()
#    word = ''.join(ch for ch in word if ch not in removechars)
    tweet = tweet.lower()    
    #Remove urls
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)    
    #Remove usernames
    tweet = re.sub('@[^\s]+','',tweet)    
    #Remove white space
    tweet = tweet.strip()    
    #Remove hashtags
    tweet = re.sub(r'#([^\s]+)', '', tweet)   
    #Remove stopwords
    tweet = " ".join([word for word in tweet.split(' ') if word not in stopwords.words('english')])
    #Remove punctuation
    tweet = "".join(l for l in tweet if l not in string.punctuation)
    print(tweet)
    return tweet

#Converting to List to Words
def stringToListofWords(sentence_list):
    wordList = []
    for i in range(len(sentence_list)):
        listOfWords = sentence_list[i].split()
        wordListSet = set(listOfWords)
        newWordList = list(wordListSet)
        wordList.extend(newWordList)
    return wordList

class Classifier(object):
    def __init__(self, train_X, train_Y, test_X, test_Y, no_of_epochs,learningRate,no_of_feats):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.no_of_epochs = no_of_epochs
        self.leanrningRate = learningRate
        self.no_of_feats = no_of_feats

    def train(self):
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def test(self):
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")