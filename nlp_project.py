# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:31:50 2017

@author: madha
"""
import project_utils
from charngram_svm import SVMClassifier
from lstmclassifier import LSTMClassifier
from gruclassifier import GRUClassifier
from rnnclassifier import RNNClassifier
import sys
from sklearn.model_selection import train_test_split
import csv

NO_OF_FEATS = 1000

def main():
    trainingfilename = sys.argv[1]
    model = int(sys.argv[2])
    print('Training filename: ',trainingfilename)
    trainingreader = csv.reader(open(trainingfilename,'r',encoding='utf8',errors='ignore'),delimiter=',')
    sentences = []
    labels = []
    i = 0
    for row in trainingreader:
        if(i==0):
            i = i + 1
            continue
        sentences.append(project_utils.pre_process(str(row[6])))
        labels.append(int(row[5]))
        i = i + 1
    print(sentences[0])
    print(sentences[1])
    print(len(sentences))
    train_X, test_X, train_Y, test_Y = train_test_split(sentences,labels,test_size = 0.33, random_state=42)
    if(model==0 or model==4):
        #SVM
        baseline_svm = SVMClassifier(train_X, train_Y, test_X, test_Y, 10,0.1,NO_OF_FEATS)
        baseline_svm.train()
        score = baseline_svm.test()
        print("SVM accuracy: " + str(score))
    if(model==1 or model==4):
        #RNN
        rnn_classifier = RNNClassifier(train_X, train_Y, test_X, test_Y, 10,0.001,NO_OF_FEATS,128,32)
        rnn_classifier.train()
        score = rnn_classifier.test()
        print("RNN accuracy: "+ str(score))
    if(model==2 or model==4):
        #LSTM
        lstm_classifier = LSTMClassifier(train_X, train_Y, test_X, test_Y, 10,0.001,NO_OF_FEATS,128,32)
        lstm_classifier.train()
        score = lstm_classifier.test()
        print("LSTM accuracy: "+ str(score))
    if(model==3 or model==4):
        #GRU
        gru_classifier = GRUClassifier(train_X, train_Y, test_X, test_Y, 10,0.001,NO_OF_FEATS,128,32)
        gru_classifier.train()
        score = gru_classifier.test()
        print("GRU accuracy: "+ str(score))

main()