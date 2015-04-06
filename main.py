'''
Only read the data as input here
'''
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from HiddenLayer import HiddenLayer
from DNN import DNN

file_ark = open('new_train.ark','r')
file_lab = open('new_train.lab','r')

n_in = 69
n_hidden = 128
n_out = 48
rng = numpy.random.RandomState(1234)
name = []


dnn = DNN(
	rng=rng,
	n_in = 69,
	n_out = 48,
	n_hidden = 128,
        layer = 2, #N layers : N hidden layers 1 output layer
	activation = T.tanh
)

label = []
for line in file_lab:
    X = line.split(',')
    label.append(int(X[1]))



L = numpy.zeros((48,),dtype=theano.config.floatX)
i=0
for line in file_ark:
    X = line.split()
    X = map(float, X[1:])
    Y = numpy.asarray(X)
    L[label[i]] = 1
    dnn.forward(Y)
    dnn.backward(Y, L)
    L[label[i]] = 0
    i = i+1
    print '-----------------{}--------------------'.format(i)
    
    


