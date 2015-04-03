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
	layer = 5, #4 hidden layers
	activation = T.tanh
)

'''
label_dict = {}
for line in file_lab:
    X = line.split(',')
    label_dict.update({X[0]: int(X[1])})
print label_dict
'''
for line in file_ark:
    X = line.split()
    X = map(float, X[1:])
    Y = numpy.asarray(X)
    dnn.forward(Y)
    dnn.forward(Y)
    
    dnn.backward(numpy.zeros((1,48),dtype=theano.config.floatX))
    #print hiddenLayer.output.eval()

    
    


