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
	activation = numpy.tanh
)

ark = []
label = []
total = 1124823 - 4823

for line in file_lab:
    X = line.split(',')
    label.append(int(X[1]))
for i in range(total):
    line = file_ark.readline()
    X = line.split()
    X = map(float, X[1:])
    Y = numpy.array(X)
    ark.append(Y)

file_lab.close()
file_ark.close()

L = numpy.zeros((48,),dtype=theano.config.floatX)
i=0
error = 0
part = 0

for Y in ark:
    L[label[i]] = 1
    if dnn.forward(Y.reshape(1, 69)) != L.argmax():
        error = error + 1
    dnn.backward(Y, L)

    L[label[i]] = 0
    i = i+1
    if i%10000 == 0:
        print '{}...Ein:{}'.format(part, error/10000)
        part = part + 1
        error = 0
    #print '-----------------{}--------------------'.format(i)
    


