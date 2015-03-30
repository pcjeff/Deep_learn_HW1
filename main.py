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

file = open('new_train.ark','r')

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
	layer = 5,
	activation = T.tanh
)

for line in file:
	X = line.split()
	X = map(float, X[1:])
	Y = numpy.asarray(X)
	dnn.forward(Y)
	print Y.shape
	#print hiddenLayer.output.eval()

