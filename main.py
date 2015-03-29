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


file = open('train.ark','r')

n_in = 69
n_hidden = 128
rng = numpy.random.RandomState(1234)
name = []

for line in file:
	X = line.split()
	X = map(float, X[1:])
	Y = numpy.asarray(X)
	print Y.shape
	hiddenLayer = HiddenLayer(
            rng=rng,
            input=Y,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
	print hiddenLayer.output.eval()

