'''
Only read the data as input here



'''



import os
import sys
import time

import numpy

import theano
import theano.tensor as T



file = open('train.ark','r')


for line in file:
		X = line.split()
		name = X[0]
		X = X[1:]
		