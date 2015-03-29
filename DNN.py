

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from HiddenLayer import HiddenLayer

class DNN(object):
	def __init__(self, rng=numpy.random.RandomState(1234), input=None, n_in=69, n_out=48, n_hidden=128, layer=1, activation=None):
	
		"""
		:type rng: numpy.random.RandomState
		:param rng: a random number generator to initalize weights

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes input data ex: an 69*1 array

		:type n_in: int
		:param n_in: input dimension of the input data 

		:type n_out: int
		:param n_out: number of output units 

		:type layer: int
		:param layer: number of layers, the unit of each layer is 128 by initial

		:type n_hidden: int
		:param n_hidden: number of units of each hidden layer

		:type activation: theano.Op or function
		:param activation: sigmod or tanh
		"""
	
		#input = numpy.ones((1,69), 'float32')
		self.MLP = []
		for i in range(layer+1):
			self.MLP.append(HiddenLayer(
				rng = rng,
				input = input if i==0 else self.MLP[i-1].output,
				n_in = 69 if i==0 else n_hidden,
				n_out = 128 if i!= layer else 48,
				activation = T.tanh if i!= layer else None
			))
			print "n_in:{0} n_out:{1}".format(
			69 if i==0 else n_hidden,
			128 if i!= layer else 48)
		#for i in self.MLP:
		#	print i.output.eval()

if __name__ == '__main__':
	DNN() 



