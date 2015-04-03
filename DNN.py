

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano import function
from HiddenLayer import HiddenLayer

class DNN(object):
	def __init__(self, rng=numpy.random.RandomState(1234),  n_in=69, n_out=48, n_hidden=128, layer=2, activation=None):
	
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
		self.n_in = n_in
		self.n_out = n_out
		self.n_hidden = n_hidden
		self.layer = layer
		self.MLP = []
		for i in range(layer+1):
			self.MLP.append(HiddenLayer(
				rng = rng,
				n_in = 69 if i==0 else n_hidden,
				n_out = 128 if i!= layer else 48,
				activation = T.tanh if i!= layer else None
			))
	def forward(self,input):
		for i in range(self.layer+1):
			self.MLP[i].compute(input = input if i==0 else self.MLP[i-1].output)
		demo = self.MLP[self.layer].output.eval()
		print "shape:{}".format(demo.shape)
		#print demo
		
	def backward(self, answer):
            z = T.dvector('z')
            cost = T.sum((T.nnet.softmax(z) - answer)**2)#  sum of  ( softmax(output array) )^2 

            cost_grad = function([z], T.grad(cost,z))
            cost = function([z], cost)
            
            dnn_output = self.MLP[self.layer].output.eval()
            last_delta = cost_grad(dnn_output)# the delta of the last layer

            #    last layer
            self.MLP[self.layer].update(
                0.1,
numpy.transpose(T.dot(last_delta.reshape(48,1), self.MLP[self.layer-1].output.eval().reshape(1,128)).eval()),                last_delta)
            
            
if __name__ == '__main__':
	DNN() 



