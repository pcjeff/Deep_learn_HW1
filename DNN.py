

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano import function
from HiddenLayer import HiddenLayer

x = T.dscalar('x')
activation = T.tanh(x)
tanh_grad = function([x], T.grad(activation, x))

z = T.dvector('z')
ans = T.dvector('ans')
cost = T.sum(((z/T.sqrt(T.sum(z**2))) - ans)**2)#  sum of  ( softmax(output array) )^2 
cost_grad = function([z,ans], T.grad(cost, z))
#need to input the list of z_i, and turn the output into array
            
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
		for i in range(0,layer+1):
			self.MLP.append(HiddenLayer(
				rng = rng,
				n_in = 69 if i==0 else n_hidden,
				n_out = 128 if i!= layer else 48,
				activation = numpy.tanh if i!= layer else None
			))
	def forward(self,input):
		for i in range(self.layer+1):
			self.MLP[i].compute(input = input if i==0 else self.MLP[i-1].output)
		return self.MLP[self.layer].output.argmax()
		
	def backward(self, input, answer):
            dnn_output = self.MLP[self.layer].output
            last_delta = cost_grad(dnn_output.reshape(48),answer.reshape(48))# the delta of the last layer
            #  update  last layer
            self.MLP[self.layer].update(
                0.1,
            numpy.dot(last_delta.reshape(48,1), self.MLP[self.layer-1].output.reshape(1,128)),      
            last_delta.reshape(48))

            #update layer-1 ~ 0
            for i in range(self.layer-1, -1, -1):

                out = self.MLP[i+1].n_out
                W_delta = numpy.dot(numpy.transpose(self.MLP[i+1].W), last_delta.reshape(out, 1))#array of delta * W
                pa_pz = numpy.array(map(tanh_grad, self.MLP[i].lin_output.reshape(self.MLP[i].n_out).tolist()))#array of partial a / partial z
                last_delta = numpy.array([a*b for a,b in zip(W_delta, pa_pz.reshape(128, 1))])
                self.MLP[i].update(
                        0.1,
                        numpy.dot(last_delta.reshape(128,1),
                        self.MLP[i-1].output.reshape(1, 128) if i!=0 else input.reshape(1,69)),
                        last_delta.reshape(128))
                
if __name__ == '__main__':
	DNN() 
