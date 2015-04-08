"""

References: 
    http://deeplearning.net/tutorial/mlp.html
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5
.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),    

"""
__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy

import theano
import theano.tensor as T

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=numpy.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
	self.n_in = n_in
	self.n_out = n_out
	self.activation = activation
        if W is None:
            W_values = numpy.array(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

        self.W = W_values
        self.b = b_values
        # parameters of the model
        self.params = [self.W, self.b]
    
    def update(self, momentum, learning_rate, W_update=None, b_update=None):
        start_ = time.time()
	self.W = momentum*self.W + learning_rate*W_update
	self.b = momentum*self.b + learning_rate*b_update
        end_ = time.time()
    def compute(self, input):
	lin_output = numpy.dot(input, self.W) + self.b
        self.lin_output = lin_output
	self.output = (
		lin_output if self.activation is None
		else self.activation(lin_output)
	)

if __name__ == '__main__':
    HiddenLayer()

