'''
Only read the data as input here
'''
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from random import shuffle
from HiddenLayer import HiddenLayer
from DNN import DNN

file_ark = open('new_train.ark','r')
file_lab = open('new_train.lab','r')
test_ark = open('test.ark','r')

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
        layer = 1, #N layers : N hidden layers 1 output layer
	activation = numpy.tanh
)

ark = []
label = []
# total = 1124823 - 4823

print 'start reading'

#reading train
for line in file_lab:
    X = line.split(',')
    label.append(int(X[1]))
for line in file_ark:
    X = line.split()
    X = map(float, X[1:])
    Y = numpy.array(X)
    ark.append(Y)

with open("Num_to_39.map")as f:
    number_map={}
    for line_number,line in enumerate(f.readlines()): 
        number_map[line_number]=line.replace('\n','') 

print 'end reading'
file_lab.close()
file_ark.close()

L = numpy.zeros(48)
max_epoch = 5


print 'start'

#train
for epoch in range(max_epoch):
    ark_range=range(len(ark))
    shuffle(ark_range)
    error=0
    for i in ark_range:
        L[label[i]-1] = 1
        if dnn.forward(ark[i].reshape(69, 1))!=L.argmax():
		error+=1
        dnn.backward(ark[i], numpy.array(L))
        L[label[i]-1] = 0
    print str(error)+"/per epoch"

#test

#reading test

for line in test_ark:
    X = line.split()
    Y = map(float, X[1:])
    print X[0] + ',' + str(number_map[dnn.forward(numpy.array(Y).reshape(69, 1))])

print 'end'


