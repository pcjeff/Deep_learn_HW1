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
total = 1124823 - 4823

print 'start reading'

#reading train
for line in file_lab:
    X = line.split(',')
    label.append(int(X[1]))
for i in range(total):
    line = file_ark.readline()
    X = line.split()
    X = map(float, X[1:])
    Y = numpy.array(X)
    ark.append(Y)

with open("Num_to_39")as f:
    number_map={}
    for line_number,line in enumerate(f.readlines()): 
        number_map[line_number]=line  

print 'end reading'
file_lab.close()
file_ark.close()

L = numpy.zeros(48)
part = 0
max_epoch = 10


print 'start'

#train
for epoch in range(max_epoch):
    shuffle(X)
    for i in range(X):
        dnn.forward(X.reshape(69, 1))   
        dnn.backward(X, numpy.array(L))

#test

#reading test
for line in test_ark:
    X = line.split(',')
    label.append(int(X[1]))
for i in range(total):
    line = test_ark.readline()
    X = line.split()
    print X[0],
    X = map(float, X[1:])
    Y = numpy.array(X)
    ark.append(Y)

global error
error=0.0

for X, Y in zip(ark, label):
    L[Y-1] = 1
    predict=dnn.forward(X.reshape(69, 1))
    if predict != L.argmax():
        error+=1
    L[Y-1] = 0
    print number_map[predict]

print str(error)+"/1000000"


print 'end'


