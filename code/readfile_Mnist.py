# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:38:18 2014

@author: Sponge
"""

import cPickle
import gzip

import numpy
from numpy import *

from whiten import whiten

import scipy
import scipy.misc
import pdb
resizeSize = 16

print '... loading data'

fr = gzip.open('../data/mnist.pkl.gz', 'rb')

fw = gzip.open('../data/resize_mnist_whiten.pkl.gz', 'wb')

#test_set = cPickle.load(fr)
#label = test_set[1]
train_set, valid_set, test_set = cPickle.load(fr)

lengthx = len(train_set[0][:,0])
trainData = numpy.zeros((lengthx,resizeSize**2),dtype=float32)

for i in xrange(len(train_set[0][:,1])):
    a0 = train_set[0][i,:]
    a0 = a0.reshape(28, 28)
    a0 = scipy.misc.imresize(a0,[16,16])
    a0 = a0.reshape(256)
    trainData[i,:] = a0

trainData = whiten(trainData)
resizeTrainSet = [trainData,train_set[1]]

lengthx = len(test_set[0][:,0])
testData = numpy.zeros((lengthx,resizeSize**2),dtype=float32)

for i in xrange(len(test_set[0][:,1])):
    a0 = test_set[0][i,:]
    a0 = a0.reshape(28, 28)
    a0 = scipy.misc.imresize(a0,[16,16])
    a0 = a0.reshape(256)
    testData[i,:] = a0

testData = whiten(testData)
resizeTestSet = [testData,test_set[1]]

lengthx = len(valid_set[0][:,0])
validData = numpy.zeros((lengthx,resizeSize**2),dtype=float32)

for i in xrange(len(valid_set[0][:,1])):
    a0 = valid_set[0][i,:]
    a0 = a0.reshape(28, 28)
    a0 = scipy.misc.imresize(a0,[16,16])
    a0 = a0.reshape(256)
    validData[i,:] = a0
    
validData = whiten(validData)
resizeValidSet = [validData,valid_set[1]]

dataset = [resizeTrainSet,resizeValidSet,resizeTestSet]
cPickle.dump(dataset,fw)

fr.close()
fw.close()
