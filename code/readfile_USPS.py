import scipy
import scipy.io

import numpy

from numpy import *
from whiten import whiten

import cPickle
import gzip

import pdb

fw = gzip.open('../data/usps_new_whiten.pkl.gz', 'wb')

mat = scipy.io.loadmat('../data/usps_all.mat')
dataMat = mat['data']
numNumber = len(dataMat[0,0,:])
sampleNumber = len(dataMat[0,:,0])
sampleDim = len(dataMat[:,0,0])
processedData = numpy.zeros((numNumber*sampleNumber,sampleDim),dtype=float32)
processedLabel = numpy.zeros((numNumber*sampleNumber),dtype=int64)

for i in xrange(numNumber):
    for j in xrange(sampleNumber):
        a0 = dataMat[:,j,i]
        a0 = a0.reshape([16,16])
        a0 = a0.transpose();
        a0 = a0.reshape(256)
        processedData[i*sampleNumber+j,:] = a0
    if i == 9:
        processedLabel[i*sampleNumber:(i+1)*sampleNumber] = numpy.zeros(sampleNumber)
    else:
        processedLabel[i*sampleNumber:(i+1)*sampleNumber] = (i+1)*numpy.ones(sampleNumber)

pdb.set_trace()
processedData = whiten(processedData)

dataset = [processedData,processedLabel]

cPickle.dump(dataset,fw)

fw.close()
