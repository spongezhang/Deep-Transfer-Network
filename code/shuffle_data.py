# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:38:18 2014

Implement data shuffling
@author: Sponge

Please refer:
Zhang, X., Yu, F. X., Chang, S. F., & Wang, S. (2015). Deep transfer network: Unsupervised domain adaptation. arXiv preprint arXiv:1503.00591.
"""
import cPickle
import gzip

import numpy
from numpy import *

import random
import copy

import pdb

batch_size = 4000
class_number = 10
category_batch_size = batch_size

fsource = gzip.open('../data/resize_mnist_whiten.pkl.gz', 'rb')
ftarget = gzip.open('../data/usps_whiten.pkl.gz', 'rb')
ftarget1 = gzip.open('../data/predict_data_CNN_new.pkl.gz', 'rb')
#
source_batch_size = 2000
target_batch_size = batch_size - source_batch_size
train_set, valid_set, test_set = cPickle.load(fsource)
target_set = cPickle.load(ftarget)
target_set1 = cPickle.load(ftarget1)
fsource.close()
ftarget.close()
ftarget1.close()

##sort source samples 
data_mat = train_set[0].copy()
label_mat = train_set[1].copy()
shuffle_source_data = data_mat.copy()
shuffle_source_label = label_mat.copy()
sample_dimension = len(data_mat[0,:])
sample_number = len(data_mat[:,0])
index_list = range(sample_number)
random.shuffle(index_list)

for i in xrange(sample_number):
    shuffle_source_data[i,:] = data_mat[index_list[i],:]
    shuffle_source_label[i] = label_mat[index_list[i]]
    
source_batch_number = sample_number/source_batch_size

data_mat = target_set[0].copy()
#using the noisy label, never use the gt label
label_mat = target_set1[1].copy()
shuffle_target_data = data_mat.copy()
shuffle_target_label = label_mat.copy()
sample_number = len(data_mat[:,0])
index_list = range(sample_number)
random.shuffle(index_list)

for i in xrange(sample_number):
    shuffle_target_data[i,:] = data_mat[index_list[i],:]
    shuffle_target_label[i] = label_mat[index_list[i]]
    
target_batch_number = sample_number/target_batch_size

final_data = numpy.zeros((source_batch_number*batch_size,sample_dimension),dtype=float32)
processedLabel = numpy.zeros((source_batch_number*batch_size),dtype=int64)

for i in xrange(source_batch_number):
        cur_index = i%(source_batch_number)
        final_data[(i*batch_size):(i*batch_size+source_batch_size)]\
         = shuffle_source_data[cur_index*source_batch_size:(cur_index+1)*source_batch_size]
         
        processedLabel[(i*batch_size):(i*batch_size+source_batch_size)]\
         = shuffle_source_label[cur_index*source_batch_size:(cur_index+1)*source_batch_size]

        cur_index = i%(target_batch_number)
        final_data[i*batch_size + source_batch_size:(i+1)*batch_size]\
         = shuffle_target_data[cur_index*target_batch_size:(cur_index+1)*target_batch_size]
        processedLabel[i*batch_size + source_batch_size:(i+1)*batch_size]\
         = shuffle_target_label[cur_index*target_batch_size:(cur_index+1)*target_batch_size]
         
resizeValidSet = [final_data,processedLabel]

fw = gzip.open("../data/shuffled_training_data_big_new.pkl.gz", 'wb')
cPickle.dump(resizeValidSet,fw)
fw.close()
