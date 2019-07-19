"""
This file implement transfer lenet model.
The whole model is based on the Lenet tutorial of Theano.

By Xu Zhang

Please refer:
Zhang, X., Yu, F. X., Chang, S. F., & Wang, S. (2015). Deep transfer network: Unsupervised domain adaptation. arXiv preprint arXiv:1503.00591.
"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from load_data import load_source_data,load_target_data,load_transfer_training_data

from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_transfer_lenet5(learning_rate=0.1, alpha = 1, n_epochs=20,
                    source_dataset='../data/resize_mnist_whiten.pkl.gz',
                    target_dataset='../data/usps_whiten.pkl.gz',
                    training_dataset='../data/shuffled_training_data_big_new.pkl.gz',
                    nkerns=[20, 50], batch_size=4000):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_source_data(source_dataset)

    target_datasets = load_target_data(target_dataset)
    
    transfer_training_datasets = load_transfer_training_data(training_dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    target_set_x, target_set_y = target_datasets
    transfer_training_set_x, transfer_training_set_y = transfer_training_datasets 

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_target_batches = target_set_x.get_value(borrow=True).shape[0]
    n_transfer_training_batches = transfer_training_set_x.get_value(borrow=True).shape[0]

    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    n_target_batches /= batch_size
    n_transfer_training_batches /= batch_size    

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
    y_in = T.ivector('y_in')
    ishape = (16, 16)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 16, 16))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 16, 16),
            filter_shape=(nkerns[0], 1, 3, 3), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 7, 7),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 2 * 2,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    
    x_prob = layer3.py_given_x()
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    marginal_params = layer1.params + layer0.params
    
    # calculate marginal MMD and its gradients
    marginal_MMD = T.mean(layer2_input[T.arange(batch_size/2)],axis = 0) - T.mean(layer2_input[T.arange(batch_size/2,batch_size,1)], axis = 0)
    marginal_grads = T.grad(T.dot(marginal_MMD,marginal_MMD), marginal_params)
    
    # the cost we minimize during training is the NLL of the model
    lost_cost = layer3.negative_log_likelihood(y)
    
    # calculate conditional MMD
    conditional_cost_all = T.mean(x_prob[0:batch_size/2:1,0:10:1],axis = 0)\
        - T.mean(x_prob[batch_size/2:batch_size:1,0:10:1],axis = 0)    
    conditional_cost = T.dot(conditional_cost_all,conditional_cost_all)
    
    #add classification loss and conditional MMD all together
    cost = lost_cost + 100*conditional_cost
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    target_model = theano.function([index], layer3.errors(y),
            givens={
                x: target_set_x[index * batch_size: (index + 1) * batch_size],
                y: target_set_y[index * batch_size: (index + 1) * batch_size]})


    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.

    marginal_updates = []
    for param_i, marginal_grad_i in zip(marginal_params, marginal_grads):
        marginal_updates.append((param_i, param_i - 100*learning_rate * marginal_grad_i))
        
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    transfer_model = theano.function([index], [cost, conditional_cost], updates=updates,
          givens={x: transfer_training_set_x[index * batch_size: (index + 1) * batch_size],
                  y: transfer_training_set_y[index * batch_size: (index + 1) * batch_size]})
                
    marginal_model = theano.function([index], marginal_MMD, updates=marginal_updates,
          givens={x: transfer_training_set_x[index * batch_size: (index + 1) * batch_size]})
    
    y_out = layer3.get_output()
            
    target_predict = theano.function([index], y_out,
                givens={
                x: target_set_x[index * batch_size: (index + 1) * batch_size]})
                
    update_target_training_label = theano.function([index], y_out, #updates=label_updates,
          givens={x: transfer_training_set_x[index * batch_size: (index + 1) * batch_size]})
                
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    batch_number = n_transfer_training_batches#n_train_batches
    validation_frequency = min(batch_number, patience / 2)
    update_frequency = min(batch_number, patience / 2)*10
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for minibatch_index in xrange(batch_number):

            iter = (epoch - 1) * batch_number + minibatch_index

            cost = transfer_model(minibatch_index)            
            MMD_margin = marginal_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, batch_number, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of the '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, batch_number,
                           test_score * 100.))
                           
                    target_losses = [target_model(i) for i in xrange(n_target_batches)]
                    target_score = numpy.mean(target_losses)
                    print(('     epoch %i, minibatch %i/%i, target error of the '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_transfer_training_batches,
                           target_score * 100.))
                                           
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    #print('Best validation score of %f %% obtained at iteration %i,'\
    #      'with test performance %f %%' %
    #      (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    noisy_y = target_set_y.eval()                
    for i in xrange(n_target_batches):
        noisy_y[i * batch_size: (i + 1) * batch_size] = target_predict(i)
        
    resizeValidSet = [target_set_x,noisy_y]
    fw = gzip.open("../data/predict_data_CNN_new.pkl.gz", 'wb')
    cPickle.dump(resizeValidSet,fw)
    fw.close()
                          
if __name__ == '__main__':
    evaluate_transfer_lenet5()


def experiment(state, channel):
    evaluate_transfer_lenet5(state.learning_rate, dataset=state.dataset)
