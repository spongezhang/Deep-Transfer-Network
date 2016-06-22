# Deep-Transfer-Network

This is the implementation of the deep transfer network.

Please refer:
Zhang, X., Yu, F. X., Chang, S. F., & Wang, S. (2015). Deep transfer network: Unsupervised domain adaptation. arXiv preprint arXiv:1503.00591.

Prerequisition,
Installing Anaconda(https://www.continuum.io/downloads) and Theano(http://www.deeplearning.net/software/theano/install.html)

The easiest way to try the code is to run the run.sh script

readfile_USPS.py, preprocessing the USPS dataset.
readfile_Mnist.py, preprocesses the MNIST dataset.
shuffle_data.py, preprocesses the training data for transfer learning.
transferLeNet.py, the model of deep transfer network.

