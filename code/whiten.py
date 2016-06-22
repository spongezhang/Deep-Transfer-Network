# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:16:31 2014

@author: Sponge
"""
from numpy import dot, sqrt, diag
from numpy.linalg import eigh
import pdb

def whiten(X,fudge=1E-5,balance=1000):
    
   # the matrix X should be observations-by-components
   # get the covariance matrix
   Xcov = dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d,V = eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.

   D = diag(balance/sqrt(d+fudge))
   
   # whitening matrix
   W = dot(dot(V,D),V.T)

   # multiply by the whitening matrix
   X = dot(X,W)

   return X