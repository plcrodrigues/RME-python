#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:36:59 2016

@author: coelhorp
"""

import numpy as np
import scipy as sp

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.utils.base     import powm, sqrtm, invsqrtm, logm
from pyriemann.utils.distance import distance_riemann, distance_logeuclid
from pyriemann.utils.mean     import mean_riemann, mean_euclid

from random import randrange

class RDR(BaseEstimator, TransformerMixin):    
    '''Riemannian Dimension Reduction
    
    Dimension reduction respecting the riemannian geometry of the SPD manifold.
    
    Parameters
    ----------
    n_components: int (default: 6)
        The number of components to reduce the dataset. 
    method: string (default: nrme)
        Which method should be used to reduce the dimension of the dataset.
        Different approaches use different cost functions and algorithms for
        solving the optimization problem. The options are:        
            - nrme-uns
            - nrme-uns-random (set nmeans, npoints)
            - covpca             
    '''
    
    def __init__(self, n_components=6, method='nrme', params={}):          
        self.n_components = n_components
        self.method = method
        self.params = params
        
    def fit(self, X, y=None):        
        self._fit(X, y)
        return self

    def transform(self, X, y=None):        
        Xnew = self._transform(X)
        return Xnew
        
    def _fit(self, X, y):   
             
        methods = {
                   'nrme-uns'        : dim_reduction_nrmeuns,
                   'nrme-uns-random' : dim_reduction_nrmeuns_random,               
                   'covpca'          : dim_reduction_covpca,                   
                  }    
                                   
        self.projector_ = methods[self.method](X=X,
                                               P=self.n_components,
                                               labels=y,
                                               params=self.params)                                         
    
    def _transform(self, X):        
        K = X.shape[0]
        P = self.n_components
        W = self.projector_    
        Xnew = np.zeros((K, P, P))        
        for k in range(K):            
            Xnew[k, :, :] = np.dot(W.T, np.dot(X[k, :, :], W))                        
        return Xnew 
    
def dim_reduction_nrmeuns(X, P, labels, params):
    
    K = X.shape[0]
    nc = X.shape[1]    
    
    S = np.zeros((nc, nc))
    
    for i in range(K):
        for j in range(K):   
            if i != j:                             
                Ci, Cj = X[i, :, :], X[j, :, :]            
                Sij = np.dot(invsqrtm(Ci), np.dot(Cj, invsqrtm(Ci)))                
                S = S + powm(logm(Sij), 2)
            
    l,v = np.linalg.eig(S)
    idx = l.argsort()[::-1]   
    l = l[idx]
    v = v[:, idx]

    W = v[:, :P]    

    return W       

def dim_reduction_nrmeuns_random(X, P, labels, params):

    nc = X.shape[1] 
    K  = X.shape[0]
    
    nmeans = params['nmeans']
    npoints = params['npoints']

    # calculate the means
    Xm = np.zeros((nmeans, nc, nc))
    for n, sn in enumerate(range(nmeans)):
        selectmeans = [randrange(0, K) for _ in range(npoints)]
        Xm[n] = mean_riemann(X[selectmeans])        
           
    W = dim_reduction_nrmeuns(Xm, P, labels, params)

    return W     

def dim_reduction_covpca(X, P, labels, params): 
    
    Xm  = np.mean(X, axis=0)
    w,v = np.linalg.eig(Xm)
    idx = w.argsort()[::-1]
    v = v[:, idx]
    W = v[:, :P]
    
    return W      
