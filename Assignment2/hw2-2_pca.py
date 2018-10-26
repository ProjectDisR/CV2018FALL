# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:34:49 2018

@author: user
"""

import os
import sys

import numpy as np
from scipy import linalg as LA

from skimage.io import imread, imsave


class PCA():
    
    def __init__(self):
        return

    def fit(self, X):
        X = X.copy()

        self.mean = X.mean(axis=0)
        mean = np.tile(self.mean, (X.shape[0], 1))
        X -= mean

        CoVar = np.cov(X, rowvar=False)        
        evals, evecs = LA.eigh(CoVar)        
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        
        self.T = evecs[:, :X.shape[0]-1]
        self.evals = evals
        
        return
    
    def transform(self, X, dims):
        X = X.copy()
        
        mean = np.tile(self.mean, (X.shape[0], 1))
        X -= mean
        X_ = np.dot(X, self.T[:, :dims])
        
        return X_
    
    def reconstruct(self, X_):
        
        X = np.dot(X_, self.T[:, :X_.shape[1]].T)
        mean = np.tile(self.mean, (X.shape[0], 1))
        X += mean
        
        return X

if __name__ == '__main__':
    trainX = []
    
    img_name_ls = os.listdir(sys.argv[1])
    
    for img_name in img_name_ls:
        
        I = imread(os.path.join(sys.argv[1], img_name))
        I = np.reshape(I, -1)
        I = I.astype('float32')
        trainX.append(I)
            
    trainX = np.array(trainX)
    
    pca = PCA()
    pca.fit(trainX)
    
    I = imread(sys.argv[2])
    h, w = I.shape
    I = np.reshape(I, (1, -1))
    I = I.astype('float32')
    
  
    I_recons = pca.reconstruct(pca.transform(I, trainX.shape[0]-1))
    imsave(sys.argv[3], np.reshape(I_recons, (h, w)).astype('uint8'))