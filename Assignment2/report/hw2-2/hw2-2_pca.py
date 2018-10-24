# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:34:49 2018

@author: user
"""
import os
import numpy as np
from scipy import linalg as LA
from skimage.io import imread, imshow, imsave


class PCA():
    def __init__(self):
        return

    def fit(self, X):
        self.mean = X.mean(axis=0)
        mean = np.tile(self.mean, (X.shape[0], 1))
        X -= mean

        CoVar = np.cov(X, rowvar=False)
        
        evals, evecs = LA.eigh(CoVar)
        
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        
        self.T = evecs[:, :X.shape[0]]
        self.evals = evals
        return
    
    def transform(self, X, dims):
        mean = np.tile(self.mean, (X.shape[0], 1))
        X -= mean
        X_ = np.dot(X, self.T[:dims])
        
        return X_
    
    def reconstruct(self, X_):
        X = np.dot(X_, self.T[:X_.shape[1]].T)
        mean = np.tile(self.mean, (X.shape[0], 1))
        X += mean
        
        return X

if __name__ == '__main__':
    
    h = 56
    w = 46
    img_ls = []
    
    for sub in range(40):
        sub = sub + 1
        for i in range(7):
            i = i + 1
            
            I = imread(os.path.join('hw2-2_data', '{}_{}.png'.format(sub, i)))
            I = I.astype('float32')
            img_ls.append(np.reshape(I, -1))
            
    img_ls = np.array(img_ls)
    pca = PCA()
    pca.fit(img_ls.copy())
    
    mean = np.reshape(pca.mean, (h, w))
    mean = mean.astype('uint8')
    imsave('mean.png', mean)
    for i in range(5):
        eigen = pca.T[:, i]
        eigen = np.reshape(eigen, (h, w))
        eigen = (eigen - np.min(eigen)) / (np.max(eigen) - np.min(eigen))  
        imsave('eigen{}.png'.format(i+1), eigen)
    
        
        
        
    
    
