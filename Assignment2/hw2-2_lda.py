# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:08:24 2018

@author: user
"""
import os
import sys
import itertools

import numpy as np
from scipy import linalg as LA

from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

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

class LDA():
    
    def __init__(self):
        return

    def fit(self, X, y):
        X = X.copy()
        
        classes = np.unique(y)
        mean = X.mean(axis=0)
        
        Sw = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
        Sb = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
        
        for c in classes:
            Xc = X[y==c,:]
            meanc = Xc.mean(axis=0)
            Sw = Sw + np.dot((Xc-meanc).T, (Xc-meanc))
            Sb = Sb + (Xc.shape[0] * np.dot((meanc - mean).T, (meanc - mean)))
            
        evals, evecs = np.linalg.eig(np.linalg.inv(Sw)*Sb)
        idx = np.argsort(evals.real)[::-1]
        evecs = evecs[:,idx]
        evals = evals[idx]
        
        self.T = evecs[:, :classes.shape[0]-1]
        self.evals = evals

        return
    
    def transform(self, X, dims):

        X_ = np.dot(X, self.T[:, :dims])
        
        return X_
    
if __name__ == '__main__':
    
    trainX = []
    trainy = []
    
    img_name_ls = os.listdir(sys.argv[1])
    
    for img_name in img_name_ls:
        
        I = imread(os.path.join(sys.argv[1], img_name))
        h, w = I.shape
        I = np.reshape(I, -1)
        I = I.astype('float32')
        trainX.append(I)
        trainy.append(int(img_name.split('_')[0]))
            
    trainX = np.array(trainX)
    
    pca = PCA()
    lda = LDA()
    pca.fit(trainX)
    trainX_ = pca.transform(trainX, trainX.shape[0] - np.unique(trainy).shape[0])
    lda.fit(trainX_, trainy)
    
    
    eigen = lda.T[:, 0:1]
    eigen = np.dot(pca.T[:, :eigen.shape[0]], eigen)
    eigen = np.reshape(eigen, (h, w))
    eigen = (eigen-np.min(eigen))*255 / (np.max(eigen)-np.min(eigen))
    eigen = eigen.astype('uint8')
    imsave(sys.argv[2], eigen)