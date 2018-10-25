# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:08:24 2018

@author: user
"""
import os
import itertools
import visdom

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
    vis = visdom.Visdom(env='hw2')
    
    h = 56
    w = 46
    
    trainX = []
    trainy = []
    testX = []
    testy = []
    
    img_name_ls = os.listdir('hw2-2_data/')
    
    for img_name in img_name_ls:
        
        I = imread(os.path.join('hw2-2_data/', img_name))
        I = np.reshape(I, -1)
        I = I.astype('float32')
        
        if int(img_name.split('_')[1].split('.')[0]) <= 7:      
            trainX.append(I)
            trainy.append(int(img_name.split('_')[0]))
            
        else:
            testX.append(I)
            testy.append(int(img_name.split('_')[0]))
            
    trainX = np.array(trainX)
    testX = np.array(testX)
    
    pca = PCA()
    lda = LDA()
    pca.fit(trainX)
    trainX_ = pca.transform(trainX, 240)
    lda.fit(trainX_, trainy)
    
    for i in range(5):
        eigen = lda.T[:, i:i+1]
        eigen = np.dot(pca.T[:, :eigen.shape[0]], eigen)
        eigen = np.reshape(eigen, (h, w))
        eigen = (eigen-np.min(eigen))*255 / (np.max(eigen)-np.min(eigen))
        eigen = eigen.astype('uint8')
        imsave('fisher{}.png'.format(i+1), eigen)
    
    
    
    
    
    
    
    
    
    testX_ = pca.transform(testX, 240)
    testX_ = lda.transform(testX_, 30)
    vis.scatter(TSNE().fit_transform(testX_), testy, win='PCA+LDA', opts={'title': 'PCA+LDA'})
    
    
    
    
    
    
    
    
    
    
    kfold = KFold(n_splits=3)
    folder_ls = ['0/', '1/', '2/']
    
    k_ls = [1, 3, 5]
    n_ls = [3 ,10, 39]
    
    for k, n in itertools.product(k_ls, n_ls):
        accuracy = []
        
        for train_index_ls, valid_index in kfold.split(folder_ls):
            
            trainX = []
            trainy = []
            validX = []
            validy = []
            
            for i in train_index_ls:
                img_name_ls = os.listdir(os.path.join('train/', folder_ls[i]))
                
                for img_name in img_name_ls:
                    
                    I = imread(os.path.join('train/', folder_ls[i], img_name))
                    I = np.reshape(I, -1)
                    I = I.astype('float32')
                    
                    trainX.append(I)
                    trainy.append(int(img_name.split('_')[0]))
              
            img_name_ls = os.listdir(os.path.join('train/', folder_ls[valid_index[0]]))
            
            for img_name in img_name_ls:
                
                I = imread(os.path.join('train/', folder_ls[valid_index[0]], img_name))
                I = np.reshape(I, -1)
                I = I.astype('float32')
                
                validX.append(I)
                validy.append(int(img_name.split('_')[0]))
            
            trainX = np.array(trainX)
            validX = np.array(validX)
            
            pca = PCA()
            lda = LDA()
            pca.fit(trainX)
            trainX_ = pca.transform(trainX, trainX.shape[0]-40)
            lda.fit(trainX_, trainy)
            trainX_ = lda.transform(trainX_, n)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(trainX_, trainy) 

            validX_ = pca.transform(validX, trainX.shape[0]-40)
            validX_ = lda.transform(validX_, n)
            accuracy.append(knn.score(validX_, validy))
            
        accuracy.append(np.mean(np.array(accuracy)))
        print(k, n, accuracy[0], accuracy[1], accuracy[2], accuracy[3])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    trainX = []
    trainy = []
    testX = []
    testy = []
    
    img_name_ls = os.listdir('hw2-2_data/')
    
    for img_name in img_name_ls:
        
        I = imread(os.path.join('hw2-2_data/', img_name))
        I = np.reshape(I, -1)
        I = I.astype('float32')
        
        if int(img_name.split('_')[1].split('.')[0]) <= 7:      
            trainX.append(I)
            trainy.append(int(img_name.split('_')[0]))
            
        else:
            testX.append(I)
            testy.append(int(img_name.split('_')[0]))
            
    trainX = np.array(trainX)
    testX = np.array(testX)
    
    pca = PCA()
    lda = LDA()
    pca.fit(trainX)
    trainX_ = pca.transform(trainX, trainX.shape[0]-40)
    lda.fit(trainX_, trainy)
    trainX_ = lda.transform(trainX_, 39)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(trainX_, trainy) 

    testX_ = pca.transform(testX, trainX.shape[0]-40)
    testX_ = lda.transform(testX_, 39)
    print(knn.score(testX_, testy))