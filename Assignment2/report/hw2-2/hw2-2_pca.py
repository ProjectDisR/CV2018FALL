# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:34:49 2018

@author: user
"""
import os
import itertools
import visdom

import numpy as np
from scipy import linalg as LA
from skimage.io import imread, imsave
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

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
    vis = visdom.Visdom(env='pca')
    
    h = 56
    w = 46
    
    trainX = []
    trainy = []
    testX = []
    testy = []
    img_name_ls = os.listdir('hw2-2_data')
    
    for img_name in img_name_ls:
        I = imread(os.path.join('hw2-2_data', img_name))
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
    pca.fit(trainX)
    
    mean = np.reshape(pca.mean, (h, w))
    mean = mean.astype('uint8')
    imsave('mean.png', mean)
    for i in range(5):
        eigen = pca.T[:, i]
        eigen = np.reshape(eigen, (h, w))
        eigen = (eigen-np.min(eigen))*255 / (np.max(eigen)-np.min(eigen))
        eigen = eigen.astype('uint8')
        imsave('eigen{}.png'.format(i+1), eigen)
    
    
    
    
    
    
    
    
    
    
    I = imread(os.path.join('hw2-2_data', '8_6.png'))
    I = np.reshape(I, (1, -1))
    I = I.astype('float32')
    for i in [5, 50, 150, 279]:
        I_recons = pca.reconstruct(pca.transform(I, i))
        imsave('8_6_{}.png'.format(i), np.reshape(I_recons, (h, w)).astype('uint8'))
        print(i, np.mean((I-I_recons)**2))








    testX_ = pca.transform(testX, 100)
    vis.scatter(TSNE().fit_transform(testX_), testy, name='tsne')
    
    
    
    
    
    
    
    folder_ls = ['0/', '1/', '2/']
    kfold = KFold(n_splits=3)
    k_ls = [1, 3, 5]
    n_ls = [3 ,50, 100]
    
    for k, n in itertools.product(k_ls, n_ls):
        accuracy = []
        
        for train_index_ls, valid_index in kfold.split(folder_ls):
            
            trainX = []
            trainy = []
            validX = []
            validy = []
            
            for i in train_index_ls:
                img_name_ls = os.listdir(os.path.join('train', folder_ls[i]))
                for img_name in img_name_ls:
                    I = imread(os.path.join('train', folder_ls[i], img_name))
                    I = np.reshape(I, -1)
                    I = I.astype('float32')
                    
                    trainX.append(I)
                    trainy.append(int(img_name.split('_')[0]))
              
            img_name_ls = os.listdir(os.path.join('train', folder_ls[valid_index[0]]))
            for img_name in img_name_ls:
                I = imread(os.path.join('train', folder_ls[valid_index[0]], img_name))
                I = np.reshape(I, -1)
                I = I.astype('float32')
                
                validX.append(I)
                validy.append(int(img_name.split('_')[0]))
            
            trainX = np.array(trainX)
            validX = np.array(validX)
            
            pca = PCA()
            pca.fit(trainX)
            trainX_ = pca.transform(trainX, n)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(trainX_, trainy) 

            validX_ = pca.transform(validX, n)
            accuracy.append(knn.score(validX_, validy))
        accuracy.append(np.mean(np.array(accuracy)))
        print(k, n, accuracy[0], accuracy[1], accuracy[2], accuracy[3])
    