# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:37:10 2018

@author: user
"""

import numpy as np
from skimage.io import imread
def RangeKernel(K, sigma):
    
    K = K / 255
    k = K.shape[0]
    center = int((k-1) / 2)
    
    if len(K.shape) == 2:
        
        center = K[center, center]
        K = K - center   
        K = K**2
        K = -K
        K = K / (2 * sigma**2)
        K = np.exp(K)
    
    else:
        
        center = K[center:center+1, center:center+1]
        center = np.repeat(center, k, 1)
        center = np.repeat(center, k, 0)
        K = K - center
        K = K**2
        K = np.sum(K, axis=2)
        K = -K
        K = K / (2 * sigma**2)
        K = np.exp(K)

        
    K = np.expand_dims(K, axis=2)
    K = np.repeat(K, 3, axis=2)
    
    return K
K = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
K = RangeKernel(255*K, 0.1)
I = imread('testdata/0a.png')
I = np.pad(I, [(2, 2), (2, 2), (0, 0)], 'edge')
b = I.shape
#b = type(I)
w = I.shape
a = np.array([[[1, 2, 5]]])


c = np.array([5.0])
c = c.astype('int')