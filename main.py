# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:57:19 2018

@author: user
"""

import numpy as np
from skimage.io import imread, imshow, imsave
import skimage
import cv2

def rgb2gray(I, wr=0.299, wg=0.587, wb=0.114):
    
    R = I[:, :, 0]
    G = I[:, :, 1]
    B = I[:, :, 2]
    
    I_gray = wr*R + wg*G + wb*B
    
    return I_gray

def GaussainKernel(k, sigma):
    
    if k % 2 == 0:
        raise Exception('k must be an odd number')

    K = np.indices((k, k))
    
    center = (k-1) / 2
    center = np.array([[[center]], [[center]]])
    center = np.repeat(center, k, 1)
    center = np.repeat(center, k, 2)
    
    K = K - center
    K = K**2
    K = np.sum(K, axis=0)
    K = -K
    K = K / (2 * sigma**2)
    K = np.exp(K)
    
    K = np.expand_dims(K, axis=2)
    K = np.repeat(K, 3, axis=2)
    
    return K

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

def JBF(I, G, k, sigma_s, sigma_r):
    
    H, W, C = I.shape
    I_filtered = np.zeros(I.shape)
    
    r = int((k-1) / 2)
    I = np.pad(I, [(r, r), (r, r), (0, 0)], 'reflect')
    G = np.pad(G, [(r, r), (r, r), (0, 0)], 'reflect')

    Ks = GaussainKernel(k, sigma_s)
    
    for h in range(H):
        for w in range(W):
            
            h_ = h + r
            w_ = w + r
            
            Kr = RangeKernel(G[h_-r:h_+r+1, w_-r:w_+r+1], sigma_r)
            K = Ks * Kr        
            sum_ = np.sum(K, axis=1)
            sum_ = np.sum(sum_, axis=0)
                   
            rgb = I[h_-r:h_+r+1, w_-r:w_+r+1] * K
            rgb = np.sum(rgb, axis=1)
            rgb = np.sum(rgb, axis=0)
            rgb = rgb / sum_
            
            I_filtered[h][w]  = rgb
        
    return I_filtered

I = imread('I.png')
G = imread('G.png')

I = skimage.color.rgba2rgb(I)*255
G = skimage.color.rgba2rgb(G)*255
I_filtered = JBF(I, G, 3, 8, 0.2)
for i in range(4):
    I_filtered = JBF(I_filtered, G, 3, 8, 0.2)
I_filtered = I_filtered.astype('uint8')
imsave('I_filtered.png', I_filtered)

    