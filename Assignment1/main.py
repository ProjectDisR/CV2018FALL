# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:57:19 2018

@author: user
"""

import numpy as np
from skimage.io import imread, imsave

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
    if len(G.shape) == 2:
        G = np.pad(G, [(r, r), (r, r)], 'reflect')
    else:    
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

def rgb2gray_X(I):
    
    I = I / 255
    sigma = [(i, j) for i in [1, 2, 3] for j in [0.05, 0.1, 0.2]]
    local_minima = np.zeros((11, 11))
    
    for sigma_s, sigma_r in sigma:
        
        k = 6*sigma_s + 1
        
        I_JBFself = JBF(I, I, k, sigma_s, sigma_r)
        delta = np.full((13, 13), 5.0)
        
        for i in range(11):
            for j in range(11-i):
                
                I_gray = rgb2gray(I, 0.1*i, 0.1*j, 1 - 0.1*i - 0.1*j)
                I_JBFgray = JBF(I, I_gray, k, sigma_s, sigma_r)
                delta[i+1][j+1] = np.average(np.sum((I_JBFself-I_JBFgray)**2, axis=2))
                
        mid = delta[1:12, 1:12]
        is_local_minima = mid < delta[:11, :11]
        is_local_minima = np.logical_and(is_local_minima, mid < delta[:11, 1:12])
        is_local_minima = np.logical_and(is_local_minima, mid < delta[:11, 2:13])
        is_local_minima = np.logical_and(is_local_minima, mid < delta[1:12, :11])
        is_local_minima = np.logical_and(is_local_minima, mid < delta[1:12, 2:13])
        is_local_minima = np.logical_and(is_local_minima, mid < delta[2:13, :11])
        is_local_minima = np.logical_and(is_local_minima, mid < delta[2:13, 1:12])
        is_local_minima = np.logical_and(is_local_minima, mid < delta[2:13, 2:13])
        
        local_minima += is_local_minima
        
    wr, wg = np.unravel_index(np.argsort(local_minima.reshape(-1)), (11, 11))
    wr = wr[::-1]
    wg = wg[::-1]
    I = I * 255
    I_gray_x = []
    
    for i in range(3):
        print(0.1*wr[i], 0.1*wg[i], local_minima[wr[i], wg[i]])
        I_gray = rgb2gray(I, 0.1*wr[i], 0.1*wg[i], 1-0.1*wr[i]-0.1*wg[i])
        I_gray = I_gray.astype('uint8')
    
        I_gray_x.append(I_gray)
        
    return I_gray_x

if __name__ == '__main__':

    subs = ['a', 'b', 'c']
    
    for sub in subs:
        
        I = imread('testdata/1{}.png'.format(sub))
        I_gray = rgb2gray(I)
        I_gray = I_gray.astype('uint8')
        imsave('testdata/1{}_y.png'.format(sub), I_gray)
        
        I_gray_x = rgb2gray_X(I)
    
        for i in range(3):
            imsave('testdata/1{}_y{}.png'.format(sub ,i+1), I_gray_x[i])

    