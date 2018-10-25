# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:12:48 2018

@author: user
"""
import os
import numpy as np
from skimage.io import imread, imsave


if __name__ == '__main__':
    np.random.seed(8)
    
    data_dir = 'hw2-2_data/'
    save_dir = 'report/hw2-2/'
    
    if not os.path.exists(os.path.join(save_dir, 'train/', '0/')):
        os.makedirs(os.path.join(save_dir, 'train/', '0/'))        
    if not os.path.exists(os.path.join(save_dir, 'train/', '1/')):
        os.makedirs(os.path.join(save_dir, 'train/', '1/'))
    if not os.path.exists(os.path.join(save_dir, 'train/', '2/')):
        os.makedirs(os.path.join(save_dir, 'train/', '2/'))
        
    for sub in range(40):
        sub = sub + 1
        i_ls = np.random.permutation(7)
        i_ls += 1
        
        for i in i_ls[:2]:
            I = imread(os.path.join(data_dir, '{}_{}.png'.format(sub, i)))
            imsave(os.path.join(save_dir, 'train/', '0/', '{}_{}.png'.format(sub, i)), I)
        for i in i_ls[2:4]:
            I = imread(os.path.join(data_dir, '{}_{}.png'.format(sub, i)))
            imsave(os.path.join(save_dir, 'train/', '1/', '{}_{}.png'.format(sub, i)), I)
        for i in i_ls[4:]:
            I = imread(os.path.join(data_dir, '{}_{}.png'.format(sub, i)))
            imsave(os.path.join(save_dir, 'train/', '2/', '{}_{}.png'.format(sub, i)), I)