# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:50:49 2018

@author: user
"""
import os

from skimage.io import imread, imsave 
   
    n = 0
    root = 'hw2-3_data/train/'
        
    digit_folder_ls = os.listdir(root)
        
    for digit_folder in digit_folder_ls:
            
        img_ls = os.listdir(digit_folder)
        for i in range(1000):
            I = imread(os.path.join(root, digit_folder, img_ls[i]))
            imsave('{}_{}.png'.format(str(n).zfill(5), digit_folder.split('_')[1]))
            n += 1