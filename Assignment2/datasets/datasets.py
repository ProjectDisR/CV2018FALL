# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:04:00 2018

@author: user
"""

import os
import numpy as np

from torch.utils import data
import torchvision as tv

from skimage.io import imread

class MNISTDigits(data.Dataset):
    
    def __init__(self, root):
        
        self.img_ls = []
        self.label_ls = []
        
        digit_folder_ls = os.listdir(root)
        
        for digit_folder in digit_folder_ls:
            
            img_ls = os.listdir(os.path.join(root, digit_folder))
            img_ls = [os.path.join(root, digit_folder, img_name) for img_name in img_ls]
            
            self.img_ls += img_ls
            self.label_ls += [int(digit_folder.split('_')[1])]*len(img_ls)
            
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5], [0.5])
                ])
        
        return    
    
    def __getitem__(self, index):
        
        I = imread(self.img_ls[index])
        I = np.expand_dims(I, axis=2)
        I = self.transforms(I)
        
        return I, self.label_ls[index]
    
    def __len__(self):
        
        return len(self.img_ls)

class MNISTDigits1000(data.Dataset):
    
    def __init__(self, root):
        
        self.img_ls = []
        self.label_ls = []
        
        digit_folder_ls = os.listdir(root)
        
        for digit_folder in digit_folder_ls:
            
            img_ls = os.listdir(os.path.join(root, digit_folder))[:100]
            img_ls = [os.path.join(root, digit_folder, img_name) for img_name in img_ls]
            
            self.img_ls += img_ls
            self.label_ls += [int(digit_folder.split('_')[1])]*len(img_ls)
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5], [0.5])
                ])
        
        return    
    
    def __getitem__(self, index):
        
        I = imread(self.img_ls[index])
        I = np.expand_dims(I, axis=2)
        I = self.transforms(I)
        
        return I, self.label_ls[index]
    
    def __len__(self):
        
        return len(self.img_ls)
    

if __name__ == '__main__': 
   print('Preprocessing data or check whether My_Dataset works!')
#    root = 'hw2-3_data/train/'
#    dataset = MNISTDigits(root)
#    data_iter = iter(dataset)
#    I, label = next(data_iter)
