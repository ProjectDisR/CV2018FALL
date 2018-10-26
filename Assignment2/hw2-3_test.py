# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 18:05:36 2018

@author: user
"""
import os
import sys

import numpy as np

import torch as t
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision as tv

from skimage.io import imread

from models.lenet5 import Lenet5



class DefaultConfig():
    
    def __init__(self):
        
        self.data_root = 'datasets/hw2-3_data/'
        self.ckpts_root = 'checkpoints/'

        self.batch_size = 1000
        
        return
    
    def print_config(self):
        
        print('\n')
        
        import inspect
        
        for k in dir(self):   
            if not k.startswith('__') and not inspect.ismethod(getattr(self, k)):
                print('   ', k, ':', getattr(self, k))
                
        return
    
    def parse(self, kwargs):
        
        for k, v in kwargs.items():
            
            if not hasattr(self, k):
                raise Exception('Unknown attr '+ k +' !')
            else:
                setattr(self, k, v)
                
        self.print_config()
        
        return

class TestData(data.Dataset):
    
    def __init__(self, root):
        
        self.root = root
        self.img_name_ls = os.listdir(self.root)
        self.img_name_ls.sort()
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5], [0.5])
                ])
        
        return    
    
    def __getitem__(self, index):
        
        I = imread(os.path.join(self.root, self.img_name_ls[index]))
        I = np.expand_dims(I, axis=2)
        I = self.transforms(I)
        
        return I, self.img_name_ls[index].split('.')[0]
    
    def __len__(self):
        
        return len(self.img_name_ls)
    

def test(csv, **kwargs):
    
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    test_dataset = TestData(opt.data_root)
    test_dataloader = DataLoader(test_dataset, opt.batch_size, shuffle=False)
    
    lenet5 = Lenet5()
    lenet5.load_state_dict(t.load(os.path.join(opt.ckpts_root)))
    lenet5 = lenet5.eval()
    
    label_ls = []
    id_ls = []
    for I, img_name in test_dataloader:         
        predicts = t.argmax(lenet5(I)[0], dim=1)
        label_ls += list(predicts.detach().numpy())
        id_ls += img_name
    
    with open(csv, 'w') as outfile:
        outfile.write('id, label\n')
        for i in range (len(id_ls)):
            outfile.write('{},{}\n'.format(id_ls[i], label_ls[i]))
    
    return

if __name__ == '__main__':
    test(csv=sys.argv[2], data_root=sys.argv[1], ckpts_root='lenet5_e50.ckpt')