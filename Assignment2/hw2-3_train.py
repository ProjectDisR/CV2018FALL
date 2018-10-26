# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:25:09 2018

@author: user
"""
import os
import sys
import time

import torch as t
from torch.utils.data import DataLoader
from torch import nn

from config import DefaultConfig
from datasets.datasets import MNISTDigits
from models.lenet5 import Lenet5

def train(**kwargs):
    
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    train_dataset = MNISTDigits(os.path.join(opt.data_root, 'train/'))
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    valid_dataset = MNISTDigits(os.path.join(opt.data_root, 'valid/'))
    valid_dataloader = DataLoader(valid_dataset, opt.batch_size, shuffle=True)
    
    lenet5 = Lenet5() 
    criterion = nn.CrossEntropyLoss() 
    optimizer = t.optim.Adam(lenet5.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    if not os.path.isdir(opt.ckpts_root):
        os.mkdir(opt.ckpts_root)
    
    for epoch in range(opt.n_epoch):
        
        print('epoch{} '.format(epoch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        avgloss = 0       
        
        for i, (I, labels) in enumerate(train_dataloader):
            
            loss = criterion(lenet5(I)[0], labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avgloss += loss.item()
        
        for i, (I, labels) in enumerate(valid_dataloader):
            
            loss = criterion(lenet5(I)[0], labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avgloss += loss.item()
        
        t.save(lenet5.state_dict(), os.path.join(opt.ckpts_root, 'lenet5_e{}.ckpt'.format(epoch+1)))
        
    return

if __name__ == '__main__':
    train(data_root=sys.argv[1])