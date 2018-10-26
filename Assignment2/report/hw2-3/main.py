# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:25:09 2018

@author: user
"""
import os
import time

from sklearn.manifold import TSNE

import torch as t
from torch.utils.data import DataLoader
from torch import nn

from config import DefaultConfig
from datasets.datasets import MNISTDigits, MNISTDigits1000
from models.lenet5 import Lenet5
from utils.visualize import Visualizer

import fire

def train(**kwargs):
    
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    train_dataset = MNISTDigits(os.path.join(opt.data_root, 'train/'))
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    valid_dataset = MNISTDigits(os.path.join(opt.data_root, 'valid/'))
    valid_dataloader = DataLoader(valid_dataset, opt.batch_size, shuffle=False)
       
    lenet5 = Lenet5() 
    criterion = nn.CrossEntropyLoss() 
    optimizer = t.optim.Adam(lenet5.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    vis = Visualizer(opt.env)
    vis.add_names('loss', 'accuracy')
    
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
        
        t.save(lenet5.state_dict(), os.path.join(opt.ckpts_root, 'lenet5_e{}.ckpt'.format(epoch+1)))
        
        vis.plot('loss', epoch, avgloss/i)
        
        lenet5 = lenet5.eval()
        n_valids = 0
        n_yes = 0
        
        for I, labels in valid_dataloader:
            
            n_valids += labels.size()[0]
            predicts = t.argmax(lenet5(I)[0], dim=1)
            n_yes += t.sum(predicts==labels).item()
            
        vis.plot('accuracy', epoch, n_yes/n_valids)
        vis.log('epoch:{}, loss:{}, accuracy:{}'.format(epoch, avgloss/i, n_yes/n_valids))

        lenet5 = lenet5.train()
        
    return

def FilterVisualization(ckpt_root):
    vis = Visualizer('hw2')
    
    imgs1 = t.rand(6, 1, 28, 28)
    imgs1 = (imgs1 - 0.5) / 0.5
    imgs1.requires_grad_()
    
    imgs2 = t.rand(16, 1, 28, 28)
    imgs2 = (imgs2 - 0.5) / 0.5
    imgs2.requires_grad_()
    
    lenet5 = Lenet5()
    lenet5.load_state_dict(t.load(ckpt_root))
    lenet5 = lenet5.eval()
    
    optimizer = t.optim.Adam([imgs1, imgs2], lr=0.1, betas=(0.5, 0.999))
    
    for epoch in range(1000):
        print('epoch{} '.format(epoch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        map1 = lenet5(imgs1)[1]
        map2 = lenet5(imgs2)[2]
        
        loss = 0
        
        for i in range(6):
            loss = loss + t.sum(t.abs(map1[i, i, :, :]))
        for i in range(16):
            loss = loss + t.sum(t.abs(map2[i, i, :, :]))
        loss = -loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        vis.imgs('conv1filter', imgs1*0.5 + 0.5)
        vis.imgs('conv2filter', imgs2*0.5 + 0.5)
        
    return

def FeatureVisualization(ckpt_root):
    vis = Visualizer('hw2')
    
    valid_dataset = MNISTDigits1000('datasets/hw2-3_data/' + 'valid/')
    valid_dataloader = DataLoader(valid_dataset, 1000, shuffle=False)
    
    lenet5 = Lenet5()
    lenet5.load_state_dict(t.load(ckpt_root))
    lenet5 = lenet5.eval()
    
    for I, labels in valid_dataloader:
        
        labels = labels.numpy()
        labels = labels + 1
        labels = labels.astype('int')
        _, map1, map2 = lenet5(I)
        map1 = map1.view(map1.size()[0], -1).detach().numpy()
        map2 = map2.view(map2.size()[0], -1).detach().numpy()
        
        vis.vis.scatter(TSNE().fit_transform(map1), labels, win='conv1feature', opts={'title': 'conv1 feature'})
        vis.vis.scatter(TSNE().fit_transform(map2), labels, win='conv2feature', opts={'title': 'conv2 feature'})
    
    return

def test(**kwargs):
    
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    train_dataset = MNISTDigits(os.path.join(opt.data_root, 'train/'))
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=False)
    valid_dataset = MNISTDigits(os.path.join(opt.data_root, 'valid/'))
    valid_dataloader = DataLoader(valid_dataset, opt.batch_size, shuffle=False)
    
    lenet5 = Lenet5()
    lenet5.load_state_dict(t.load(os.path.join(opt.ckpts_root, 'lenet5_e{}.ckpt'.format(opt.n_epoch))))
    lenet5 = lenet5.eval()
    
    n_trains = 0
    n_yes = 0
    
    for I, labels in train_dataloader:
            
        n_trains += labels.size()[0]
        predicts = t.argmax(lenet5(I)[0], dim=1)
        n_yes += t.sum(predicts==labels).item()
        
    print('train accuracy: {}'.format(n_yes/n_trains))
    
    n_valids = 0
    n_yes = 0
    for I, labels in valid_dataloader:
            
        n_valids += labels.size()[0]
        predicts = t.argmax(lenet5(I)[0], dim=1)
        n_yes += t.sum(predicts==labels).item()
        
    print('valid accuracy: {}'.format(n_yes/n_valids))
    
    return

if __name__ == '__main__':
    fire.Fire()