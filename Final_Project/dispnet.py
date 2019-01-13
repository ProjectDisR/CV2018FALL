# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:25:09 2018

@author: user
"""
import os
import time

import numpy as np

import torch as t
from torch.utils.data import DataLoader
from torch import nn

from configs import DefaultConfig
from datasets.datasets import get_loaders
from models.model import Model
from utils.visualize import Visualizer
from utils.meters import AverageMeter

from skimage.io import imsave

import fire


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep) / (total_ep + 1 - start_decay_at_ep))))
    
    return


def train(**kwargs):
    
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    data_root = os.path.join('datasets/', opt.dataset)
    
    dataloaders = get_loaders(data_root, opt.receptive_size, opt.max_disp, opt.batch_size, opt.num_workers)
       
    model = Model()
    model = model.cuda()
    
    optimizer = t.optim.Adagrad(model.parameters(), opt.lr)
    
    vis = Visualizer(opt.env, opt.port)
    vis.add_wins('loss', 'accuracy')
    
    if not os.path.exists(opt.ckpts):
        os.makedirs(opt.ckpts)
    
    for epoch in range(opt.n_epoch):
        print('epoch{} '.format(epoch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        adjust_lr_exp(optimizer, opt.lr, epoch + 1, opt.n_epoch, opt.exp_decay_at_epoch)
        
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        model.train()
       
        for i, (left, right, d) in enumerate(dataloaders['train']):
            
            left = left.cuda()
            right = right.cuda()
            d = d.cuda()
            
            soft_label = t.zeros([1, opt.max_disp+1])
            soft_label[0, 0:4] = t.tensor([0.5, 0.3, 0.15, 0.05])
            soft_label = soft_label.repeat(d.shape[0], 1).cuda()
            output = model(left, right, train=True)
            loss = -(output*soft_label).sum(dim=1)
            print(output[0:10][0:10])
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), d.shape[0])
            
        t.save(model.state_dict(), os.path.join(opt.ckpts, '{}.ckpt'.format(epoch)))
        
        model.eval()
        
        for i, (left, right, d) in enumerate(dataloaders['valid']):
            
            left = left.cuda()
            right = right.cuda()
            d = d.cuda()
            
            output = model(left, right, train=True)
            d_ = t.argmax(output, dim=1)
            
            accuracy_meter.update((d_ == d).sum().item()/d.shape[0], d.shape[0])
        
        vis.plot('loss', epoch, loss_meter.avg)
        vis.plot('accuracy', epoch, accuracy_meter.avg)
        vis.log('epoch:{}, loss:{}, accuracy:{}'.format(epoch, loss_meter.avg, accuracy_meter.avg))
        
    return


def test(**kwargs):
            
    return


def help():
    print("""
    usage : python {0} <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test
            python {0} help
    avaiable args:\n""".format(__file__))
        
    opt = DefaultConfig()
    opt.print_config()
    
    return

if __name__ == '__main__':
    fire.Fire()