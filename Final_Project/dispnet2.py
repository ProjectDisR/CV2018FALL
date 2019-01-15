# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:25:09 2018

@author: user
"""
import os
import time
import cv2

import numpy as np

import torch as t
from torch.utils.data import DataLoader
from torch import nn

from configs import DefaultConfig
from datasets.datasets import get_loaders
from models.model import Model
from utils.visualize import Visualizer
from utils.meters import AverageMeter

from main import computeDisp
from err import ERR
from optimize import Optimize
from util import writePFM

import fire

def hisEqulColor(img):
    
	ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(ycrcb)
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, ycrcb)
	cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    
	return img


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
    vis.add_wins('loss', 'accuracy', 'err')
    
    if not os.path.exists(opt.ckpts):
        os.makedirs(opt.ckpts)
    
    t.save(model.state_dict(), os.path.join(opt.ckpts, 'best.ckpt'))
    besterr = 100.0
    for epoch in range(opt.n_epoch):
        print('epoch{} '.format(epoch), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
        adjust_lr_exp(optimizer, opt.lr, epoch + 1, opt.n_epoch, opt.exp_decay_at_epoch)
        
        loss_meter = AverageMeter()
        
        model.train()
       
        for i, (left, right, d) in enumerate(dataloaders['train']):
            
            left = left.cuda()
            right = right.cuda()
            d = d.cuda()
            
            soft_label = t.zeros([1, opt.max_disp+1])
            soft_label[0, 0:3] = t.tensor([0.5, 0.4, 0.1])
            soft_label = soft_label.repeat(d.shape[0], 1).cuda()

            output = model(left, right, train=True)
            loss = -(output*soft_label).sum(dim=1)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), d.shape[0])
        
        if epoch % 50 == 1:
            model.eval()
            
            for i in range(10):
                print(i)
                
                img_left = cv2.imread(os.path.join(opt.testdata, 'TL{}.png'.format(i)))
                img_right = cv2.imread(os.path.join(opt.testdata, 'TR{}.png'.format(i)))
                img_left = hisEqulColor(img_left)
                img_right = hisEqulColor(img_right)
                # Mirror image
                img_left_1 = np.fliplr(img_right)
                img_right_1 = np.fliplr(img_left)
                
                disp = computeDisp(img_left, img_right)              
                disp_1 = computeDisp(img_left_1, img_right_1)
                disp_1 = np.fliplr(disp_1)
                
                disp = np.uint8(disp)
                disp_1 = np.uint8(disp_1)

                disp = Optimize(img_left,disp,disp_1)
                disp = cv2.medianBlur(np.uint8(disp), 11)
                disp = disp.astype(np.float32)
                
                writePFM(os.path.join(opt.testdata, 'TL{}.pfm'.format(i)), disp)
                
            err = ERR(opt.testdata)
            if err < besterr:
                t.save(model.state_dict(), os.path.join(opt.ckpts, 'best.ckpt'))
                
            vis.plot('err', epoch, err)
        
        vis.plot('loss', epoch, loss_meter.avg)
        
    return


def help():
        
    opt = DefaultConfig()
    opt.print_config()
    
    return

if __name__ == '__main__':
    fire.Fire()