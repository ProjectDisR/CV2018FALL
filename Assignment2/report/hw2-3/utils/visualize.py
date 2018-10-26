# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:09:16 2018

@author: user
"""

import visdom

import numpy as np


class Visualizer():
    
    def __init__(self, env='hw2'):
        
        self.vis = visdom.Visdom(env=env)
        self.names = set()
        self.log_text = ''
        
        return
    
    def add_names(self, *args):
        
        for name in args:
            self.names.add(name)
            
        return
    
    def plot(self, name, epoch, value):
        
        if not name in self.names:
            raise Exception('Unknown name for plotting! Use add_names to add a new name.')
            
        else:
            opts = {'xlabel':'epoch', 'ylabel':name} 
            self.vis.line(Y=np.array([value]), X=np.array([epoch]), win=name,
                          opts=opts, update='append')
            
        return
    
    def imgs(self, win, I):
        
        self.vis.images(I, win=win, opts={'title':win})
        
        return
        
    def log(self, info, win='log'):
        
        self.log_text += '{} <br>'.format(info)
        self.vis.text(self.log_text, win, opts={'title':win})
        
        return
    