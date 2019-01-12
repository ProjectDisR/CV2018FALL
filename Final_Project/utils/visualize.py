# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:09:16 2018

@author: user
"""

import numpy as np

import visdom

class Visualizer():
    
    def __init__(self, env='env_for_visdom', port=8888):
        
        self.vis = visdom.Visdom(env=env, port=port)
        self.wins = set()
        self.log_text = ''
        
        return
    
    def add_wins(self, *args):
        
        for win in args:
            self.wins.add(win)
            
        return
    
    def plot(self, win, epoch, value, name=None):
        assert win in self.wins, 'Unknown win {} for plotting! Use add_wins to add a new window.'.format(win)
            
        opts = {'xlabel':'epoch', 'ylabel':win, 'showlegend':True}
        if name is None:
            name = win
            
        self.vis.line(Y=np.array([value]), X=np.array([epoch]), win=win,
                          opts=opts, update='append', name=name)
            
        return
    
    def imgs(self, win, I):
        
        self.vis.images(I, nrow=4, win=win, opts={'title':win})
        
        return
        
    def log(self, info, win='log'):
        
        self.log_text += '{} <br>'.format(info)
        self.vis.text(self.log_text, win, opts={'title':win})
        
        return