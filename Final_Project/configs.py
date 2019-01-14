# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:49:28 2018

@author: user
"""

class DefaultConfig():
    
    def __init__(self):
        
        self.env = 'CVFALL2018'
        self.port = 8888
        
        self.dataset = 'kitti2012/training/'
        self.max_disp = 128
        self.receptive_size = 11
        self.num_workers = 0
        
        self.n_epoch = 20000
        self.batch_size = 64
        self.lr = 0.01
        self.exp_decay_at_epoch = int(self.n_epoch*3/4)
        
        self.ckpts = 'ckpts/'
        self.testdata = 'data/Synthetic/'
        
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
            assert hasattr(self, k), 'Unknown attr '+ k +' !'
            
            setattr(self, k, v)
                
        self.print_config()
        
        return