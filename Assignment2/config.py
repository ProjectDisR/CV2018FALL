# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:49:28 2018

@author: user
"""

class DefaultConfig():
    
    def __init__(self):
        
        self.data_root = 'datasets/hw2-3_data/'
        self.ckpts_root = 'checkpoints/'
        
        self.n_epoch = 50
        self.batch_size = 1000
        self.lr = 0.001
        
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
    
if __name__ == '__main__':
    opt = DefaultConfig()
    opt.parse({'n_epoch':5})