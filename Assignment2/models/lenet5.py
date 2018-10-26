# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 14:22:37 2018

@author: user
"""
import torch as t
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    
    def __init__(self):
        super(Lenet5, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        return
        
    def forward(self, x):
        
        map1 = F.relu(self.conv1(x))
        max1 = F.max_pool2d(map1, (2, 2))
        
        map2 = F.relu(self.conv2(max1))
        max2 = F.max_pool2d(map2, (2, 2))
        
        max2 = max2.view(max2.size()[0], -1)
        
        max2 = F.relu(self.fc1(max2))
        max2 = F.relu(self.fc2(max2))
        max2 = self.fc3(max2)
        
        return max2, map1, map2
    
if __name__ == '__main__':
    print('Check if My_Moel works!')
    print(Lenet5())

