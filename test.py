# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:37:10 2018

@author: user
"""

import numpy as np
from skimage.io import imread

I = imread('testdata/0a.png')
I = np.pad(I, [(2, 2), (2, 2), (0, 0)], 'edge')
b = I.shape
#b = type(I)
w = I.shape
a = np.array([[[1, 2, 5]]])


c = np.array([5.0])
c = c.astype('int')