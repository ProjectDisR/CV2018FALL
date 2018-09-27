# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:37:10 2018

@author: user
"""
import numpy as np
a = np.array([[3, 2, 6], [4, 5, 6], [8, 10, 2]])
wr, wg = np.unravel_index(np.argsort(a.reshape(-1)), (3, 3))