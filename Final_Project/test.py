# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:00:24 2019

@author: user
"""

import os
import cv2
import numpy as np

from main import computeDisp
from optimize import Optimize
from err import ERR

from util import writePFM

def hisEqulColor(img):
    
	ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(ycrcb)
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, ycrcb)
	cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    
	return img

def test():
    for i in range(10):
        print(i)
        
        img_left = cv2.imread(os.path.join('data/Synthetic/', 'TL{}.png'.format(i)))
        img_right = cv2.imread(os.path.join('data/Synthetic/', 'TR{}.png'.format(i)))
        img_left = hisEqulColor(img_left)
        img_right = hisEqulColor(img_right)
        # Mirror image
        img_left_1 = np.fliplr(img_right)
        img_right_1 = np.fliplr(img_left)
        
        disp = computeDisp(img_left, img_right)              
        disp_1 = computeDisp(img_left_1, img_right_1)
        disp_1 = np.fliplr(disp_1)
        
        disp = np.int32(disp)
        disp_1 = np.int32(disp_1)

        disp = Optimize(img_left,disp,disp_1)
        disp = cv2.medianBlur(np.uint8(disp), 3)
        disp = disp.astype(np.float32)
        
        writePFM(os.path.join('data/Synthetic/', 'TL{}.pfm'.format(i)), disp)
        
    err = ERR('data/Synthetic/')
    
    return err

test()