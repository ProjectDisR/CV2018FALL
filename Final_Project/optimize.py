from numpy import matlib
import math
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter
from numpy import matlib
import math

def fillPixelsReference(Il, final_labels, gamma_c, gamma_d, r_median, max_disp):
    h,w = final_labels.shape
    occPix = np.zeros((h, w))
    occPix[final_labels<0] = 1

    # Streak-based filling of invalidated pixels from left
    fillVals = np.ones((h, 1))*max_disp
    final_labels_filled = final_labels.copy()
    for col in range(w) : 
        curCol = final_labels[:,col]
        curCol = np.array([curCol])
        curlTmp = curCol
        curCol = curCol.T
        curlTmp = curlTmp.T
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curlTmp[curlTmp!=-1]
        final_labels_filled[:,col] = curCol.flatten()

    # Streak-based filling of invalidated pixels from right
    fillVals = np.ones((h,1))*max_disp
    final_labels_filled1 = final_labels.copy()
    for col in range(w) : 
        col = (w - 1) - col
        curCol = final_labels[:,col]
        curCol = np.array([curCol])
        curlTmp = curCol
        curCol = curCol.T
        curlTmp = curlTmp.T
        curCol[curCol==-1] = fillVals[curCol==-1]
        fillVals[curCol!=-1] = curlTmp[curlTmp!=-1]
        final_labels_filled1[:,col] = curCol.flatten()


    final_labels = np.minimum(final_labels_filled,final_labels_filled1)

    # Weighted median filtering on occluded regions (post processing) NOT DONE YET
    #final_labels_smoothed = weightedMedian(Il,final_labels,r_median,gamma_c,gamma_d);
    #final_labels[occPix==1] = final_labels_smoothed[occPix==1]
    return final_labels

def Optimize(Il,labels_left,labels_right):

    h,w,ch = Il.shape
    # Parameter settings
    r = 9                  # filter kernel in eq. (3) has size r \times r
    eps = 0.0001           # \epsilon in eq. (3)
    thresColor = 7/255     # \tau_1 in eq. (5)
    thresGrad = 2/255      # \tau_2 in eq. (5)
    gamma = 0.11           # (1- \alpha) in eq. (5)
    threshBorder = 3/255   # some threshold for border pixels
    gamma_c = 0.1          # \sigma_c in eq. (6)
    gamma_d = 9            # \sigma_s in eq. (6)
    r_median = 19          # filter kernel of weighted median in eq. (6) has size r_median \times r_median
    max_disp = 64



    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering

    # Left-right consistency check
    y1 = np.arange(1,h+1)
    y1 = np.array([y1])
    Y = np.matlib.repmat(y1.T,1,w)
    x1 = np.arange(1,w+1)
    X = np.matlib.repmat(x1,h,1)- labels_left
    X[X<1] = 1

    final_labels = labels_left
    
    for i in range(h*w):
        col = i // h
        row = i % h
        temp_num = np.absolute(labels_left[row][col] - labels_right[Y[row][col]-1][X[row][col]-1])
        if temp_num >= 1:
            final_labels[row][col] = -1
    # Fill and filter (post-process) pixels that fail the consistency check
    inputLabels = final_labels
    final_labels = fillPixelsReference(Il, inputLabels, gamma_c, gamma_d, r_median, max_disp);


    labels = final_labels
    return labels