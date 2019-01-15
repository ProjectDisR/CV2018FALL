import numpy as np
import argparse
import cv2
import time
import torch
import torchvision as tv

from models.model import Model
from optimize import Optimize

from util import writePFM

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')

def hisEqulColor(img):
    
	ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(ycrcb)
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, ycrcb)
	cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    
	return img

# You can modify the function interface as you like
def computeDisp(Il, Ir):
    h, w, ch = Il.shape
    model = Model()
    model.cuda().eval()
    model.load_state_dict(torch.load('ckpts/best.ckpt'))
    Il = (cv2.copyMakeBorder(Il,5,5,5,5,cv2.BORDER_REFLECT)).astype(np.uint8)
    Ir = (cv2.copyMakeBorder(Ir,5,5,5,5,cv2.BORDER_REFLECT)).astype(np.uint8)
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    Il = transforms(Il).unsqueeze(0).cuda()
    Ir = transforms(Ir).unsqueeze(0).cuda()
    featureL, featureR = model(Il,Ir, train=False)
    featureL = featureL.squeeze().data.cpu().numpy()
    featureR = featureR.squeeze().data.cpu().numpy()
    disp = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            curDisp = 0
            MaxCos = -np.inf
            for d in range(min(j,64)):
                cosine = np.sum(featureL[:,i,j] * featureR[:,i,j-d])
                if cosine > MaxCos:
                    curDisp = d
                    MaxCos = cosine
            disp[i,j] = curDisp   
    
    return disp

def Disparity(img_left, img_right):
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
    disp = cv2.medianBlur(np.uint8(disp), 11)
    disp = disp.astype(np.float32)
    
    return disp

def main():
    args = parser.parse_args()
    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = Disparity(img_left, img_right)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
