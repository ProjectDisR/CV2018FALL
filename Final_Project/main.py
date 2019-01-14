import numpy as np
import argparse
import cv2
import time
import torch
import torchvision as tv

from models.model import Model

from util import writePFM

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


# You can modify the function interface as you like
def computeDisp(Il, Ir):
    h, w, ch = Il.shape
    model = Model()
    model.cuda().eval()
    model.load_state_dict(torch.load('ckpts/300.ckpt'))
    Il = (cv2.copyMakeBorder(Il,4,4,4,4,cv2.BORDER_REFLECT)).astype(np.uint8)
    Ir = (cv2.copyMakeBorder(Ir,4,4,4,4,cv2.BORDER_REFLECT)).astype(np.uint8)
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    Il = transforms(Il).unsqueeze(0).cuda()
    Ir = transforms(Ir).unsqueeze(0).cuda()
    featureL, featureR = model(Il,Ir, train=False)
    featureL = featureL.squeeze().data.cpu().numpy
    featureR = featureR.squeeze().data.cpu().numpy
    disp = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            curDisp = 0
            MaxCos = -np.inf
            for d in range(min(i,64)):
                cosine = featureL[:,h,w] * featureR[:,h,w-d]
                if cosine > MaxCos:
                    curDisp = d
                    MaxCos = cosine
            disp[i,j] = curDisp
    return disp


def main():
    args = parser.parse_args()
    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(img_left, img_right)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
