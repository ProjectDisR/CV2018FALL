import cv2

import torch as t
from torch import nn
import torchvision as tv
from torch.nn import functional as F

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
#            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
#            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
#            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3)
            )
        
        return

    def forward(self, left, right, train):
        
        left = self.cnn(left)    
        right = self.cnn(right)
        
        if train:
            
            left = left.squeeze()
            left = left.view(left.shape[0], 1, -1)
            
            right = right.squeeze()
        
            out = t.bmm(left, right)
            out = out.squeeze()
            out = F.log_softmax(out, dim=1)
        
            return out
        
        return left, right

if __name__ == '__main__':
    
    left = t.randn((2, 3, 9, 9))
    right = t.randn((2, 3, 9, 136))
    
    model = Model()

    out = model(left, right)[2]
    print('out:', out.shape)
    
    model = Model2()
    right = cv2.imread('data/Synthetic/TL0.png')
    left = cv2.imread('data/Synthetic/TR0.png')
    transforms = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    left, right = model(transforms(left).unsqueeze(0), transforms(right).unsqueeze(0))
    