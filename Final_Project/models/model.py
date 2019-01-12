import torch as t
from torch import nn

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3)
            )
        
        return

    def forward(self, left, right):
        
        left = self.cnn(left)
        left = left.squeeze()
        left = left.view(left.shape[0], 1, -1)
        
        right = self.cnn(right)
        right = right.squeeze()
        
        out = t.bmm(left, right)
        out = out.squeeze()
        
        return out
    

if __name__ == '__main__':
    
    left = t.randn((2, 1, 9, 9))
    right = t.randn((2, 1, 9, 136))
    
    model = Model()

    out = model.forward(left, right)
    print('out:', out.shape)