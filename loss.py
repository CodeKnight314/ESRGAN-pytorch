import torch.nn as nn 
from torchvision.models import vgg19
from torch.nn import functional as F

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(weights='DEFAULT').features[:35].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        sr_vgg = self.vgg(sr)
        hr_vgg = self.vgg(hr)
        loss = self.criterion(sr_vgg, hr_vgg)
        return loss
