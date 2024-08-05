import torch 
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
    
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                         dtype=torch.float32, 
                                         requires_grad=False).view(1, 1, 3, 3)
        
        self.edge_sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                         dtype=torch.float32, 
                                         requires_grad=False).view(1, 1, 3, 3)

    def forward(self, predicted, target):
        edge_pred_x = F.conv2d(predicted, self.edge_sobel_x, padding=1)
        edge_pred_y = F.conv2d(predicted, self.edge_sobel_y, padding=1)
        edge_target_x = F.conv2d(target, self.edge_sobel_x, padding=1)
        edge_target_y = F.conv2d(target, self.edge_sobel_y, padding=1)

        loss_x = F.l1_loss(edge_pred_x, edge_target_x)
        loss_y = F.l1_loss(edge_pred_y, edge_target_y)
        return loss_x + loss_y
