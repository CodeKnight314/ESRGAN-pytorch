import torch 
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.block1 = self._make_layer(in_channels, 64, normalize=False)
        self.block2 = self._make_layer(64, 64, stride=2)
        self.block3 = self._make_layer(64, 128)
        self.block4 = self._make_layer(128, 128, stride=2)
        self.block5 = self._make_layer(128, 256)
        self.block6 = self._make_layer(256, 256, stride=2)
        self.block7 = self._make_layer(256, 512)
        self.block8 = self._make_layer(512, 512, stride=2)
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc = nn.Linear(1024 * 4 * 4 * 4, 1)

    def _make_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.leaky_relu(self.conv1(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
