import torch 
import torch.nn as nn 

class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.conv5(torch.cat([x, out1, out2, out3, out4], 1))
        return out5 * 0.2 + x 

class RRDB(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(RRDB, self).__init__()
        self.dense1 = DenseBlock(in_channels, growth_rate)
        self.dense2 = DenseBlock(in_channels, growth_rate)
        self.dense3 = DenseBlock(in_channels, growth_rate)

    def forward(self, x):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return out * 0.2 + x 

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_rrdb_blocks=23, hidden_channels=64, growth_rate=32):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        self.rrdb_blocks = nn.Sequential(*[RRDB(hidden_channels) for _ in range(num_rrdb_blocks)])
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.rrdb_blocks(out1)
        out = self.conv2(out)
        out = self.upsample(out + out1)
        out = self.conv3(out)
        return out