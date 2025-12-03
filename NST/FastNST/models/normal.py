import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        return F.relu(self.inorm(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Identity(),                           
            nn.Conv2d(channels, channels, 3, 1, 1),    
            nn.InstanceNorm2d(channels, affine=True),  
            nn.Identity(),                         
            nn.Identity(),                            
            nn.Conv2d(channels, channels, 3, 1, 1),   
            nn.InstanceNorm2d(channels, affine=True),  
            nn.ReLU(inplace=True)                      
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel, upsample=None):
        super().__init__()
        padding = kernel // 2

        self.deconv = nn.ConvTranspose2d(
            in_c, out_c,
            kernel_size=kernel,
            stride=upsample,
            padding=padding,
            output_padding=(upsample - 1) if upsample else 0
        )

        self.inorm = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        x = self.deconv(x)
        return F.relu(self.inorm(x))


class TransformerNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        # Downsampling
        self.c1 = ConvLayer(4, 32, 9, 1)
        self.c2 = ConvLayer(32,  64, 3, 2)
        self.c3 = ConvLayer(64, 128, 3, 2)

        self.r1 = ResidualBlock(128)
        self.r2 = ResidualBlock(128)
        self.r3 = ResidualBlock(128)
        self.r4 = ResidualBlock(128)
        self.r5 = ResidualBlock(128)

        self.u1 = UpsampleConv(128, 64, 3, upsample=2)
        self.u2 = UpsampleConv(64,  32, 3, upsample=2)

        self.conv_out = nn.Conv2d(32, 3, 9, 1, padding=4)

    def forward(self, x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.c3(y)

        y = self.r1(y)
        y = self.r2(y)
        y = self.r3(y)
        y = self.r4(y)
        y = self.r5(y)

        y = self.u1(y)
        y = self.u2(y)

        y = self.conv_out(y)
        return torch.sigmoid(y)
