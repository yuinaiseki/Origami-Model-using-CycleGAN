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


class StylizedResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in1   = nn.InstanceNorm2d(channels, affine=True)

        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.in2   = nn.InstanceNorm2d(channels, affine=True)

        self.style_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out  = F.relu(self.in1(self.conv1(x)))
        out  = self.in2(self.conv2(out))

        gate = self.style_gate(out)

        return x + out * gate

class UpsampleConv(nn.Module):
    def __init__(self, in_c, out_c, kernel, upsample=None):
        super().__init__()
        self.upsample = upsample

        padding = kernel // 2
        self.conv  = nn.Conv2d(in_c, out_c, kernel, 1, padding)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upsample, mode='nearest')
        return F.relu(self.inorm(self.conv(x)))


class TransformerNetStylized(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1 = ConvLayer(4, 32, 9, 1)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.conv3 = ConvLayer(64, 128, 3, 2)

        self.res1 = StylizedResidualBlock(128)
        self.res2 = StylizedResidualBlock(128)
        self.res3 = StylizedResidualBlock(128)
        self.res4 = StylizedResidualBlock(128)
        self.res5 = StylizedResidualBlock(128)

        self.up1 = UpsampleConv(128, 64, 3, upsample=2)
        self.up2 = UpsampleConv(64, 32, 3, upsample=2)

        self.conv_out = nn.Conv2d(32, 3, 9, 1, padding=4)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.up1(y)
        y = self.up2(y)

        y = self.conv_out(y)
        return torch.sigmoid(y)
