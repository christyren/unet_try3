# gitに落ちてたやつ元論文とは違いpadding1でやってるのでsize落ちない
import torch


import torch.nn.functional as F

from .unet_parts import *


class UNet_half(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_half, self).__init__()
        self.inc = inconv2(n_channels, 64)
        self.down1 = down2(64, 128)
        self.down2 = down2(128, 256)
        self.down3 = down2(256, 512)
        self.down4 = down2(512, 512)
        self.up1 = up2(1024, 256)
        self.up2 = up2(512, 128)
        self.up3 = up2(256, 64)
        self.up4 = up2(128, 64)
        self.outc = outconv2(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # BCEloss使うならsigmoidいるけどcross_entropyはいらない
        # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        # このサイトのexsample
        # return torch.sigmoid(x)
        return x


class UNet_2(nn.Module):
    # Simplify the input and output layer
    def __init__(self, n_channels, n_classes):
        super(UNet_2, self).__init__()
        self.inc = inconv2(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv2(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # BCEloss使うならsigmoidいるけどcross_entropyはいらない
        # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        # このサイトのexsample
        # return torch.sigmoid(x)
        return x


class UNet_3(nn.Module):
    # Simplify the middle layer
    def __init__(self, n_channels, n_classes):
        super(UNet_3, self).__init__()
        self.inc = inconv_simple(n_channels, 64)
        self.down1 = down_simple(64, 128)
        self.down2 = down_simple(128, 256)
        self.down3 = down_simple(256, 512)
        self.down4 = down_simple(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # BCEloss使うならsigmoidいるけどcross_entropyはいらない
        # https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
        # このサイトのexsample
        # return torch.sigmoid(x)
        return x