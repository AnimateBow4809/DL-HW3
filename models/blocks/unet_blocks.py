from typing import Any
from torch import nn
from models.blocks.base_blocks import BasicConv2d

'''
   if expansion is used the output channel for the corresponding is halved
'''
class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False,
                 pooling=None, expansion=None, padding_mode="reflect"):
        super(UnetBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                 use_bn=use_bn, use_relu=True, padding="same", padding_mode=padding_mode)
        self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                 use_bn=use_bn, use_relu=True, padding="same", padding_mode=padding_mode)

        self.post_op = nn.Identity()
        if pooling == "avg":
            self.post_op = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == "max":
            self.post_op = nn.MaxPool2d(kernel_size=2, stride=2)

        elif expansion == "up":
            self.post_op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, stride=1, padding="same")
            )
        elif expansion == "trans":
            self.post_op = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.post_op(x)
        return x, y