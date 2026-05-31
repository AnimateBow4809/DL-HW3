import torch
import torch.nn as nn

from models.blocks.base_blocks import BasicConv2d


class InceptionBlock(nn.Module):
    """
    Inception Block based on figure 5 of https://arxiv.org/abs/1512.00567
    """

    def __init__(self, in_channels, branch1_out, branch2_out, branch3_out, branch4_out, pooling="None"):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, branch1_out, kernel_size=1, padding=0)
        )
        self.branch2 = BasicConv2d(in_channels, branch2_out, kernel_size=1, padding=0)

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, branch3_out, kernel_size=1, padding=0),
            BasicConv2d(branch3_out, branch3_out, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, branch4_out, kernel_size=1, padding=0),
            BasicConv2d(branch4_out, branch4_out, kernel_size=3, padding=1),
            BasicConv2d(branch4_out, branch4_out, kernel_size=3, padding=1)
        )
        if pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        return self.pool(torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1))
