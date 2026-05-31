import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode="zeros",
                 groups=1, bias=False, use_relu=True, use_bn=True, pooling="None"):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, padding_mode=padding_mode, groups=groups, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.Identity()

        if pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        x = self.pool(x)
        return x
