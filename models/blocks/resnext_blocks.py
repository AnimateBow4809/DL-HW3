from torch import nn

from models.blocks.base_blocks import BasicConv2d


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels,out_channels, group_width, cardinality, stride=1, pooling="None"):
        super().__init__()
        intermediate_channels = cardinality * group_width
        self.conv1 = BasicConv2d(in_channels, intermediate_channels, kernel_size=1)
        self.conv2 = BasicConv2d(intermediate_channels, intermediate_channels, kernel_size=3,
                                 stride=stride, padding=1, groups=cardinality)

        self.conv3 = BasicConv2d(intermediate_channels, out_channels, kernel_size=1, use_relu=False)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = BasicConv2d(in_channels, out_channels, kernel_size=1,
                                        stride=stride, use_relu=False)
        if pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += identity
        out = self.relu(out)

        return self.pool(out)
