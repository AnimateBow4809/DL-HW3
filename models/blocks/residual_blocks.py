from torch import nn

from models.blocks.base_blocks import BasicConv2d


class ResidualBlock(nn.Module):
    """
    implementation of residual block using original resnet paper
    """

    def __init__(self, in_channels, out_channels, stride=1, use_projection=False,pooling="None"):
        super().__init__()
        self.main_path = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, use_relu=True),
            BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_relu=False)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels or use_projection:
            self.shortcut = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                        use_relu=False)

        self.final_relu = nn.ReLU(inplace=True)
        if pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity
        out = self.final_relu(out)
        return self.pool(out)


class Bottleneck(nn.Module):
    """
    implementation of bottleneck block using resnetB from bags of tricks paper
    """

    def __init__(self, in_channels, intermediate_channels, out_channels, stride=1,pooling="None"):
        super().__init__()
        self.main_path = nn.Sequential(
            BasicConv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, use_relu=True),
            BasicConv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1,
                        use_relu=True),
            BasicConv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0, use_relu=False)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                        use_relu=False)

        self.final_relu = nn.ReLU(inplace=True)

        if pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out += identity
        out = self.final_relu(out)
        return self.pool(out)