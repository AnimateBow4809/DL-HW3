import torch
from torch import nn

from models.blocks.unet_blocks import UnetBlock


class UNet(nn.Module):
    def __init__(self, depth=5, in_channels=3, out_channels=3, use_bn=False, pooling="max"
                 , expansion="up", use_skip=True):
        super(UNet, self).__init__()
        self.encoder_dim_list = [64 * (2 ** i) for i in range(depth)]
        self.decoder_dim_list = list(reversed(self.encoder_dim_list))
        self.encoder_dim_list = [in_channels] + self.encoder_dim_list
        self.decoder_dim_list = self.decoder_dim_list + [out_channels]
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.depth = depth
        self.use_skip = use_skip
        for i in range(self.depth - 1):
            self.encoder_list.append(
                UnetBlock(self.encoder_dim_list[i], self.encoder_dim_list[i + 1], use_bn=use_bn, pooling=pooling))
            current_expansion = expansion if i < (self.depth - 2) else None
            decoder_in = self.decoder_dim_list[i]
            if not self.use_skip:
                decoder_in //= 2
            self.decoder_list.append(
                UnetBlock(decoder_in, self.decoder_dim_list[i + 1], use_bn=use_bn,
                          expansion=current_expansion))

        self.bridge = UnetBlock(self.encoder_dim_list[depth - 1], self.encoder_dim_list[depth], use_bn=use_bn,
                                expansion=expansion)
        self.head = nn.Conv2d(self.decoder_dim_list[depth - 1], self.decoder_dim_list[depth], kernel_size=1, stride=1)

    def forward(self, x):
        skip_list = []
        for i in range(self.depth - 1):
            layer = self.encoder_list[i]
            skip, x = layer(x)
            skip_list.append(skip)
        _, up = self.bridge(x)
        skip_list = list(reversed(skip_list))
        for i in range(self.depth - 1):
            skip = skip_list[i]
            if self.use_skip:
                x = torch.cat((up, skip), dim=1)
            else:
                x = up
            layer = self.decoder_list[i]
            x, up = layer(x)
        out = self.head(x)
        return out
