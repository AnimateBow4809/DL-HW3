from torch import nn

from models.blocks.unet_blocks import UnetBlock


class UNet(nn.Module):
    def __init__(self, depth=5, in_channels=3, out_channels=3, use_bn=False, pooling="max"
                 , expansion="up"):
        super(UNet, self).__init__()
        self.encoder_dim_list = [64 * (2 ** i) for i in range(depth)]
        self.decoder_dim_list = list(reversed(self.encoder_dim_list))
        self.encoder_dim_list = [in_channels] + self.encoder_dim_list
        self.decoder_dim_list = self.decoder_dim_list + [out_channels]
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.depth = depth
        for i in range(self.depth - 1):
            self.encoder_list.append(
                UnetBlock(self.encoder_dim_list[i], self.encoder_dim_list[i + 1], use_bn=use_bn, pooling=pooling))
            self.decoder_list.append(
                UnetBlock(self.decoder_dim_list[i], self.decoder_dim_list[i + 1], use_bn=use_bn, expansion=expansion))

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
            x = torch.cat((up, skip), dim=1)
            layer = self.decoder_list[i]
            x, up = layer(x)
        out = self.head(x)
        return out
