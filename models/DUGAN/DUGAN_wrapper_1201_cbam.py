from torch import nn
import torch
import torch.nn.functional as F


def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(chan_out, chan_out, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Use Conv2d as MLP to handle channel reduction
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Check if planes is large enough for reduction
        if planes // ratio < 1:
            ratio = planes
            
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, kernel_size=1, stride=(2 if downsample else 1))
        self.net = double_conv(input_channels, filters)
        self.cbam = CBAM(filters) # Add CBAM
        self.down = nn.Conv2d(filters, filters, kernel_size=4, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = self.cbam(x) # Apply CBAM
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res


class UpBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Conv2d(input_channels // 2, out_channels, kernel_size=1)
        self.conv = double_conv(input_channels, out_channels)
        self.cbam = CBAM(out_channels) # Add CBAM
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, up):
        x = self.up(x)
        p = self.conv(torch.cat((x, up), dim=1))
        p = self.cbam(p) # Apply CBAM
        sc = self.shortcut(x)
        return p + sc


class UNet(nn.Module):
    def __init__(self, repeat_num, use_tanh=False, use_sigmoid=False, skip_connection=True, use_discriminator=True,
                 conv_dim=64, in_channels=1):
        super().__init__()
        self.use_tanh = use_tanh
        self.skip_connection = skip_connection
        self.use_discriminator = use_discriminator
        self.use_sigmoid = use_sigmoid

        filters = [in_channels] + [min(conv_dim * (2 ** i), 512) for i in range(repeat_num + 1)]
        filters[-1] = filters[-2]

        channel_in_out = list(zip(filters[:-1], filters[1:]))

        self.down_blocks = nn.ModuleList()

        for i, (in_channel, out_channel) in enumerate(channel_in_out):
            self.down_blocks.append(DownBlock(in_channel, out_channel, downsample=(i != (len(channel_in_out) - 1))))

        last_channel = filters[-1]
        if self.use_discriminator:
            self.to_logit = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(),
                nn.Linear(last_channel, 1)
            )

        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), channel_in_out[:-1][::-1])))

        self.conv = double_conv(last_channel, last_channel)
        self.cbam = CBAM(last_channel) # Add CBAM at bottleneck
        
        self.conv_out = nn.Conv2d(in_channels, 1, 1)
        self.__init_weights()


    def forward(self, input):
        x = input
        residuals = []

        for i in range(len(self.down_blocks)):
            x, unet_res = self.down_blocks[i](x)
            residuals.append(unet_res)

        # Bottleneck
        feat = self.conv(x)
        feat = self.cbam(feat)
        bottom_x = feat + x
        
        x = bottom_x
        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)
        dec_out = self.conv_out(x)

        if self.use_discriminator:
            enc_out = self.to_logit(bottom_x)
            if self.use_sigmoid:
                dec_out = torch.sigmoid(dec_out)
                enc_out = torch.sigmoid(enc_out)
            return enc_out.squeeze(), dec_out

        if self.skip_connection:
            dec_out += input
        if self.use_tanh:
            dec_out = torch.tanh(dec_out)

        return dec_out

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                if self.use_discriminator:
                    m.weight.data.normal_(0, 0.01)
                    if hasattr(m.bias, 'data'):
                        m.bias.data.fill_(0)
                else:
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')


if __name__ == '__main__':
    D = UNet(repeat_num=3, use_discriminator=True)
    inputs = torch.randn(4, 1, 64, 64)
    enc_out, dec_out = D(inputs)
    print(enc_out.shape, dec_out.shape)
