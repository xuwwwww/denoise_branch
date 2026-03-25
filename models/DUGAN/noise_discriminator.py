import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(NoiseDiscriminator, self).__init__()
        
        # Encoder
        # Stage 1: 3 -> 64
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=2, dilation=2), # Dilated conv for artifacts
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels)
        )
        self.pool1 = nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1) # Downsample

        # Stage 2: 64 -> 128
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*2),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*2)
        )
        self.pool2 = nn.Conv2d(base_channels*2, base_channels*2, 4, stride=2, padding=1)

        # Stage 3: 128 -> 256
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*4),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*4)
        )
        self.pool3 = nn.Conv2d(base_channels*4, base_channels*4, 4, stride=2, padding=1)

        # Stage 4: 256 -> 512
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*8),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*8)
        )
        self.pool4 = nn.Conv2d(base_channels*8, base_channels*8, 4, stride=2, padding=1)

        # Bottleneck: Multi-scale block
        self.bottleneck_d1 = nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1, dilation=1)
        self.bottleneck_d2 = nn.Conv2d(base_channels*8, base_channels*8, 3, padding=2, dilation=2)
        self.bottleneck_d4 = nn.Conv2d(base_channels*8, base_channels*8, 3, padding=4, dilation=4)
        self.bottleneck_fusion = nn.Conv2d(base_channels*8 * 3, base_channels*8, 1)

        # Decoder
        # Stage 4: 512 -> 256
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4_conv = nn.Conv2d(base_channels*8 + base_channels*8, base_channels*4, 3, padding=1) # Concat skip
        self.dec4_block = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*4)
        )

        # Stage 3: 256 -> 128
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3_conv = nn.Conv2d(base_channels*4 + base_channels*4, base_channels*2, 3, padding=1)
        self.dec3_block = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels*2)
        )

        # Stage 2: 128 -> 64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2_conv = nn.Conv2d(base_channels*2 + base_channels*2, base_channels, 3, padding=1)
        self.dec2_block = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels)
        )

        # Stage 1: 64 -> 64 (Final processing before heads)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1_conv = nn.Conv2d(base_channels + base_channels, base_channels, 3, padding=1)
        self.dec1_block = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(base_channels)
        )

        # Heads
        self.adv_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid() # Real/Fake score map
        )
        self.noise_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.ReLU(inplace=True) # Noise intensity map (non-negative)
        )

    def forward(self, x_noisy, y_hat, noise_level_map):
        """
        Args:
            x_noisy: Noisy image [B, 1, H, W]
            y_hat: Denoised image [B, 1, H, W]
            noise_level_map: Explicit noise map [B, 1, H, W].
        """
        
        # 1. Prepare Input
        diff = torch.abs(x_noisy - y_hat)
        
        # Input concatenation: [y_hat, residual_abs, noise_level_map]
        inp = torch.cat([y_hat, diff, noise_level_map], dim=1) # [B, 3, H, W]

        # 2. Encoder
        e1 = self.enc1(inp)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # 3. Bottleneck
        b1 = self.bottleneck_d1(p4)
        b2 = self.bottleneck_d2(p4)
        b4 = self.bottleneck_d4(p4)
        bottleneck_out = self.bottleneck_fusion(torch.cat([b1, b2, b4], dim=1))
        bottleneck_out = F.leaky_relu(bottleneck_out, 0.2, inplace=True)

        # 4. Decoder
        d4 = self.up4(bottleneck_out)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4_conv(d4)
        d4 = self.dec4_block(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3_conv(d3)
        d3 = self.dec3_block(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2_conv(d2)
        d2 = self.dec2_block(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_conv(d1)
        d1 = self.dec1_block(d1)

        # 5. Heads
        adv_map = self.adv_head(d1)
        noise_pred_map = self.noise_head(d1)

        return adv_map, noise_pred_map, bottleneck_out
