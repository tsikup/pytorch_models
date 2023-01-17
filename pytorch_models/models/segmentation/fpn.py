import torch
from torch import nn

from ..base import BaseModel


class FPN(BaseModel):
    def __init__(self, config, pyramid_channels=256, segmentation_channels=256):
        super(FPN, self).__init__(config)

        # Bottom-up layers
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)
        self.conv_down5 = double_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)

        # Top layer
        self.toplayer = nn.Conv2d(
            1024, 256, kernel_size=1, stride=1, padding=0
        )  # Reduce channels

        # Lateral layers
        self.lat_layer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lat_layer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Segmentation block layers
        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(
                    pyramid_channels, segmentation_channels, n_upsamples=n_upsamples
                )
                for n_upsamples in [0, 1, 2, 3]
            ]
        )

        # Last layer
        self.last_conv = nn.Conv2d(
            256, self.n_classes, kernel_size=1, stride=1, padding=0
        )

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        upsample = nn.Upsample(size=(H, W), mode="bilinear", align_corners=True)

        return upsample(x) + y

    def upsample(self, x, h, w):
        sample = nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)
        return sample(x)

    def forward(self, x):
        # Bottom-up
        c1 = self.maxpool(self.conv_down1(x))
        c2 = self.maxpool(self.conv_down2(c1))
        c3 = self.maxpool(self.conv_down3(c2))
        c4 = self.maxpool(self.conv_down4(c3))
        c5 = self.maxpool(self.conv_down5(c4))

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self.upsample_add(p5, self.lat_layer1(c4))
        p3 = self.upsample_add(p4, self.lat_layer2(c3))
        p2 = self.upsample_add(p3, self.lat_layer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Segmentation
        _, _, h, w = p2.size()
        feature_pyramid = [
            seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])
        ]

        out = self.upsample(self.last_conv(sum(feature_pyramid)), 4 * h, 4 * w)

        out = torch.sigmoid(out)
        return out


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class ConvReluUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.make_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = self.make_upsample(x)
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            ConvReluUpsample(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(
                    ConvReluUpsample(out_channels, out_channels, upsample=True)
                )

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.convrelu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)

        self.deconv = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )

        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)

        return x
