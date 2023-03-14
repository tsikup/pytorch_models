import os
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch import nn
from dotmap import DotMap

from ....utils.weight_init import init_weights
from ...base import BaseModel
from .layers import unetConv2, unetUp, unetUp_origin


class UNet(BaseModel):
    """
    Original UNet

    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image
    segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp.
    234-241). Springer, Cham.
    """

    def __init__(
        self,
        config,
        in_channels=3,
        n_classes=1,
        num_filters=[64, 128, 256, 512, 1024],
        is_deconv=True,
        is_batchnorm=True,
        apply_last_layer=True,
    ):
        super(UNet, self).__init__(config, in_channels=in_channels, n_classes=n_classes)
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.apply_last_layer = apply_last_layer
        filters = num_filters
        depth = len(filters)

        # Encoder
        in_channels = self.in_channels
        for d in range(depth):
            self.add_module(
                "conv" + str(d + 1),
                unetConv2(in_channels, filters[d], self.is_batchnorm),
            )
            self.add_module("maxpool" + str(d + 1), nn.MaxPool2d(kernel_size=2))
            in_channels = filters[d]

        self.add_module(
            "center",
            unetConv2(filters[depth - 2], filters[depth - 1], self.is_batchnorm),
        )

        # Decoder
        for d in reversed(range(1, depth)):
            self.add_module(
                "up_concat" + str(d), unetUp(filters[d], filters[d - 1], self.is_deconv)
            )

        if self.apply_last_layer:
            self.add_module(
                "outconv", nn.Conv2d(filters[0], self.n_classes, 3, padding=1)
            )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming", init_bias=True)
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming", init_bias=True)

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def _forward(self, inputs):
        # conv1 = self.conv1(inputs)  # 16*512*1024
        # maxpool1 = self.maxpool1(conv1)  # 16*256*512
        #
        # conv2 = self.conv2(maxpool1)  # 32*256*512
        # maxpool2 = self.maxpool2(conv2)  # 32*128*256
        #
        # conv3 = self.conv3(maxpool2)  # 64*128*256
        # maxpool3 = self.maxpool3(conv3)  # 64*64*128
        #
        # conv4 = self.conv4(maxpool3)  # 128*64*128
        # maxpool4 = self.maxpool4(conv4)  # 128*32*64
        #
        # center = self.center(maxpool4)  # 256*32*64
        #
        # up4 = self.up_concat4(center, conv4)  # 128*64*128
        # up3 = self.up_concat3(up4, conv3)  # 64*128*256
        # up2 = self.up_concat2(up3, conv2)  # 32*256*512
        # up1 = self.up_concat1(up2, conv1)  # 16*512*1024
        #
        # d1 = self.outconv1(up1)  # 256
        return self(inputs)


class ResNetUnet(BaseModel):
    def __init__(
        self, config: DotMap, in_channels: int, n_classes: int, pretrained=True
    ):
        super(ResNetUnet, self).__init__(
            config, in_channels=in_channels, n_classes=n_classes
        )
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=self.in_channels,
            classes=self.n_classes,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import json
    from dotmap import DotMap
    from definitions import ROOT_DIR

    with open(
        os.path.join(ROOT_DIR, "assets/configs/training_seg_config.json"), "r"
    ) as f:
        config = DotMap(json.load(f))

    a = torch.rand(2, 3, 512, 512)
    model = ResNetUnet(config, pretrained=False)
    y = model.forward(a)  # [2, 1, 512, 512]
    print(y.shape)
