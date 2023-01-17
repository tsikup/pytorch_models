# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9288896

from typing import List, Dict
import torch
from torch import nn, Tensor
from dotmap import DotMap
from collections import OrderedDict
import torch.nn.functional as F

from .resnet import _ConvBatchNormReLU, _ResBlock
from ....models.base import BaseModel
from .deeplabv3 import _ASPPModule


class TriUpSegNet_DeepLab_Backbone(nn.Module):
    def __init__(
        self,
        n_classes,
        n_blocks=[3, 4, 23, 3],
        grids=[1, 2, 4],
        output_stride=16,
        pyramids=[6, 12, 18],
    ):
        super(TriUpSegNet_DeepLab_Backbone, self).__init__()
        # **************** #
        # DeepLab backbone #
        # **************** #

        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        # Encoder
        self.layer0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBatchNormReLU(3, 64, 7, 2, 2, 1)),
                    ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                ]
            )
        )

        self.layer1 = _ResBlock(n_blocks[0], 64, 64, 256, stride[0], dilation[0])

        self.layer2 = _ResBlock(n_blocks[1], 256, 128, 512, stride[1], dilation[1])

        self.layer3 = _ResBlock(n_blocks[2], 512, 256, 1024, stride[2], dilation[2])

        self.layer4 = _ResBlock(
            n_blocks[3], 1024, 512, 2048, stride[3], dilation[3], mg=grids
        )

        self.aspp = _ASPPModule(2048, 256, pyramids)

        # reduce layer in deeplab implementation
        self.fc1 = _ConvBatchNormReLU(256, 48, 1, 1, 0, 1)

        self.fc3 = _ConvBatchNormReLU(512, 48, 1, 1, 0, 1)

        # fc2 in deeplab implementation
        self.fc5 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBatchNormReLU(304, 256, 3, 1, 1, 1)),
                    ("conv2", _ConvBatchNormReLU(256, 256, 3, 1, 1, 1)),
                    ("conv3", nn.Conv2d(256, n_classes, kernel_size=1)),
                ]
            )
        )

    def forward(self, x):
        # Layer 0
        h = self.layer0(x)
        # Layer 1 + fc1
        h = self.layer1(h)
        h_ = self.fc1(h)
        # Layer 2 + fc3
        h = self.layer2(h)
        h_2 = self.fc3(h)
        # Layer 3
        h = self.layer3(h)
        # Layer 4
        h = self.layer4(h)
        # Layer ASPP
        h = self.aspp(h)
        return h, h_, h_2

    def forward_fc(self, x):
        return self.fc5(x)

    def freeze_bn(self):
        layers = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.aspp,
            self.fc1,
            self.fc3,
            self.fc5,
        ]
        for layer in layers:
            for m in layer.named_modules():
                if isinstance(m[1], nn.BatchNorm2d):
                    m[1].eval()


class TriUpSegNetA(BaseModel):
    def __init__(
        self,
        config: DotMap,
        n_blocks=[3, 4, 23, 3],
        grids=[1, 2, 4],
        output_stride=16,
        pyramids=[6, 12, 18],
    ):
        super(TriUpSegNetA, self).__init__(config)

        # **************** #
        # DeepLab backbone #
        # **************** #

        self.backbone = TriUpSegNet_DeepLab_Backbone(
            self.n_classes, n_blocks, grids, output_stride, pyramids
        )

        # ************ #
        # TriUpSegNetA #
        # ************ #

        # fc1 in deeplab implementation
        self.fc2 = _ConvBatchNormReLU(256 * (len(pyramids) + 2), 256, 1, 1, 0, 1)

        self.fc4 = _ConvBatchNormReLU(304, 256, 3, 1, 1, 1)

    def forward(self, x):
        h, h_, h_2 = self.backbone(x)
        # Layer fc2 + upsampling + concat
        h = self.fc2(h)
        h = F.interpolate(h, size=h_2.shape[2:], mode="bilinear")
        h = torch.cat((h, h_2), dim=1)
        # Layer fc4 + upsampling + concat
        h = self.fc4(h)
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear")
        h = torch.cat((h, h_), dim=1)
        # Final FC5 on concatenated layers
        h = self.backbone.forward_fc(h)
        # Final upsample
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear")
        return h

    def freeze_bn(self):
        layers = [self.fc2, self.fc4]
        self.backbone.freeze_bn()
        for layer in layers:
            for m in layer.named_modules():
                if isinstance(m[1], nn.BatchNorm2d):
                    m[1].eval()


class TriUpSegNetB(BaseModel):
    def __init__(
        self,
        config: DotMap,
        n_blocks=[3, 4, 23, 3],
        grids=[1, 2, 4],
        output_stride=16,
        pyramids=[6, 12, 18],
    ):
        super(TriUpSegNetB, self).__init__(config)

        # **************** #
        # DeepLab backbone #
        # **************** #

        self.backbone = TriUpSegNet_DeepLab_Backbone(
            self.n_classes, n_blocks, grids, output_stride, pyramids
        )

        # ************ #
        # TriUpSegNetB #
        # ************ #

        # fc1 in deeplab implementation
        self.fc2 = _ConvBatchNormReLU(256 * (len(pyramids) + 2), 256, 1, 1, 0, 1)

    def forward(self, x):
        h, h_, h_2 = self.backbone(x)
        # Layer fc2
        h = self.fc2(h)
        # upsample h to h_
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear")
        # upsample h_2 to h_
        h_2 = F.interpolate(h_2, size=h_.shape[2:], mode="bilinear")
        # concatenate h, h_2, h_
        h = torch.cat((h, h_2, h_), dim=1)
        # Final FC5 on concatenated layers
        h = self.backbone.forward_fc(h)
        # Final upsample
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear")
        return h

    def freeze_bn(self):
        self.backbone.freeze_bn()
        for m in self.fc2.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()
