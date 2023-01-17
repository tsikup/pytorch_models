import os
import torch
from torch import nn
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class ResNet50_SimCLR(nn.Module):
    def __init__(self, ckpt=None):
        super(ResNet50_SimCLR, self).__init__()

        self._model = torchvision_ssl_encoder("resnet50", pretrained=True)

        if ckpt is not None:
            assert os.path.isfile(ckpt)
            missing_keys, unexpected_keys = self._model.load_state_dict(
                torch.load(ckpt), strict=False
            )
            print("Missing keys in ResNet50 simclr:", missing_keys)
            print("Unexpected keys in ResNet50 simclr:", unexpected_keys)

    def freeze(self):
        for p in self._model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self._model.forward(x)[0]
