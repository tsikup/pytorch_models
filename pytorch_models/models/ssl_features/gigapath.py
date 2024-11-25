"""
https://huggingface.co/prov-gigapath/prov-gigapath
"""

import torch
import timm
from torchvision import transforms


class GigaPath(torch.nn.Module):
    def __init__(self):
        super(GigaPath, self).__init__()
        # need to specify MLP layer and activation function for proper init

        self.model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )

    def get_transforms(self):
        transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return transform

    def forward(self, x):
        embedding = self.model(x).squeeze()
        return embedding


def gigapath():
    return GigaPath()
