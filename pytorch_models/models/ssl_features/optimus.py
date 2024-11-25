"""
https://huggingface.co/bioptimus/H-optimus-0
"""

import timm
import torch

from torchvision import transforms


class Optimus(torch.nn.Module):
    def __init__(self, slide_level=False):
        super(Optimus, self).__init__()

        self.model = timm.create_model("hf_hub:bioptimus/H-optimus-0", pretrained=True)

    def get_transforms(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        return transform

    def forward(self, x):
        embedding = self.model(x).squeeze()
        return embedding


def optimus():
    return Optimus()
