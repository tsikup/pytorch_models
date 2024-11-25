"""
https://huggingface.co/prov-gigapath/prov-gigapath
"""

import timm
import torch

# import gigapath
from torchvision import transforms


class GigaPath(torch.nn.Module):
    def __init__(self, slide_level=False):
        super(GigaPath, self).__init__()

        self.model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )

        # if slide_level:
        #     self.slide_encoder = gigapath.slide_encoder.create_model(
        #         "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
        #     )

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

    # def slide_representation(self, x, coords):
    #     patch_embedding = self.forward(x)
    #     embedding = self.slide_encoder(patch_embedding, coords).squeeze()
    #     return embedding


def gigapath():
    return GigaPath()
