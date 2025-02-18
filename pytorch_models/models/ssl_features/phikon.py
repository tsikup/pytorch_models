"""
https://github.com/owkin/HistoSSLscaling/tree/main
"""

import torch
from transformers import ViTModel
from transformers import AutoModel


class PhikonModel(torch.nn.Module):
    def __init__(self, v2=False):
        super(PhikonModel, self).__init__()
        if v2:
            self.model = AutoModel.from_pretrained("owkin/phikon-v2")
        else:
            self.model = ViTModel.from_pretrained(
                "owkin/phikon", add_pooling_layer=False
            )

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[
            :, 0, :
        ]  # (1, 768) shape for v1 and (1, 1024) for v2
        return features


def phikon(v2=False):
    return PhikonModel(v2=v2)
