import numpy as np
import torch
from torch import nn


class pattern_norm(nn.Module):
    def __init__(self, scale=1.0):
        super(pattern_norm, self).__init__()
        self.scale = scale

    def forward(self, features):
        sizes = features.size()
        if len(sizes) > 2:
            features = features.view(-1, np.prod(sizes[1:]))
            features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12)
            features = features.view(sizes)
        return features
