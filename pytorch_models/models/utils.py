import math

import torch
import torch.nn as nn
from torch.nn import Parameter


class LinearWeighted(nn.Module):
    def __init__(self, in_features):
        super(LinearWeighted, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(1, in_features))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return input * self.weight


class LinearWeightedSum(nn.Module):
    def __init__(self, in_features, n_resolutions):
        super(LinearWeightedSum, self).__init__()
        self.in_features = in_features
        self.n_resolutions = n_resolutions
        self.linear = []
        for _ in range(n_resolutions):
            self.linear.append(LinearWeighted(in_features))
        self.linear = nn.ModuleList(self.linear)

    def forward(self, feats):
        return sum([l(f) for l, f in zip(self.linear, feats)])


class LinearWeightedTransformationSum(nn.Module):
    def __init__(self, in_features, n_resolutions):
        super(LinearWeightedTransformationSum, self).__init__()
        self.in_features = in_features
        self.n_resolutions = n_resolutions
        self.linear = []
        for _ in range(n_resolutions):
            self.linear.append(nn.Linear(in_features, in_features, bias=False))
        self.linear = nn.ModuleList(self.linear)

    def forward(self, feats):
        return sum([l(f) for l, f in zip(self.linear, feats)])
