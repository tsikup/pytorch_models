import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from typing import List

__all__ = ["JointLoss", "WeightedLoss"]


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, losses: List[nn.Module], weights: List[int]):
        super().__init__()
        if weights is None:
            weights = [1 for _ in range(len(losses))]
        assert len(losses) == len(weights)
        self.loss = []
        for i in range(len(losses)):
            l = losses[i]
            w = weights[i]
            self.loss.append(WeightedLoss(l, w))

    def forward(self, *input):
        loss = [l(*input) for l in self.loss]
        return torch.stack(loss).sum()
