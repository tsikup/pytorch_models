# https://github.com/mdsatria/MultiAttentionMIL
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap

from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features


class MultiAttentionMIL(nn.Module):
    def __init__(
        self, num_classes=1, size=(384, 128), use_dropout=False, n_dropout=0.4
    ):
        super(MultiAttentionMIL, self).__init__()
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.n_dropout = n_dropout
        self.num_features = size[0]
        self.D = size[1]

        self.fc1 = nn.Sequential(
            nn.Linear(self.num_features, self.D),
            nn.ReLU(),
        )
        self.attention1 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention2 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        self.attention3 = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, 1)
        )

        self.fc4 = nn.Sequential(nn.Linear(self.D, self.num_classes))

    def forward(self, x):
        ################
        x1 = x.squeeze(0)
        x1 = self.fc1(x1)
        if self.use_dropout:
            x1 = nn.Dropout(self.n_dropout)(x1)
        # -------
        a1 = self.attention1(x1)
        a1 = torch.transpose(a1, 1, 0)
        a1 = nn.Softmax(dim=1)(a1)
        # -------
        m1 = torch.mm(a1, x1)
        m1 = m1.view(-1, 1 * self.D)

        ################
        x2 = self.fc2(x1)
        if self.use_dropout:
            x2 = nn.Dropout(self.n_dropout)(x2)
        # -------
        a2 = self.attention2(x2)
        a2 = torch.transpose(a2, 1, 0)
        a2 = nn.Softmax(dim=1)(a2)
        # -------
        m2 = torch.mm(a2, x2)
        m2 = m2.view(-1, 1 * self.D)
        m2 += m1

        ################
        x3 = self.fc3(x2)
        if self.use_dropout:
            x3 = nn.Dropout(self.n_dropout)(x3)
        # -------
        a3 = self.attention3(x3)
        a3 = torch.transpose(a3, 1, 0)
        a3 = nn.Softmax(dim=1)(a3)
        # -------
        m3 = torch.mm(a3, x3)
        m3 = m3.view(-1, 1 * self.D)
        m3 += m2

        logits = self.fc4(m3)

        return logits, a1, a2, a3


class MAMIL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        multires_aggregation: Union[None, str] = None,
    ):
        super(MAMIL, self).__init__(config, n_classes=n_classes)
        assert len(size) == 2, "size must be a tuple of (n_features, layer_size)"
        assert self.n_classes > 0, "n_classes must be greater than 0"
        if self.n_classes == 2:
            self.n_classes = 1

        self.multires_aggregation = multires_aggregation
        self.dropout = dropout

        self.model = MultiAttentionMIL(self.n_classes, size, use_dropout=self.dropout)

    def forward(self, batch, is_predict=False):
        raise NotImplementedError
        # Batch
        features, target = batch

        # Prediction
        logits, a1, a2, a3 = self._forward(features)
        logits = logits.squeeze(dim=1)
        target = target.squeeze(dim=1)

        loss = None
        if not is_predict:
            loss = self.loss.forward(logits.float(), target.float())

        preds = torch.sigmoid(logits) if self.n_classes == 1 else F.softmax(logits)

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
        }

    def _forward(self, features):
        h: List[torch.Tensor] = [features[key] for key in features]
        h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
        if len(h.shape) == 3:
            h = h.squeeze(dim=0)
        return self.model.forward(h)
