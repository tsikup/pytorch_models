"""
Ilse, M., Tomczak, J. M., & Welling, M. (2018).
Attention-based Deep Multiple Instance Learning. arXiv preprint arXiv:1802.04712.
https://arxiv.org/pdf/1802.04712.pdf
https://github.com/AMLab-Amsterdam/AttentionDeepMIL
"""
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features


class Attention(nn.Module):
    def __init__(self, L=384, D=128, n_classes=1):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = n_classes

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, H):
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        logits = self.classifier(M)

        if self.K == 1:
            Y_hat = torch.ge(torch.sigmoid(logits), 0.5).float()
        else:
            Y_hat = torch.topk(logits, 1, dim=1)[1].float()

        return logits, Y_hat, A


class GatedAttention(nn.Module):
    def __init__(self, L=384, D=128, n_classes=1):
        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.K = n_classes

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1), nn.Sigmoid())

    def forward(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        logits = self.classifier(M)

        if self.K == 1:
            Y_hat = torch.ge(torch.sigmoid(logits), 0.5).float()
        else:
            Y_hat = torch.topk(logits, 1, dim=1)[1].float()

        return logits, Y_hat, A


class AttentionDeepMIL_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        gated: bool = True,
        multires_aggregation: Union[None, str] = None,
    ):
        super(AttentionDeepMIL_PL, self).__init__(config, n_classes=n_classes)

        assert len(size) == 2, "size must be a tuple of (n_features, layer_size)"
        assert self.n_classes > 0, "n_classes must be greater than 0"
        if self.n_classes == 2:
            self.n_classes = 1

        self.multires_aggregation = multires_aggregation

        if gated:
            self.model = GatedAttention(L=size[0], D=size[1], n_classes=self.n_classes)
        else:
            self.model = Attention(L=size[0], D=size[1], n_classes=self.n_classes)

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch

        # Prediction
        logits, preds, _ = self._forward(features)

        loss = None
        if not is_predict:
            loss = self.loss.forward(logits, target.squeeze(dim=1))

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


if __name__ == "__main__":
    # create test data and model
    n_features = 384
    n_classes = 1
    n_samples = 100

    features = torch.rand(n_samples, n_features)

    model = GatedAttention(L=n_features, D=128, n_classes=n_classes)

    # test forward
    logits, preds, A = model.forward(features)
    print(
        logits.shape,
        preds.shape,
        A.shape,
    )
