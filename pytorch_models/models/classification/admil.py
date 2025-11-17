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
    def __init__(self, L=384, D=128, K=1, n_classes=1):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.n_classes = n_classes

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Linear(self.L * self.K, self.n_classes)

    def forward(self, H):
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        logits = self.classifier(M)

        if self.n_classes == 1:
            Y_hat = torch.ge(torch.sigmoid(logits), 0.5).float()
        else:
            Y_hat = torch.topk(logits, 1, dim=1)[1].float()

        return logits, Y_hat, A


class GatedAttention(nn.Module):
    def __init__(self, L=384, D=128, K=1, n_classes=1):
        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.n_classes = n_classes

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Linear(self.L * self.K, self.n_classes)

    def forward(self, H):
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        logits = self.classifier(M)

        if self.n_classes == 1:
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
        K: int = 1,
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
            self.model = GatedAttention(
                L=size[0], D=size[1], K=K, n_classes=self.n_classes
            )
        else:
            self.model = Attention(L=size[0], D=size[1], K=K, n_classes=self.n_classes)

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch["features"], batch["labels"]

        # Prediction
        logits, Y_hat, A = self._forward(features)

        loss = None
        if not is_predict:
            loss = self.loss.forward(
                logits, target.float() if logits.shape[1] == 1 else target.view(-1)
            )
        
        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "attention": A,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch):
        logits = []
        Y_hat = []
        A = []
        for idx, singlePatientFeatures in enumerate(features_batch):
            h = [
                singlePatientFeatures[f].squeeze() for f in singlePatientFeatures.keys()
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits, _Y_hat, _A = self.model.forward(h)
            logits.append(_logits)
            Y_hat.append(_Y_hat)
            A.append(_A)
        return torch.vstack(logits), torch.vstack(Y_hat), A
