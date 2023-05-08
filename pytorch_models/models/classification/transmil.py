from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, size=(1024, 512)):
        super(TransMIL, self).__init__()

        if len(size) > 2:
            p_size = size[:-1]
            size = size[-2:]
        else:
            p_size = size

        assert p_size[-1] == size[0], "p_size[-1] != size[0]"

        self._fc1 = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(p_size[i], p_size[i + 1]), nn.ReLU())
                for i in range(len(p_size) - 1)
            ]
        )

        self.pos_layer = PPEG(dim=size[1])
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=size[1])
        self.layer2 = TransLayer(dim=size[1])
        self.norm = nn.LayerNorm(size[1])
        self._fc2 = nn.Linear(size[1], self.n_classes)

    def forward(self, h: torch.Tensor):  # list of [B, n, 1024] if size[0] == 1024
        device = h.device

        h = self._fc1(h)  # [B, n, 512] if size[1] == 512

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512] if size[0] == 512

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # .cuda()
        cls_tokens = cls_tokens.to(device)

        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512] if size[0] == 512

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512] if size[0] == 512

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512] if size[0] == 512

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat}
        return results_dict


class TransMIL_Features_PL(BaseMILModel):
    def __init__(
        self,
        config,
        n_classes,
        size=(1024, 512),
        multires_aggregation=None,
    ):
        self.multires_aggregation = multires_aggregation
        super(TransMIL_Features_PL, self).__init__(config, n_classes=n_classes)
        self.model = TransMIL(n_classes=n_classes, size=size)

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch

        # Prediction
        results_dict = self._forward(data=features)
        logits = results_dict["logits"]
        preds = results_dict["Y_prob"]

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss = self.loss.forward(logits, target.squeeze(dim=1))

        if self.n_classes in [1, 2]:
            preds = preds[:, 1]
            preds = torch.unsqueeze(preds, dim=1)

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
        }

    def _forward(self, data: Dict[str, torch.Tensor]):
        h = [data[key] for key in data]
        h = aggregate_features(h, method=self.multires_aggregation)
        return self.model(h)


if __name__ == "__main__":
    _data = torch.randn((1, 6000, 1024))
    _model = TransMIL(n_classes=2)
    print(_model.eval())
    _results_dict = _model(data=_data)
    print(_results_dict)
