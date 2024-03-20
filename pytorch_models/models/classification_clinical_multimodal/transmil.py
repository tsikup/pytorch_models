from typing import List

import numpy as np
import torch
from pytorch_models.models.base import BaseClinicalMultimodalMILModel
from pytorch_models.models.classification.transmil import TransMIL
from torch import nn


class TransMIL_Clinical_Multimodal(TransMIL):
    def __init__(
        self,
        n_classes,
        size,
        multimodal_odim,
    ):
        super(TransMIL_Clinical_Multimodal, self).__init__(n_classes, size)

        self._fc2 = nn.Linear(multimodal_odim, self.n_classes)

    def forward_imaging(
        self, h: torch.Tensor
    ):  # list of [B, n, 1024] if size[0] == 1024
        device = h.device

        if len(h.shape) == 2:
            h = h.unsqueeze(0)

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

        return h

    def forward(self, h):
        # ---->predict
        return self._fc2(h)


class TransMIL_Clinical_Multimodal_PL(BaseClinicalMultimodalMILModel):
    def __init__(
        self,
        config,
        n_classes,
        size: List[int],
        size_cat: int,
        size_cont: int,
        clinical_layers: List[int],
        multimodal_odim: int,
        embed_size: list = None,
        batch_norm: bool = True,
        bilinear_scale_dim1: int = 1,
        bilinear_scale_dim2: int = 1,
        multires_aggregation=None,
        multimodal_aggregation="concat",
        n_resolutions: int = 1,  # not used
        dropout=0.5,
    ):
        if n_classes == 1:
            n_classes = 2
        super(TransMIL_Clinical_Multimodal_PL, self).__init__(
            config,
            n_classes=n_classes,
            size=size,
            size_cat=size_cat,
            size_cont=size_cont,
            clinical_layers=clinical_layers,
            multimodal_odim=multimodal_odim,
            embed_size=embed_size,
            batch_norm=batch_norm,
            bilinear_scale_dim1=bilinear_scale_dim1,
            bilinear_scale_dim2=bilinear_scale_dim2,
            dropout=dropout,
            multires_aggregation=multires_aggregation,
            multimodal_aggregation=multimodal_aggregation,
            n_resolutions=n_resolutions,
        )
        self.model = TransMIL_Clinical_Multimodal(
            n_classes=n_classes, size=size, multimodal_odim=self.multimodal_odim
        )
