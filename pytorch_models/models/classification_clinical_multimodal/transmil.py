import numpy as np
import torch
from pytorch_models.models.base import BaseClinincalMultimodalMILModel
from pytorch_models.models.classification.transmil import TransMIL
from pytorch_models.models.multimodal.two_modalities import IntegrateTwoModalities
from torch import nn


class TransMIL_Clinical_Multimodal(TransMIL):
    def __init__(
        self, n_classes, size, size_clinical, multimodal_aggregation, dropout=0.5
    ):
        super(TransMIL_Clinical_Multimodal, self).__init__(n_classes, size)
        self.size_clinical = size_clinical
        self.multimodal_aggregation = multimodal_aggregation

        self.clinical_dense = nn.Sequential(
            nn.Linear(size_clinical, size_clinical),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.integration_model = IntegrateTwoModalities(
            dim1=size[-1],
            dim2=size_clinical,
            odim=size[-1],
            method=multimodal_aggregation,
            dropout=dropout,
        )

        if multimodal_aggregation == "concat":
            self._fc2 = nn.Linear(size[1] + size_clinical, self.n_classes)
        elif multimodal_aggregation == "kron":
            self._fc2 = nn.Linear(size[1] * size_clinical, self.n_classes)

    def forward(
        self, h: torch.Tensor, clinical: torch.Tensor, return_features=False
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

        # ---->multimodal
        clinical = self.clinical_dense(clinical)
        h = self.integration_model(h, clinical)

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        if return_features:
            return logits, h
        return logits


class TransMIL_Clinical_Multimodal_PL(BaseClinincalMultimodalMILModel):
    def __init__(
        self,
        config,
        n_classes,
        size=(1024, 512),
        size_clinical=None,
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
            size_clinical=size_clinical,
            multires_aggregation=multires_aggregation,
            multimodal_aggregation=multimodal_aggregation,
            n_resolutions=n_resolutions,
        )
        self.model = TransMIL_Clinical_Multimodal(
            n_classes=n_classes,
            size=size,
            size_clinical=size_clinical,
            multimodal_aggregation=multimodal_aggregation,
            dropout=dropout,
        )
