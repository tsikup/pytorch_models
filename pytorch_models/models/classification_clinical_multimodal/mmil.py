# https://github.com/hustvl/MMIL-Transformer

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from pytorch_models.models.base import BaseClinincalMultimodalMILModel
from pytorch_models.models.classification.mmil import MultipleMILTransformer
from pytorch_models.models.multimodal.two_modalities import IntegrateTwoModalities
from pytorch_models.utils.tensor import aggregate_features


class MultipleMILTransformer_Clinical_Multimodal(MultipleMILTransformer):
    def __init__(
        self,
        in_chans: int = 1024,
        embed_dim: int = 512,
        size_clinical: int = None,
        n_classes: int = 2,
        num_msg: int = 1,
        num_subbags: int = 16,
        mode: str = "random",
        ape: bool = True,
        num_layers: int = 2,
        max_size: int = int(5e4),
        multimodal_aggregation: str = "concat",
        dropout: float = 0.5,
    ):
        super(MultipleMILTransformer_Clinical_Multimodal, self).__init__(
            in_chans,
            embed_dim,
            n_classes,
            num_msg,
            num_subbags,
            mode,
            ape,
            num_layers,
            max_size,
        )
        self.size_clinical = size_clinical
        self.multimodal_aggregation = multimodal_aggregation

        self.clinical_dense = nn.Sequential(
            nn.Linear(size_clinical, size_clinical),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.integration_model = IntegrateTwoModalities(
            dim1=embed_dim,
            dim2=size_clinical,
            odim=embed_dim,
            method=multimodal_aggregation,
            dropout=dropout,
        )

        if multimodal_aggregation == "concat":
            self.fc2 = nn.Linear(embed_dim + size_clinical, n_classes)
        elif multimodal_aggregation == "kron":
            self.fc2 = nn.Linear(embed_dim * size_clinical, n_classes)

    def head(self, x):
        return self.fc2(x)

    def forward(self, x, clinical, coords=None, mask_ratio=0, return_features=False):
        # ---> init
        x = self.fc1(x)
        if self.ape:
            x = x + self.absolute_pos_embed.expand(1, x.shape[0], self.embed_dim)

        if self.mode != "coords":
            x_groups = self.grouping_features(x)
        else:
            assert (
                coords is not None
            ), "Centroid coordinates of each patch must be given, since clustering mode is `coords`"
            x_groups = self.grouping_features(x, coords)

        msg_tokens = self.msg_tokens.expand(1, 1, self.msg_tokens_num, -1)
        msg_cls = self.msgcls_token
        x_groups = self.cat_msg2cluster_group(x_groups, msg_tokens)
        data = (msg_cls, x_groups, self.msg_tokens_num)
        # ---> feature forward
        for i in range(len(self.layers)):
            if i == 0:
                mr = mask_ratio
                data = self.layers[i](data, mr)
            else:
                mr = 0
                data = self.layers[i](data, mr)
        # ---> head
        msg_cls, _, _ = data
        msg_cls = msg_cls.view(1, self.embed_dim)

        # ---> multimodal
        clinical = self.clinical_dense(clinical)
        msg_cls = self.integration_model(msg_cls, clinical)

        logits = self.head(msg_cls)

        if return_features:
            return logits, msg_cls
        return logits


class MMIL_Clinical_Multimodal_PL(BaseClinincalMultimodalMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        size_clinical: int = None,
        num_msg: int = 1,
        num_subbags: int = 16,
        mode: str = "random",
        ape: bool = True,
        num_layers: int = 2,
        multires_aggregation: Union[None, str] = None,
        multimodal_aggregation: str = "concat",
        n_resolutions: int = 1,
        dropout: float = 0.5,
    ):
        if n_classes == 1:
            n_classes = 2
        super(MMIL_Clinical_Multimodal_PL, self).__init__(
            config,
            n_classes=n_classes,
            size=size,
            size_clinical=size_clinical,
            multires_aggregation=multires_aggregation,
            multimodal_aggregation=multimodal_aggregation,
            n_resolutions=n_resolutions,
        )
        assert len(size) == 2, "size must be a tuple of size 2"

        self.size = size
        self.num_msg = num_msg
        self.num_subbags = num_subbags
        self.grouping_mode = mode
        self.ape = ape
        self.num_layers = num_layers

        self.model = MultipleMILTransformer_Clinical_Multimodal(
            in_chans=self.size[0],
            embed_dim=self.size[1],
            size_clinical=size_clinical,
            n_classes=self.n_classes,
            num_msg=self.num_msg,
            num_subbags=self.num_subbags,
            mode=self.grouping_mode,
            ape=self.ape,
            num_layers=self.num_layers,
            multimodal_aggregation=multimodal_aggregation,
            dropout=dropout,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, coords = batch["features"], batch["labels"], None
        if self.grouping_mode == "coords":
            coords = batch["coords"]

        # Prediction
        logits = self._forward(features, coords)
        preds = F.softmax(logits, dim=1)

        loss = None
        if not is_predict:
            loss = self.loss.forward(logits, target.squeeze(dim=1))

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch, coords_batch=None):
        logits = []
        for idx, singlePatientFeatures in enumerate(features_batch):
            clinical = singlePatientFeatures.pop("clinical", None)
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits = self.model.forward(
                h, clinical, coords_batch[idx] if coords_batch else None
            )
            logits.append(_logits)
        return torch.vstack(logits)

    def _compute_metrics(self, preds, target, mode):
        if mode == "val":
            metrics = self.val_metrics
        elif mode == "train":
            metrics = self.train_metrics
        elif mode in ["eval", "test"]:
            metrics = self.test_metrics
        if self.n_classes in [1, 2]:
            metrics(
                preds,
                nn.functional.one_hot(target.view(-1), num_classes=self.n_classes),
            )
        else:
            metrics(preds, target.view(-1))
