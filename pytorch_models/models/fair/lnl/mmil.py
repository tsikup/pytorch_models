from typing import List, Tuple, Union

import torch
import torch.nn as nn
from dotmap import DotMap
from pytorch_models.models.base_fair import BaseMILModel_LNL
from pytorch_models.models.classification.mmil import MultipleMILTransformer
from pytorch_models.models.fair.lnl.base import _BaseLNL
from pytorch_models.models.fair.utils import grad_reverse
from pytorch_models.utils.tensor import aggregate_features


class MMIL_LNL(_BaseLNL):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        n_classes: int,
        num_msg: int = 1,
        num_subbags: int = 16,
        mode: str = "random",
        ape: bool = True,
        num_layers: int = 2,
    ):
        super(MMIL_LNL, self).__init__()
        self.main_model = MultipleMILTransformer(
            in_chans=in_chans,
            embed_dim=embed_dim,
            n_classes=n_classes,
            num_msg=num_msg,
            num_subbags=num_subbags,
            mode=mode,
            ape=ape,
            num_layers=num_layers,
        )
        self.aux_model = nn.Linear(embed_dim, n_classes)

    def forward(self, x, coords, is_adv=True):
        logits, feats = self.main_model(x, coords=coords, return_features=True)
        if not is_adv:
            feats_aux = grad_reverse(feats)
        else:
            feats_aux = feats
        logits_aux = self.aux_model(feats_aux)
        return logits, logits_aux


class MMIL_LNL_PL(BaseMILModel_LNL):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        size: Union[List[int], Tuple[int, int]] = None,
        num_msg: int = 1,
        num_subbags: int = 16,
        mode: str = "random",
        ape: bool = True,
        num_layers: int = 2,
        multires_aggregation: Union[None, str] = None,
    ):
        if n_classes == 1:
            n_classes = 2
        super(MMIL_LNL_PL, self).__init__(
            config, n_classes=n_classes, n_groups=n_groups
        )
        assert len(size) == 2, "size must be a tuple of size 2"

        self.size = size
        self.num_msg = num_msg
        self.num_subbags = num_subbags
        self.grouping_mode = mode
        self.ape = ape
        self.num_layers = num_layers
        self.multires_aggregation = multires_aggregation

        self.model = MMIL_LNL(
            in_chans=self.size[0],
            embed_dim=self.size[1],
            n_classes=self.n_classes,
            num_msg=self.num_msg,
            num_subbags=self.num_subbags,
            mode=self.grouping_mode,
            ape=self.ape,
            num_layers=self.num_layers,
        )

    def forward(self, batch, is_predict=False, is_adv=True):
        # Batch
        features, target, sensitive_attr, coords = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
            None,
        )
        if self.grouping_mode == "coords":
            coords = batch["coords"]
        # Prediction
        logits, logits_aux = self._forward(features, coords, is_adv)

        loss = None
        _loss = None
        _loss_aux_adv = None
        _loss_aux_mi = None
        if not is_predict:
            if is_adv:
                # Loss (on logits)
                _loss = self.loss.forward(logits, target.float())
                preds_aux = self.aux_act(logits_aux)
                _loss_aux_adv = torch.mean(
                    torch.sum(preds_aux * torch.log(preds_aux), 1)
                )

                loss = _loss + _loss_aux_adv * self.aux_lambda
            else:
                _loss_aux_mi = self.loss_aux.forward(logits_aux, sensitive_attr)
                loss = _loss_aux_mi

        # Sigmoid or Softmax activation
        preds = torch.nn.functional.softmax(logits, dim=1)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "main_loss": _loss,
            "aux_adv_loss": _loss_aux_adv,
            "aux_mi_loss": _loss_aux_mi,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch, coords_batch=None, is_adv=True):
        logits = []
        logits_aux = []
        for idx, singlePatientFeatures in enumerate(features_batch):
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits, _logits_aux = self.model.forward(
                h, coords=coords_batch[idx] if coords_batch else None, is_adv=is_adv
            )
            logits.append(_logits)
            logits_aux.append(_logits_aux)
        return torch.vstack(logits), torch.vstack(logits_aux)

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
