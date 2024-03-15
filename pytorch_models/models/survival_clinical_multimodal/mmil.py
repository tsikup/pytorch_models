from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseClinicalMultimodalMILSurvModel
from pytorch_models.models.classification_clinical_multimodal.mmil import (
    MultipleMILTransformer_Clinical_Multimodal,
)
from pytorch_models.utils.tensor import aggregate_features


class MMIL_Clinical_Multimodal_PL_Surv(BaseClinicalMultimodalMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        loss_type="cox",
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
        self.multires_aggregation = multires_aggregation
        super(MMIL_Clinical_Multimodal_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            size_clinical=size_clinical,
            multires_aggregation=multires_aggregation,
            multimodal_aggregation=multimodal_aggregation,
            n_resolutions=n_resolutions,
        )

        assert len(size) == 2, "size must be a tuple of size 2"
        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.size = size
        self.num_msg = num_msg
        self.num_subbags = num_subbags
        self.grouping_mode = mode
        self.ape = ape
        self.num_layers = num_layers
        self.multires_aggregation = multires_aggregation

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
        features, event, survtime, coords = (
            batch["features"],
            batch["event"],
            batch["survtime"],
            None,
        )
        if self.grouping_mode == "coords":
            coords = batch["coords"]
        # Prediction
        logits = self._forward(features, coords)
        # Loss (on logits)
        loss = self.compute_loss(survtime, event, logits)
        if self.l1_reg_weight:
            loss = loss + self.l1_regularisation(l_w=self.l1_reg_weight)

        return {
            "event": event,
            "survtime": survtime,
            "preds": logits,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch: List[Dict[str, torch.Tensor]],
        coords_batch: List[torch.Tensor] = None,
    ):
        logits = []
        for idx, features in enumerate(features_batch):
            clinical = features.pop("clinical", None)
            h: List[torch.Tensor] = [features[key] for key in features]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 2:
                h = h.unsqueeze(dim=0)
            logits.append(
                self.model.forward(
                    h, clinical, coords_batch[idx] if coords_batch else None
                ).squeeze(dim=1)
            )
        return torch.stack(logits)
