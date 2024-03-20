from typing import Dict, List

from pytorch_models.models.base import BaseClinicalMultimodalMILSurvModel
from pytorch_models.models.classification_clinical_multimodal.transmil import (
    TransMIL_Clinical_Multimodal,
)


class TransMIL_Clinical_Multimodal_PL_Surv(BaseClinicalMultimodalMILSurvModel):
    def __init__(
        self,
        config,
        n_classes,
        size,
        size_cat: int,
        size_cont: int,
        clinical_layers: List[int],
        multimodal_odim: int,
        embed_size: list = None,
        batch_norm: bool = True,
        bilinear_scale_dim1: int = 1,
        bilinear_scale_dim2: int = 1,
        loss_type="cox",
        multires_aggregation=None,
        multimodal_aggregation="concat",
        n_resolutions: int = 1,
        dropout=0.5,
    ):
        self.multires_aggregation = multires_aggregation
        super(TransMIL_Clinical_Multimodal_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
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

        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.model = TransMIL_Clinical_Multimodal(
            n_classes=n_classes,
            size=size,
            multimodal_odim=multimodal_odim,
        )
