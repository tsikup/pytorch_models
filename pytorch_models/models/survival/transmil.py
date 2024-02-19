from typing import Dict, List

import torch
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.transmil import TransMIL
from pytorch_models.utils.tensor import aggregate_features


class TransMIL_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config,
        n_classes,
        loss_type="cox",
        size=(1024, 512),
        multires_aggregation=None,
    ):
        self.multires_aggregation = multires_aggregation
        super(TransMIL_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
        )

        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.model = TransMIL(n_classes=n_classes, size=size)
