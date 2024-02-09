from typing import Dict, List, Tuple, Union

import torch
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.dsmil import DSMIL
from pytorch_models.utils.tensor import aggregate_features


class DSMIL_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config,
        loss_type="cox",
        size: Union[List[int], Tuple[int, int]] = (384, 128),
        n_classes=1,
        dropout=0.0,
        nonlinear=True,
        passing_v=False,
        multires_aggregation: Union[None, str] = None,
    ):
        self.multires_aggregation = multires_aggregation
        super(DSMIL_PL_Surv, self).__init__(
            config, n_classes=n_classes, loss_type=loss_type, size=size
        )

        assert len(size) >= 2, "size must be a tuple with 2 or more elements"
        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.model = DSMIL(
            size=size,
            n_classes=self.n_classes,
            dropout=dropout,
            nonlinear=nonlinear,
            passing_v=passing_v,
        )
