from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.minet import (
    MI_Net_DS,
    mi_NET,
    MI_Net,
    MI_Net_RC,
    get_minet_model,
)
from pytorch_models.utils.tensor import aggregate_features


class MINet_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        loss_type="cox",
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        pooling_mode="max",
        multires_aggregation: Union[None, str] = None,
    ):
        self.multires_aggregation = multires_aggregation
        super(MINet_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
        )

        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.dropout = dropout
        self.pooling_mode = pooling_mode
        self.multires_aggregation = multires_aggregation

        if self.config.model.classifier == "minet_naive":
            raise NotImplementedError
        self.model = get_minet_model(
            config,
            self.n_classes,
            size,
            pooling_mode,
            return_features=False,
            return_preds=False,
        )
