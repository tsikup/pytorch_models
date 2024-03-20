from typing import List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.mamil import MultiAttentionMIL


class MAMIL_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        loss_type="cox",
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        multires_aggregation: Union[None, str] = None,
        n_resolutions: int = 1,
    ):
        super(MAMIL_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
        )
        assert (
            len(size) >= 2
        ), "size must be a tuple of (n_features, layer1_size, layer2_size, ...)"

        self.multires_aggregation = multires_aggregation
        self.dropout = dropout

        self.model = MultiAttentionMIL(self.n_classes, size, use_dropout=self.dropout)
