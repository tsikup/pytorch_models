from typing import Dict, List, Tuple, Union

from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.minet import get_minet_model


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
        n_resolutions: int = 1,
    ):
        self.multires_aggregation = multires_aggregation
        super(MINet_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
        )

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
