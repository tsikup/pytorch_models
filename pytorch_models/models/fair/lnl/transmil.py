# https://github.com/mdsatria/MultiAttentionMIL
from typing import Union

import torch.nn as nn
from dotmap import DotMap
from pytorch_models.models.base_fair import BaseMILModel_LNL
from pytorch_models.models.classification.transmil import TransMIL
from pytorch_models.models.fair.lnl.base import _BaseLNL


class TransMIL_LNL(_BaseLNL):
    def __init__(self, n_classes, size=(1024, 512)):
        super(TransMIL_LNL, self).__init__()
        self.main_model = TransMIL(n_classes=n_classes, size=size)
        self.aux_model = nn.Linear(size[1], n_classes)


class TransMIL_LNL_PL(BaseMILModel_LNL):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        aux_lambda: float = 1.0,
        size=(1024, 512),
        multires_aggregation: Union[None, str] = None,
        gradient_clip_value: float = 0.5,
        gradient_clip_algorithm: str = "norm",
    ):
        if n_classes == 2:
            n_classes = 1
        if n_groups == 2:
            n_groups = 1
        super(TransMIL_LNL_PL, self).__init__(
            config,
            n_classes=n_classes,
            n_groups=n_groups,
            aux_lambda=aux_lambda,
            multires_aggregation=multires_aggregation,
            size=size,
            gradient_clip_value=gradient_clip_value,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        assert (
            len(size) >= 2
        ), "size must be a tuple of (n_features, layer1_size, layer2_size, ...)"

        self.model = TransMIL_LNL(n_classes=n_classes, size=size)
