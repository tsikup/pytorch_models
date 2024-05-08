# https://github.com/mdsatria/MultiAttentionMIL
from typing import List, Tuple, Union

import torch.nn as nn
from dotmap import DotMap
from pytorch_models.models.base_fair import BaseMILModel_LNL
from pytorch_models.models.classification.mamil import MultiAttentionMIL
from pytorch_models.models.fair.lnl.base import _BaseLNL


class MultiAttentionMIL_LNL(_BaseLNL):
    def __init__(
        self, num_classes=1, size=(384, 128), use_dropout=False, n_dropout=0.4
    ):
        super(MultiAttentionMIL_LNL, self).__init__()
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.n_dropout = n_dropout
        self.D = size[-1]

        self.main_model = MultiAttentionMIL(
            num_classes=self.num_classes,
            size=size,
            use_dropout=self.use_dropout,
            n_dropout=self.n_dropout,
        )
        self.aux_model = nn.Linear(self.D, self.num_classes)


class MAMIL_LNL_PL(BaseMILModel_LNL):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        aux_lambda=0.1,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        multires_aggregation: Union[None, str] = None,
        gradient_clip_value: float = 0.5,
        gradient_clip_algorithm: str = "norm",
    ):
        if n_groups == 2:
            n_groups = 1
        if n_classes == 2:
            n_classes = 1
        super(MAMIL_LNL_PL, self).__init__(
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

        self.dropout = dropout

        self.model = MultiAttentionMIL_LNL(
            self.n_classes, size, use_dropout=self.dropout
        )
