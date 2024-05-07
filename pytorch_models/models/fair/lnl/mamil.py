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
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        multires_aggregation: Union[None, str] = None,
    ):
        if n_groups == 2:
            n_groups = 1
        if n_classes == 2:
            n_classes = 1
        super(MAMIL_LNL_PL, self).__init__(
            config,
            n_classes=n_classes,
            n_groups=n_groups,
            multires_aggregation=multires_aggregation,
            size=size,
        )
        assert (
            len(size) >= 2
        ), "size must be a tuple of (n_features, layer1_size, layer2_size, ...)"

        self.dropout = dropout

        self.model = MultiAttentionMIL_LNL(
            self.n_classes, size, use_dropout=self.dropout
        )


if __name__ == "__main__":
    import torch

    config = DotMap(
        {
            "num_classes": 1,
            "model": {
                "input_shape": 256,
                "classifier": "minet_ds",
            },
            "trainer": {
                "optimizer_params": {"lr": 1e-3},
                "batch_size": 1,
                "loss": ["ce"],
                "classes_loss_weights": None,
                "multi_loss_weights": None,
                "samples_per_class": None,
                "sync_dist": False,
                "l1_reg_weight": None,
                "l2_reg_weight": None,
            },
            "devices": {
                "nodes": 1,
                "gpus": 1,
            },
            "metrics": {"threshold": 0.5},
        }
    )

    x = torch.rand(10, 384)
    y = torch.randint(0, 2, [32, 1])
    y_aux = torch.randint(0, 2, [32, 1])

    data = dict(
        features=[dict(target=x) for _ in range(32)],
        labels=y,
        labels_group=y_aux,
        slide_name="tmp",
    )

    model = MAMIL_LNL_PL(config, n_classes=1, n_groups=1, size=[384, 256, 128, 64])

    o = model.forward(data, is_adv=True)
    o2 = model.forward(data, is_adv=False)

    print(o)
