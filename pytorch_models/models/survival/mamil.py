from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.mamil import MultiAttentionMIL
from pytorch_models.utils.tensor import aggregate_features


class MAMIL_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        multires_aggregation: Union[None, str] = None,
    ):
        super(MAMIL_PL_Surv, self).__init__(config, n_classes=n_classes)
        assert (
            len(size) >= 2
        ), "size must be a tuple of (n_features, layer1_size, layer2_size, ...)"
        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.multires_aggregation = multires_aggregation
        self.dropout = dropout

        self.model = MultiAttentionMIL(self.n_classes, size, use_dropout=self.dropout)

    def _forward(self, features_batch: List[Dict[str, torch.Tensor]]):
        logits = []
        for features in features_batch:
            h: List[torch.Tensor] = [features[key] for key in features]
            h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            logits.append(self.model.forward(h)[0].squeeze(dim=1))
        return torch.stack(logits)


if __name__ == "__main__":
    from dotmap import DotMap
    from pytorch_models.utils.survival import (
        AccuracyCox,
        CIndex,
        CoxLogRank,
        cindex_lifeline,
    )

    x = [
        {"target": torch.rand(100, 384), "x10": torch.rand(100, 384)} for _ in range(32)
    ]
    survtime = torch.rand(32, 1) * 100
    censor = torch.randint(0, 2, (32, 1))

    config = DotMap(
        {
            "num_classes": 1,
            "model": {"input_shape": 384},
            # "trainer.optimizer_params.lr"
            "trainer": {
                "optimizer_params": {"lr": 1e-3},
                "batch_size": 1,
                "loss": ["ce"],
                "classes_loss_weights": None,
                "multi_loss_weights": None,
                "samples_per_class": None,
                "sync_dist": False,
            },
            "devices": {
                "nodes": 1,
                "gpus": 1,
            },
            "metrics": {"threshold": 0.5},
        }
    )

    model = MAMIL_PL_Surv(
        config=config,
        size=[384, 256, 128],
        n_classes=1,
        multires_aggregation=None,
    )

    # run model
    batch = {
        "features": x,
        "censor": censor,
        "survtime": survtime,
        "slide_name": ["lol" for _ in range(32)],
    }

    out = model.forward(batch)
    metric = CIndex()
    metric.update(out["preds"], out["censor"], out["survtime"])
    metric = metric.compute()
    print(out)
