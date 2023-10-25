from typing import Dict, List, Tuple, Union

import torch
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.transmil import TransMIL
from pytorch_models.utils.tensor import aggregate_features


class TransMIL_Features_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config,
        n_classes,
        size=(1024, 512),
        multires_aggregation=None,
        l1_reg_weight: float = 3e-4,
    ):
        self.multires_aggregation = multires_aggregation
        super(TransMIL_Features_PL_Surv, self).__init__(config, n_classes=n_classes)

        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"
        self.lambda_reg = l1_reg_weight

        self.model = TransMIL(n_classes=n_classes, size=size)

    def _forward(self, features_batch: List[Dict[str, torch.Tensor]]):
        logits = []
        for features in features_batch:
            h = [features[key] for key in features]
            h = aggregate_features(h, method=self.multires_aggregation)
            logits.append(self.model(h.unsqueeze(dim=0))["logits"].squeeze(dim=1))
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

    model = TransMIL_Features_PL_Surv(
        config=config,
        size=(384, 128),
        n_classes=1,
        multires_aggregation="mean",
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
    print(metric)
