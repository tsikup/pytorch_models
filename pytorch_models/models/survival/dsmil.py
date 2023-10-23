from typing import Dict, List, Tuple, Union

import torch
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.dsmil import DSMIL
from pytorch_models.utils.survival import coxloss
from pytorch_models.utils.tensor import aggregate_features


class DSMIL_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config,
        size: Union[List[int], Tuple[int, int]] = (384, 128),
        n_classes=1,
        dropout=0.0,
        nonlinear=True,
        passing_v=False,
        multires_aggregation: Union[None, str] = None,
    ):
        self.multires_aggregation = multires_aggregation
        super(DSMIL_PL_Surv, self).__init__(config, n_classes=n_classes)

        assert len(size) >= 2, "size must be a tuple with 2 or more elements"
        if self.n_classes == 2:
            self.n_classes = 1

        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"
        self.lambda_reg = 3e-4

        self.model = DSMIL(
            size=size,
            n_classes=self.n_classes,
            dropout=dropout,
            nonlinear=nonlinear,
            passing_v=passing_v,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, censors, survtimes = (
            batch["features"],
            batch["censor"],
            batch["survtime"],
        )
        # Prediction
        classes, logits, A, B = self._forward(features)
        # Loss (on logits)
        loss_cox = coxloss(survtimes, censors, logits, logits.device)
        loss_reg = self.l1_regularisation()
        if hasattr(self, "lambda_reg"):
            loss = loss_cox + self.lambda_reg * loss_reg
        else:
            loss = loss_cox

        return {
            "censors": censors,
            "survtimes": survtimes,
            "preds": logits,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch: List[Dict[str, torch.Tensor]]):
        classes = []
        logits = []
        A = []
        B = []
        for features in features_batch:
            h: List[torch.Tensor] = [features[key] for key in features]
            h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _classes, _logits, _A, _B = self.model(h)
            classes += [_classes.squeeze()]
            logits += [_logits.squeeze()]
            A += [_A.squeeze()]
            B += [_B.squeeze()]
        return (
            torch.stack(classes, dim=0),
            torch.stack(logits, dim=0).unsqueeze(dim=1),
            torch.stack(A, dim=0),
            torch.stack(B, dim=0),
        )


if __name__ == "__main__":
    from dotmap import DotMap
    from pytorch_models.utils.survival import (
        CoxLogRank,
        AccuracyCox,
        CIndex,
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

    model = DSMIL_PL_Surv(
        config=config,
        size=(384, 128),
        n_classes=1,
        dropout=0.5,
        nonlinear=True,
        passing_v=False,
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
    metric.update(out["preds"], out["censors"], out["survtimes"])
    metric = metric.compute()
    print(out)
