from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.dtfd import DTFD
from pytorch_models.utils.survival import coxloss
from pytorch_models.utils.tensor import aggregate_features


class DTFD_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        K: int = 1,
        n_bags=3,
        dropout=0.25,
        multires_aggregation: Union[None, str] = None,
        l1_reg_weight: float = 3e-4,
    ):
        super(DTFD_PL_Surv, self).__init__(config, n_classes=n_classes)

        assert len(size) == 3, "size must be a tuple of size 3"
        assert self.n_classes == 1, "n_classes must be 1 for survival model"

        self.multires_aggregation = multires_aggregation

        self.lambda_reg = l1_reg_weight

        self.model = DTFD(
            size=size,
            n_classes=self.n_classes,
            K=K,
            n_bags=n_bags,
            dropout=dropout,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, censor, survtime = (
            batch["features"],
            batch["censor"],
            batch["survtime"],
        )

        # Prediction
        logits, sub_logits = self._forward(
            features
        )  ### batch_size, batch_size x numGroup x fs

        sub_censor = censor.repeat(1, sub_logits.shape[1]).reshape(
            -1
        )  ### batch_size x numGroup -> batch_size * numGroup x 1
        sub_survtime = survtime.repeat(1, sub_logits.shape[1]).reshape(
            -1
        )  ### batch_size x numGroup -> batch_size * numGroup x 1
        sub_logits = sub_logits.reshape(
            -1
        )  ### batch_size x numGroup -> batch_size * numGroup x 1

        loss_cox = coxloss(survtime, censor, logits)
        loss_cox += coxloss(sub_survtime, sub_censor, sub_logits)
        if hasattr(self, "lambda_reg"):
            loss = loss_cox + self.lambda_reg * self.l1_regularisation()
        else:
            loss = loss_cox

        return {
            "censor": censor.squeeze(),
            "survtime": survtime.squeeze(),
            "preds": logits.squeeze(),
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch: List[Dict[str, torch.Tensor]]):
        logits = []
        sub_logits = []
        for features in features_batch:
            h: List[torch.Tensor] = [features[key] for key in features]
            h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits, _sub_logits = self.model.forward(h)
            logits.append(_logits.squeeze())
            sub_logits.append(torch.stack(_sub_logits).squeeze())
        return torch.stack(logits), torch.stack(sub_logits)


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

    model = DTFD_PL_Surv(
        config=config,
        size=[384, 256, 128],
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
    print(out)