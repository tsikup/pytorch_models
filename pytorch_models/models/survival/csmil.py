from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.csmil import CSMIL
from pytorch_models.utils.tensor import aggregate_features


class CSMIL_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        loss_type="cox",
        size: int = 1024,
        cluster_num: int = 1,
        multires_aggregation: Union[None, str] = None,
        n_resolutions: int = 1,
    ):
        super(CSMIL_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=[size],
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
        )

        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.multires_aggregation = multires_aggregation

        self.model = CSMIL(
            cluster_num=cluster_num, feature_size=size, n_classes=self.n_classes
        )

    def _forward(self, features_batch: List[Dict[str, torch.Tensor]]):
        logits = []
        for features in features_batch:
            h = [features[key] for key in features]
            if self.multires_aggregation == "concat":
                h = torch.stack(h, dim=-1)
            elif self.multires_aggregation == "linear":
                h = [self.linear_agg[i](h[i]) for i in range(len(h))]
                h = self._aggregate_multires_features(
                    h,
                    method="sum",
                    is_attention=False,
                )
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
                h = h.unsqueeze(dim=-1)
            h = h.unsqueeze(dim=-1)
            if len(h.shape) == 4:
                h = h.unsqueeze(dim=0)
            # h -> [n_clusters, n_patches, n_features, n_resolutions, 1]
            logits.append(self.model.forward(h)[0])
        return torch.stack(logits).squeeze(dim=1)


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
    event = torch.randint(0, 2, (32, 1))

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

    model = CSMIL_PL_Surv(
        config=config,
        n_classes=1,
        size=384,
        multires_aggregation="mean",
    )

    # run model
    batch = {
        "features": x,
        "event": event,
        "survtime": survtime,
        "slide_name": ["lol" for _ in range(32)],
    }

    out = model.forward(batch)
    metric = CIndex()
    metric.update(out["preds"], out["event"], out["survtime"])
    metric = metric.compute()
    print(out)
