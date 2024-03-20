from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.mmil import MultipleMILTransformer
from pytorch_models.utils.tensor import aggregate_features


class MMIL_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        loss_type="cox",
        size: Union[List[int], Tuple[int, int]] = None,
        num_msg: int = 1,
        num_subbags: int = 16,
        mode: str = "random",
        ape: bool = True,
        num_layers: int = 2,
        multires_aggregation: Union[None, str] = None,
        n_resolutions: int = 1,
    ):
        self.multires_aggregation = multires_aggregation
        super(MMIL_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
        )

        assert len(size) == 2, "size must be a tuple of size 2"

        self.size = size
        self.num_msg = num_msg
        self.num_subbags = num_subbags
        self.grouping_mode = mode
        self.ape = ape
        self.num_layers = num_layers
        self.multires_aggregation = multires_aggregation

        self.model = MultipleMILTransformer(
            in_chans=self.size[0],
            embed_dim=self.size[1],
            n_classes=self.n_classes,
            num_msg=self.num_msg,
            num_subbags=self.num_subbags,
            mode=self.grouping_mode,
            ape=self.ape,
            num_layers=self.num_layers,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, event, survtime, coords = (
            batch["features"],
            batch["event"],
            batch["survtime"],
            None,
        )

        if self.loss_type != "cox_loss" or (
            len(survtime.shape) > 1 and survtime.shape[1] > 1
        ):
            survtime = torch.argmax(survtime, dim=1).view(self.batch_size, 1)

        if self.grouping_mode == "coords":
            coords = batch["coords"]
        # Prediction
        logits = self._forward(features, coords)
        logits = torch.sigmoid(logits)

        S, risk = None, None
        if self.n_classes > 1 and logits.shape[1] > 1:
            S = torch.cumprod(1 - logits, dim=1)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

        # Loss (on logits)
        loss = self.compute_loss(survtime, event, logits, S)
        if self.l1_reg_weight:
            loss = loss + self.l1_regularisation(l_w=self.l1_reg_weight)

        return {
            "event": event,
            "survtime": survtime,
            "hazards": logits,
            "risk": risk,
            "S": S,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch: List[Dict[str, torch.Tensor]],
        coords_batch: List[torch.Tensor] = None,
    ):
        logits = []
        for idx, features in enumerate(features_batch):
            h: List[torch.Tensor] = [features[key] for key in features]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 2:
                h = h.unsqueeze(dim=0)
            logits.append(
                self.model.forward(
                    h, coords_batch[idx] if coords_batch else None
                ).squeeze(dim=1)
            )
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

    model = MMIL_PL_Surv(
        config=config,
        size=[384, 128],
        n_classes=1,
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
    print(metric)
