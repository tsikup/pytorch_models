from typing import Dict, List, Tuple, Union

import torch
from pytorch_models.utils.survival import coxloss
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.clam import CLAM_SB, CLAM_MB


class CLAM_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config,
        n_classes,
        size=None,
        gate: bool = True,
        dropout=False,
        k_sample: int = 8,
        instance_eval: bool = False,
        instance_loss: str = "ce",
        instance_loss_weight: float = 0.3,
        subtyping: bool = False,
        multibranch=False,
        multires_aggregation=None,
        linear_feature: bool = False,
        attention_depth=None,
        classifier_depth=None,
        l1_reg_weight: float = 3e-4,
    ):
        super(CLAM_PL_Surv, self).__init__(config, n_classes=n_classes)

        self.size = size
        self.dropout = dropout
        self.gate = gate
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.instance_eval = instance_eval
        self.instance_loss_weight = instance_loss_weight
        self.multires_aggregation = multires_aggregation
        self.multibranch = multibranch
        self.attention_depth = attention_depth
        self.classifier_depth = classifier_depth
        self.linear_feature = linear_feature

        self.lambda_reg = l1_reg_weight

        if not self.multibranch:
            self.model = CLAM_SB(
                gate=self.gate,
                size=self.size,
                dropout=self.dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=instance_loss,
                subtyping=self.subtyping,
                linear_feature=self.linear_feature,
                multires_aggregation=self.multires_aggregation,
                attention_depth=self.attention_depth,
                classifier_depth=self.classifier_depth,
            )
        else:
            self.model = CLAM_MB(
                gate=self.gate,
                size=self.size,
                dropout=self.dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=instance_loss,
                subtyping=self.subtyping,
                linear_feature=self.linear_feature,
                multires_aggregation=self.multires_aggregation,
                attention_depth=self.attention_depth,
                classifier_depth=self.classifier_depth,
            )

    def forward(self, batch, is_predict=False):
        # Batch
        features, censor, survtime = (
            batch["features"],
            batch["censor"],
            batch["survtime"],
        )

        # Prediction
        logits, instance_loss = self._forward(
            features_batch=features,
            censor_batch=censor,
            instance_eval=self.instance_eval and not is_predict,
            return_features=False,
            attention_only=False,
        )
        logits = logits.unsqueeze(1)

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss_cox = coxloss(survtime, censor, logits)
            if hasattr(self, "lambda_reg"):
                loss = loss_cox + self.lambda_reg * self.l1_regularisation()
            else:
                loss = loss_cox
            if self.instance_eval:
                loss = (
                    1 - self.instance_loss_weight
                ) * loss + self.instance_loss_weight * instance_loss

        return {
            "censor": censor,
            "preds": logits,
            "survtime": survtime,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch: List[Dict[str, torch.Tensor]],
        censor_batch=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        logits = []
        instance_loss = []
        for idx, features in enumerate(features_batch):
            h = features["features"]
            h_context = (
                features["features_context"]
                if self.multires_aggregation is not None
                and "features_context" in features
                else None
            )

            _logits, _, _, _, _results_dict = self.model.forward(
                h=h.squeeze(),
                h_context=h_context.squeeze() if h_context is not None else None,
                label=censor_batch[idx] if censor_batch is not None else None,
                instance_eval=instance_eval,
                return_features=return_features,
                attention_only=attention_only,
            )
            logits.append(_logits.squeeze()[1])
            instance_loss.append(_results_dict["instance_loss"])

        return torch.stack(logits), torch.mean(torch.stack(instance_loss))


if __name__ == "__main__":
    from dotmap import DotMap
    from pytorch_models.utils.survival import (
        CoxLogRank,
        AccuracyCox,
        CIndex,
        cindex_lifeline,
    )

    x = [
        {"features": torch.rand(100, 384), "features_context": torch.rand(100, 384)}
        for _ in range(32)
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

    model = CLAM_PL_Surv(
        config=config,
        n_classes=2,
        size=[384, 256, 128],
        gate=True,
        instance_eval=True,
        instance_loss="ce",
        multires_aggregation={"features": "mean", "attention": None},
        classifier_depth=1,
        attention_depth=1,
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
    print(metric)