from typing import Dict, List, Tuple, Union

import torch
from pytorch_models.models.base import BaseClinicalMultimodalMILSurvModel
from pytorch_models.models.classification_clinical_multimodal.clam import (
    CLAM_SB_ClinincalMultimodal,
)


class CLAM_Clinical_Multimodal_PL_Surv(BaseClinicalMultimodalMILSurvModel):
    def __init__(
        self,
        config,
        n_classes,
        loss_type="cox",
        size=None,
        size_clinical=None,
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
        n_resolutions: int = 1,
        multimodal_aggregation: str = "concat",
    ):
        super(CLAM_Clinical_Multimodal_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
            multimodal_aggregation=multimodal_aggregation,
            n_resolutions=n_resolutions,
        )

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
        self.size_clinical = size_clinical

        if not self.multibranch:
            self.model = CLAM_SB_ClinincalMultimodal(
                gate=self.gate,
                size=self.size,
                size_clinical=self.size_clinical,
                dropout=self.dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                instance_loss_fn=instance_loss,
                subtyping=self.subtyping,
                linear_feature=self.linear_feature,
                multires_aggregation=self.multires_aggregation,
                multimodal_aggregation=multimodal_aggregation,
                attention_depth=self.attention_depth,
                classifier_depth=self.classifier_depth,
                n_resolutions=self.n_resolutions,
            )
        else:
            raise NotImplementedError

    def forward(self, batch, is_predict=False):
        # Batch
        features, event, survtime = (
            batch["features"],
            batch["event"],
            batch["survtime"],
        )

        # Prediction
        logits, instance_loss = self._forward(
            features_batch=features,
            event_batch=event,
            instance_eval=self.instance_eval and not is_predict,
            return_features=False,
            attention_only=False,
        )
        logits = logits.unsqueeze(1)

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss = self.compute_loss(survtime, event, logits)
            if self.l1_reg_weight:
                loss = loss + self.l1_regularisation(self.l1_reg_weight)
            if self.instance_eval:
                loss = (
                    1 - self.instance_loss_weight
                ) * loss + self.instance_loss_weight * instance_loss

        return {
            "event": event,
            "preds": logits,
            "survtime": survtime,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch: List[Dict[str, torch.Tensor]],
        event_batch=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        logits = []
        instance_loss = []
        for idx, features in enumerate(features_batch):
            clinical = features.pop("clinical", None)
            feats = [features[f].squeeze() for f in features.keys()]
            _logits, _, _, _, _results_dict = self.model.forward(
                features=feats,
                clinical=clinical,
                label=event_batch[idx] if event_batch is not None else None,
                instance_eval=instance_eval,
                return_features=return_features,
                attention_only=attention_only,
            )
            logits.append(_logits.squeeze()[1])
            if instance_eval:
                instance_loss.append(_results_dict["instance_loss"])

        if instance_eval:
            return torch.stack(logits), torch.mean(torch.stack(instance_loss))
        else:
            return torch.stack(logits), None
