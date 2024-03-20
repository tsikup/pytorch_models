from typing import Dict, List

import torch
from pytorch_models.models.base import BaseClinicalMultimodalMILSurvModel
from pytorch_models.models.classification_clinical_multimodal.clam import (
    CLAM_SB_ClinincalMultimodal,
)
from pytorch_models.models.multimodal.two_modalities import IntegrateTwoModalities


class CLAM_Clinical_Multimodal_PL_Surv(BaseClinicalMultimodalMILSurvModel):
    def __init__(
        self,
        config,
        n_classes,
        size: List[int],
        size_cat: int,
        size_cont: int,
        clinical_layers: List[int],
        multimodal_odim: int,
        embed_size: list = None,
        batch_norm: bool = True,
        bilinear_scale_dim1: int = 1,
        bilinear_scale_dim2: int = 1,
        loss_type="cox",
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
            size_cat=size_cat,
            size_cont=size_cont,
            clinical_layers=clinical_layers,
            multimodal_odim=multimodal_odim,
            embed_size=embed_size,
            batch_norm=batch_norm,
            dropout=dropout,
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

        if isinstance(classifier_depth, int):
            c_size = size[classifier_depth]
        else:
            c_size = size[classifier_depth[0]]
        if multimodal_aggregation == "concat":
            self.multimodal_odim = c_size + clinical_layers[-1]
        elif multimodal_aggregation == "kron":
            self.multimodal_odim = c_size * clinical_layers[-1]

        if not self.multibranch:
            self.model = CLAM_SB_ClinincalMultimodal(
                gate=self.gate,
                size=self.size,
                multimodal_odim=self.multimodal_odim,
                dropout=self.dropout and self.dropout > 0,
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

        self.integration_model = IntegrateTwoModalities(
            dim1=size[classifier_depth],
            dim2=clinical_layers[-1],
            odim=self.multimodal_odim,
            method=multimodal_aggregation,
            bilinear_scale_dim1=bilinear_scale_dim1,
            bilinear_scale_dim2=bilinear_scale_dim2,
            dropout=dropout,
        )

        self.instance_integration_models = torch.nn.ModuleList(
            [
                IntegrateTwoModalities(
                    dim1=size[classifier_depth],
                    dim2=clinical_layers[-1],
                    odim=self.multimodal_odim,
                    method=multimodal_aggregation,
                    bilinear_scale_dim1=bilinear_scale_dim1,
                    bilinear_scale_dim2=bilinear_scale_dim2,
                    dropout=dropout,
                )
                for _ in range(self.n_classes)
            ]
        )

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
        A = []
        M = []
        instance_loss = []
        clinical_cat = []
        clinical_cont = []
        for idx, singlePatientFeatures in enumerate(features_batch):
            clinical_cat.append(singlePatientFeatures.pop("clinical_cat", None))
            clinical_cont.append(singlePatientFeatures.pop("clinical_cont", None))

        clinical = self.clinical_model(
            torch.stack(clinical_cat, dim=0), torch.stack(clinical_cont, dim=0)
        )

        for idx, features in enumerate(features_batch):
            _clinical = clinical[idx]
            feats = [features[f].squeeze() for f in features.keys()]

            _h, _A = self.model.forward_imaging(feats)

            _M, _results_dict = self.model.forward_instance_eval(
                _h,
                _A,
                _clinical,
                event_batch[idx] if event_batch is not None else None,
                self.instance_integration_models,
                instance_eval=instance_eval,
            )

            A.append(_A)
            M.append(_M)
            instance_loss.append(
                _results_dict["instance_loss"] if instance_eval else None
            )

        M = self.integration_model(torch.cat(M, dim=0), clinical)
        logits, preds, _ = self.model.forward(M)

        return (
            logits[:, 1],
            torch.mean(torch.stack(instance_loss)) if instance_eval else None,
        )
