from typing import List

import torch
from pytorch_models.models.base_fair import BaseMILModel_LAFTR
from pytorch_models.models.classification.clam import CLAM_SB


class CLAM_LAFTR_PL(BaseMILModel_LAFTR):
    def __init__(
        self,
        config,
        n_classes: int,
        n_groups: int,
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
        n_resolutions: int = 1,
        SensWeights=None,
        LabelSensWeights=None,
        adversary_size: int = 32,
        model_var: str = "eqodd",
        aud_steps: int = 1,
        class_coeff: float = 1.0,
        fair_coeff: float = 1.0,
        gradient_clip_value: float = 0.5,
        gradient_clip_algorithm: str = "norm",
    ):
        super(CLAM_LAFTR_PL, self).__init__(
            config,
            n_classes=n_classes,
            n_groups=n_groups,
            hidden_size=size[classifier_depth],
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
            adversary_size=adversary_size,
            model_var=model_var,
            aud_steps=aud_steps,
            class_coeff=class_coeff,
            fair_coeff=fair_coeff,
            SensWeights=SensWeights,
            LabelSensWeights=LabelSensWeights,
            gradient_clip_value=gradient_clip_value,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        self.size = size
        self.dropout = dropout
        self.gate = gate
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.instance_eval = instance_eval
        self.instance_loss_weight = instance_loss_weight
        self.multibranch = multibranch
        self.attention_depth = attention_depth
        self.classifier_depth = classifier_depth
        self.linear_feature = linear_feature

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
                n_resolutions=n_resolutions,
            )
        else:
            raise NotImplementedError

        self.hidden_size = self.model.classifier_size
        self.discriminator = self._build_discriminator(
            self.hidden_size, self.adversary_size
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )

        # Prediction
        logits, preds, logits_adv, _, A, results_dict = self._forward(
            features,
            labels=target,
            instance_eval=self.instance_eval and not is_predict,
        )

        loss = None
        class_loss = None
        weighted_aud_loss = None
        if not is_predict:
            # Loss (on logits)
            class_loss = self.class_coeff * self.loss(
                logits, target.float() if logits.shape[1] == 1 else target.squeeze()
            )
            aud_loss = -self.fair_coeff * self.l1_loss(sensitive_attr, logits_adv)
            weighted_aud_loss = self.get_weighted_aud_loss(
                aud_loss,
                target,
                sensitive_attr,
                self.A_weights,
                self.YA_weights,
            )
            weighted_aud_loss = torch.mean(weighted_aud_loss)
            loss = class_loss + weighted_aud_loss
            if self.instance_eval:
                instance_loss = torch.mean(
                    torch.stack([r["instance_loss"] for r in results_dict])
                )
                loss = (
                    1 - self.instance_loss_weight
                ) * loss + self.instance_loss_weight * instance_loss

        if self.n_classes in [1, 2]:
            preds = preds[:, 1]
            preds = torch.unsqueeze(preds, dim=1)

        return {
            "target": target,
            "preds": preds,
            "attention": A,
            "loss": loss,
            "main_loss": class_loss,
            "weighted_aud_loss": weighted_aud_loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch,
        labels=None,
        instance_eval=False,
    ):
        logits = []
        logits_adv = []
        preds = []
        A = []
        results_dict = []
        for idx, singlePatientFeatures in enumerate(features_batch):
            feats = [
                singlePatientFeatures[f].squeeze() for f in singlePatientFeatures.keys()
            ]

            if labels is not None:
                label = labels[idx]
                label = label.squeeze(dim=0) if label is not None else None
            else:
                label = None

            _logits, _preds, _, _A, _results_dict = self.model.forward(
                features=feats,
                label=label,
                instance_eval=instance_eval,
                return_features=True,
            )

            _features = _results_dict["features"]
            if label is not None:
                if self.model_var != "laftr-dp":
                    _features = torch.cat(
                        [
                            _features,
                            label.float().view(-1, 1).to(self.device),
                        ],
                        axis=1,
                    )
                # For discriminator loss
                logits_adv.append(torch.squeeze(self.discriminator(_features), dim=1))

            logits.append(_logits)
            preds.append(_preds)
            A.append(_A)
            results_dict.append(_results_dict)

        return (
            torch.vstack(logits),
            torch.vstack(preds),
            torch.vstack(logits_adv),
            None,
            A,
            results_dict,
        )
