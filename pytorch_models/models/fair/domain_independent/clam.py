import torch
from pytorch_models.models.base_fair import BaseMILModel_DomainIndependent
from pytorch_models.models.classification.clam import CLAM_SB


class CLAM_DomainIndependent_PL(BaseMILModel_DomainIndependent):
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
        fairness_mode: str = "domain_independent",
    ):
        super(CLAM_DomainIndependent_PL, self).__init__(
            config,
            n_classes=n_classes,
            n_groups=n_groups,
            fairness_mode=fairness_mode,
            multires_aggregation=multires_aggregation,
            size=size,
            n_resolutions=n_resolutions,
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

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )

        # Prediction
        logits, _, _, A, results_dict = self._forward(
            features,
            labels=target,
            instance_eval=self.instance_eval and not is_predict,
            return_features=False,
            attention_only=False,
        )

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss = self._criterion(logits, target, sensitive_attr)
            if self.instance_eval:
                instance_loss = torch.mean(
                    torch.stack([r["instance_loss"] for r in results_dict])
                )
                loss = (
                    1 - self.instance_loss_weight
                ) * loss + self.instance_loss_weight * instance_loss

        preds = self._inference(logits)
        if self.class_num in [1, 2]:
            preds = preds[:, 1]
            preds = torch.unsqueeze(preds, dim=1)

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "attention": A,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch,
        labels=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        logits = []
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

            _logits, _, _, _A, _results_dict = self.model.forward(
                features=feats,
                label=label,
                instance_eval=instance_eval,
                return_features=return_features,
                attention_only=attention_only,
            )
            logits.append(_logits)
            A.append(_A)
            results_dict.append(_results_dict)

        return torch.vstack(logits), None, None, A, results_dict
