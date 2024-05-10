import torch
from pytorch_models.models.base_fair import BaseMILModel_EnD
from pytorch_models.models.classification.clam import CLAM_SB


class CLAM_EnD_PL(BaseMILModel_EnD):
    def __init__(
        self,
        config,
        n_classes: int,
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
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super(CLAM_EnD_PL, self).__init__(
            config,
            n_classes=n_classes,
            multires_aggregation=multires_aggregation,
            size=size,
            n_resolutions=n_resolutions,
            alpha=alpha,
            beta=beta,
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
        logits, preds, _features, A, results_dict = self._forward(
            features,
            labels=target,
            instance_eval=self.instance_eval and not is_predict,
            return_features=True,
            attention_only=False,
        )

        loss = None
        ce_loss = None
        abs_loss = None
        if not is_predict:
            loss = 0.0
            # Loss (on logits)
            ce_loss = self.loss.forward(logits, target.reshape(-1))
            loss += ce_loss
            abs_loss = self.abs_regu(
                _features, target, sensitive_attr, self.alpha, self.beta
            )
            loss += abs_loss
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
            "loss": loss,
            "ce_loss": ce_loss,
            "abs_loss": abs_loss,
            "attention": A,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch,
        labels=None,
        instance_eval=False,
        return_features=True,
        attention_only=False,
    ):
        logits = []
        preds = []
        A = []
        features = []
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
                return_features=return_features,
                attention_only=attention_only,
            )
            logits.append(_logits)
            preds.append(_preds)
            A.append(_A)
            features.append(results_dict["features"])
            results_dict.append(_results_dict)

        return (
            torch.vstack(logits),
            torch.vstack(preds),
            torch.vstack(features),
            A,
            results_dict,
        )
