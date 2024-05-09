from typing import List, Union, Dict

import numpy as np
import torch
from dotmap import DotMap
from torch import nn

from pytorch_models.losses.gce import EMA, GeneralizedCELoss
from pytorch_models.losses.losses import get_loss
from pytorch_models.models.base import BaseMILModel, BaseModel
from pytorch_models.models.fair.group_dro.loss import GroupDROLoss
from pytorch_models.models.utils import (
    LinearWeightedTransformationSum,
    LinearWeightedSum,
)
from pytorch_models.utils.tensor import aggregate_features


class BaseMILModel_LAFTR(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        hidden_size: int,
        size: List[int],
        multires_aggregation: str,
        n_resolutions: int = 1,
        adversary_size: int = 32,
        model_var="eqodd",
        aud_steps: int = 1,
        class_coeff: float = 1.0,
        fair_coeff: float = 1.0,
        SensWeights=None,  # dataset.get_label_distribution("labels_group")
        LabelSensWeights=None,  # dataset.get_label_distribution(["labels", "labels_group"]).to_numpy().reshape(self.n_classes, self.n_groups)
        gradient_clip_value: float = 0.5,
        gradient_clip_algorithm: str = "norm",
    ):
        super(BaseMILModel_LAFTR, self).__init__(
            config,
            n_classes=n_classes,
            multires_aggregation=multires_aggregation,
            size=size,
            n_resolutions=n_resolutions,
        )
        self.adversary_size = adversary_size
        self.n_groups = n_groups
        assert n_groups == 1, "Only two groups supported. (binary)"
        assert model_var in [
            "dp",
            "eqodd",
            "eqopp0",
            "eqopp1",
        ], "Model variant not supported."
        self.aud_steps = aud_steps
        self.model_var = model_var
        self.class_coeff = class_coeff
        self.fair_coeff = fair_coeff
        self.A_weights = SensWeights
        self.YA_weights = LabelSensWeights
        self.hidden_size = hidden_size
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_value = gradient_clip_value

        self.discriminator = self._build_discriminator(hidden_size, adversary_size)

    def _build_discriminator(self, hidden_size, adversary_size):
        if self.model_var != "laftr-dp":
            adv_neurons = [hidden_size + self.n_groups, adversary_size, self.n_groups]
        else:
            adv_neurons = [hidden_size, adversary_size, self.n_groups]

        num_adversaries_layers = len(adv_neurons)
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        return nn.Sequential(
            *[
                nn.Linear(adv_neurons[i], adv_neurons[i + 1])
                for i in range(num_adversaries_layers - 1)
            ]
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )
        # Prediction
        logits, logits_adv = self._forward(features, target)
        # Loss (on logits)
        loss = None
        class_loss = None
        weighted_aud_loss = None
        if not is_predict:
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

        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "main_loss": class_loss,
            "weighted_aud_loss": weighted_aud_loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch, target=None):
        logits = []
        logits_adv = []
        for idx, patientFeatures in enumerate(features_batch):
            h: List[torch.Tensor] = [patientFeatures[key] for key in patientFeatures]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits, _features = self.model.forward(h, return_features=True)
            logits.append(_logits)
            if target is not None:
                _logits_d = []
                if self.model_var != "laftr-dp":
                    _f = torch.cat(
                        [
                            _features,
                            target[idx].float().view(-1, 1).to(self.device),
                        ],
                        axis=1,
                    )
                _f = self.discriminator(_f)
                # For discriminator loss
                _logits_d.append(torch.squeeze(_f, dim=1))
                logits_adv.append(torch.mean(torch.stack(_logits_d)))
        if target is not None:
            return (
                torch.vstack(logits),
                torch.vstack(logits_adv),
            )
        else:
            return torch.vstack(logits), None

    def l1_loss(self, y, y_logits):
        """Returns l1 loss"""
        y_hat = torch.sigmoid(y_logits)
        return torch.squeeze(torch.abs(y - y_hat))

    def get_weighted_aud_loss(self, L, Y, A, A_wts, YA_wts):
        """Returns weighted discriminator loss"""
        Y = Y[:, 0]
        if self.model_var == "dp":
            A0_wt = A_wts[0]
            A1_wt = A_wts[1]
            wts = A0_wt * (1 - A) + A1_wt * A
            wtd_L = L * torch.squeeze(wts)
        elif (
            self.model_var == "eqodd"
            or self.model_var == "eqopp0"
            or self.model_var == "eqopp1"
        ):
            Y0_A0_wt = YA_wts[0][0]
            Y0_A1_wt = YA_wts[0][1]
            Y1_A0_wt = YA_wts[1][0]
            Y1_A1_wt = YA_wts[1][1]
            if self.model_var == "eqodd":
                wts = (
                    Y0_A0_wt * (1 - A) * (1 - Y)
                    + Y1_A0_wt * (1 - A) * Y
                    + Y0_A1_wt * A * (1 - Y)
                    + Y1_A1_wt * A * Y
                )
            elif self.model_var == "eqopp0":
                wts = Y0_A0_wt * (1 - A) * (1 - Y) + Y0_A1_wt * A * (1 - Y)
            elif self.model_var == "eqopp1":
                wts = Y1_A0_wt * (1 - A) * Y + Y1_A1_wt * A * Y
            wtd_L = L * torch.squeeze(wts)
        else:
            raise Exception("Wrong model name")
            exit(0)
        return wtd_L

    def configure_optimizers(self):
        optimizer_main = self._get_optimizer(self.model.parameters())
        optimizer_disc = self._get_optimizer(self.discriminator.parameters())

        if self.config.trainer.lr_scheduler is not None:
            scheduler_main = self._get_scheduler(optimizer_main)
            scheduler_disc = self._get_scheduler(optimizer_disc)
            return [optimizer_main, optimizer_disc], [scheduler_main, scheduler_disc]
        else:
            return optimizer_main, optimizer_disc

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        opt = optimizers[0]
        opt_disc = optimizers[1]

        # ******************************** #
        # Main task + Adversarial AUX task #
        # ******************************** #
        output = self.forward(batch)
        target, preds, loss, main_loss, weighted_aud_loss = (
            output["target"],
            output["preds"],
            output["loss"],
            output["main_loss"],
            output["weighted_aud_loss"],
        )

        opt.optimizer.zero_grad()
        opt_disc.optimizer.zero_grad()

        self.manual_backward(loss, retain_graph=True)
        if self.gradient_clip_value is not None:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_value,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )

        for i in range(self.aud_steps):
            if i != self.aud_steps - 1:
                self.manual_backward(loss, retain_graph=True)
            else:
                self.manual_backward(loss)
            if self.gradient_clip_value is not None:
                self.clip_gradients(
                    opt_disc,
                    gradient_clip_val=self.gradient_clip_value,
                    gradient_clip_algorithm=self.gradient_clip_algorithm,
                )
            opt_disc.step()
        opt.step()

        lr_schedulers = self.lr_schedulers()
        sch = lr_schedulers[0]
        sch_adv = lr_schedulers[1]
        sch.step()
        sch_adv.step()

        self._log_metrics(
            preds,
            target,
            dict(
                loss=loss,
                main_loss=main_loss,
                weighted_aud_loss=weighted_aud_loss,
            ),
            "train",
        )
        return {
            "loss": loss,
            "main_loss": main_loss,
            "weighted_aud_loss": weighted_aud_loss,
            "preds": preds,
            "target": target,
        }

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        target, preds, loss, main_loss, weighted_aud_loss = (
            output["target"],
            output["preds"],
            output["loss"],
            output["main_loss"],
            output["weighted_aud_loss"],
        )

        self._log_metrics(
            preds,
            target,
            dict(
                loss=loss,
                main_loss=main_loss,
                weighted_aud_loss=weighted_aud_loss,
            ),
            "val",
        )
        return {
            "val_loss": loss,
            "val_main_loss": main_loss,
            "weighted_aud_loss": weighted_aud_loss,
            "val_preds": preds,
            "val_target": target,
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        target, preds, loss, main_loss, weighted_aud_loss = (
            output["target"],
            output["preds"],
            output["loss"],
            output["main_loss"],
            output["weighted_aud_loss"],
        )

        self._log_metrics(
            preds,
            target,
            dict(
                loss=loss,
                main_loss=main_loss,
                weighted_aud_loss=weighted_aud_loss,
            ),
            "test",
        )

        return {
            "test_loss": loss,
            "test_main_loss": main_loss,
            "weighted_aud_loss": weighted_aud_loss,
            "test_preds": preds,
            "test_target": target,
        }

    def predict_step(self, batch, batch_idx):
        output = self.forward(batch, is_predict=True)
        return output["preds"], batch["labels"], batch["slide_name"]


class BaseMILModel_EnD(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super(BaseMILModel_EnD, self).__init__(config, n_classes=n_classes)
        self.alpha = alpha
        self.beta = beta

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )
        # Prediction
        logits, _features = self._forward(features)
        # Loss (on logits)
        loss = None
        if not is_predict:
            loss = 0
            bce_loss = self.loss(
                logits, target.float() if logits.shape[1] == 1 else target.squeeze()
            )
            for _f in _features:
                abs_loss = self.abs_regu(
                    _f, target, sensitive_attr, self.alpha, self.beta
                )
                loss += abs_loss
            loss += bce_loss
        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch):
        logits = []
        features = []
        for patientFeatures in features_batch:
            h: List[torch.Tensor] = [patientFeatures[key] for key in patientFeatures]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits, _features = self.model.forward(h, return_features=True)
            if (not isinstance(_features, tuple)) and (not isinstance(_features, list)):
                _features = [_features]
            logits.append(_logits)
            features.append(_features)
        return (
            torch.vstack(logits),
            [torch.vstack([_f[i] for _f in features]) for i in range(len(features[0]))],
        )

    def abs_orthogonal_blind(self, output, gram, target_labels, bias_labels):
        # For each discriminatory class, orthogonalize samples
        bias_classes = torch.unique(bias_labels)
        orthogonal_loss = torch.tensor(0.0).to(output.device)
        M_tot = 0.0

        for bias_class in bias_classes:
            bias_mask = (bias_labels == bias_class).type(torch.float).view(-1, 1)
            bias_mask = torch.tril(
                torch.mm(bias_mask, torch.transpose(bias_mask, 0, 1)), diagonal=-1
            )
            M = bias_mask.sum()
            M_tot += M

            if M > 0:
                orthogonal_loss += torch.abs(torch.sum(gram * bias_mask))

        if M_tot > 0:
            orthogonal_loss /= M_tot
        return orthogonal_loss

    def abs_parallel(self, gram, target_labels, bias_labels):
        # For each target class, parallelize samples belonging to
        # different discriminatory classes
        target_classes = torch.unique(target_labels)
        bias_classes = torch.unique(bias_labels)

        parallel_loss = torch.tensor(0.0).to(gram.device)
        M_tot = 0.0

        for target_class in target_classes:
            class_mask = (target_labels == target_class).type(torch.float).view(-1, 1)

            for idx, bias_class in enumerate(bias_classes):
                bias_mask = (bias_labels == bias_class).type(torch.float).view(-1, 1)

                for other_bias_class in bias_classes[idx:]:
                    if other_bias_class == bias_class:
                        continue

                    other_bias_mask = (
                        (bias_labels == other_bias_class).type(torch.float).view(-1, 1)
                    )
                    mask = torch.tril(
                        torch.mm(
                            class_mask * bias_mask,
                            torch.transpose(class_mask * other_bias_mask, 0, 1),
                        ),
                        diagonal=-1,
                    )
                    M = mask.sum()
                    M_tot += M

                    if M > 0:
                        parallel_loss -= torch.sum((1.0 + gram) * mask * 0.5)
        if M_tot > 0:
            parallel_loss = 1.0 + (parallel_loss / M_tot)
        return parallel_loss

    def abs_regu(self, feat, target_labels, bias_labels, alpha=1.0, beta=1.0, sum=True):
        D = feat
        if len(D.size()) > 2:
            D = D.view(-1, np.prod((D.size()[1:])))

        gram_matrix = torch.tril(torch.mm(D, torch.transpose(D, 0, 1)), diagonal=-1)
        # not really needed, just for safety for approximate repr
        gram_matrix = torch.clamp(gram_matrix, -1, 1.0)

        zero = torch.tensor(0.0).to(target_labels.device)
        # TODO: Check if this is correct because gram_matrix is much larger than 1.0,
        #  which is the max value
        R_ortho = (
            self.abs_orthogonal_blind(D, gram_matrix, target_labels, bias_labels)
            if alpha != 0
            else zero
        )
        R_parallel = (
            self.abs_parallel(gram_matrix, target_labels, bias_labels)
            if beta != 0
            else zero
        )

        if sum:
            return alpha * R_ortho + beta * R_parallel
        return alpha * R_ortho, beta * R_parallel


class BaseMILModel_LearningFromFailure(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        labels: List[int],
    ):
        super(BaseMILModel_LearningFromFailure, self).__init__(
            config, n_classes=n_classes
        )
        assert self.n_classes > 1

        self.loss = None
        self.loss = get_loss(
            config_losses=config.trainer.loss,
            n_classes=self.n_classes,
            classes_loss_weights=config.trainer.classes_loss_weights,
            multi_loss_weights=config.trainer.multi_loss_weights,
            samples_per_cls=config.trainer.samples_per_class,
            reduction="none",
        )
        self.bias_loss = GeneralizedCELoss()

        self.sample_loss_ema_b = EMA(torch.LongTensor(labels), alpha=0.7)
        self.sample_loss_ema_d = EMA(torch.LongTensor(labels), alpha=0.7)

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, index = (
            batch["features"],
            batch["labels"],
            batch["index"],
        )
        # Prediction
        logits_b, logits_d = self._forward(features)

        # Loss
        loss = None
        loss_per_sample_b = None
        loss_per_sample_d = None
        if not is_predict:
            loss_b = self.loss(logits_b, target.squeeze()).cpu().detach()
            loss_d = self.loss(logits_d, target.squeeze()).cpu().detach()
            loss_per_sample_b = loss_b
            loss_per_sample_d = loss_d

            # EMA sample loss
            self.sample_loss_ema_b.update(loss_b, index)
            self.sample_loss_ema_d.update(loss_d, index)

            # class-wise normalize
            loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()
            loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()

            target_cpu = target.cpu()

            for c in range(self.n_classes):
                class_index = np.where(target_cpu == c)[0]
                max_loss_b = self.sample_loss_ema_b.max_loss(c)
                max_loss_d = self.sample_loss_ema_d.max_loss(c)
                loss_b[class_index] /= max_loss_b
                loss_d[class_index] /= max_loss_d

            # re-weighting based on loss value / generalized CE for biased model
            loss_weight = loss_b / (loss_b + loss_d + 1e-8)
            loss_b_update = self.bias_loss(logits_b, target.squeeze())
            loss_d_update = self.loss(logits_d, target.squeeze()) * loss_weight.to(
                logits_d.device
            )
            loss = loss_b_update.mean() + loss_d_update.mean()

        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits_d.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits_d, dim=1)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "loss_b": loss_per_sample_b.mean(),
            "loss_d": loss_per_sample_d.mean(),
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch):
        logits_b = []
        logits_d = []
        for singlePatientFeatures in features_batch:
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits_b = self.model_b.forward(h)
            _logits_d = self.model_d.forward(h)
            logits_b.append(_logits_b)
            logits_d.append(_logits_d)
        return torch.vstack(logits_b), torch.vstack(logits_d)

    def configure_optimizers(self):
        optimizer_b = self._get_optimizer(self.model_b.parameters())
        optimizer_d = self._get_optimizer(self.model_d.parameters())

        if self.config.trainer.lr_scheduler is not None:
            scheduler_b = self._get_scheduler(optimizer_b)
            scheduler_d = self._get_scheduler(optimizer_d)
            return [optimizer_b, optimizer_d], [scheduler_b, scheduler_d]
        else:
            return optimizer_b, optimizer_d

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        opt_b = optimizers[0]
        opt_d = optimizers[1]

        # ******************************** #
        # Main task + Adversarial AUX task #
        # ******************************** #
        output = self.forward(batch)
        target, preds, loss, loss_d, loss_b = (
            output["target"],
            output["preds"],
            output["loss"],
            output["loss_d"],
            output["loss_b"],
        )

        opt_d.optimizer.zero_grad()
        opt_b.optimizer.zero_grad()
        self.manual_backward(loss)
        opt_d.step()
        opt_b.step()

        lr_schedulers = self.lr_schedulers()
        sch_b = lr_schedulers[0]
        sch_d = lr_schedulers[1]
        sch_b.step()
        sch_d.step()

        self._log_metrics(
            preds,
            target,
            dict(loss=loss, loss_b=loss_b, loss_d=loss_d),
            "train",
        )
        return {
            "loss": loss,
            "loss_b": loss_b,
            "loss_d": loss_d,
            "preds": preds,
            "target": target,
        }

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        target, preds, loss, loss_d, loss_b = (
            output["target"],
            output["preds"],
            output["loss"],
            output["loss_d"],
            output["loss_b"],
        )

        self._log_metrics(
            preds,
            target,
            dict(
                loss=loss,
                loss_d=loss_d,
                loss_b=loss_b,
            ),
            "val",
        )
        return {
            "val_loss": loss,
            "val_loss_b": loss_b,
            "val_loss_d": loss_d,
            "val_preds": preds,
            "val_target": target,
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        target, preds, loss, loss_d, loss_b = (
            output["target"],
            output["preds"],
            output["loss"],
            output["loss_d"],
            output["loss_b"],
        )

        self._log_metrics(
            preds, target, dict(loss=loss, loss_b=loss_b, loss_d=loss_d), "test"
        )

        return {
            "test_loss": loss,
            "test_loss_b": loss_b,
            "test_loss_d": loss_d,
            "test_preds": preds,
            "test_target": target,
        }


class BaseMILModel_GroupDRO(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,  # dataset.n_groups
        group_counts: List[int],  # dataset.get_label_distribution("labels_group")
        alpha: float = 0.2,
        gamma: float = 0.1,
        generalization_adjustment: str = "0",
        is_robust: bool = True,
        multires_aggregation: Union[Dict[str, str], str, None] = None,
        size: List[int] = None,
        n_resolutions: int = 1,
    ):
        super(BaseMILModel_GroupDRO, self).__init__(
            config,
            n_classes=n_classes,
            multires_aggregation=multires_aggregation,
            size=size,
            n_resolutions=n_resolutions,
        )

        adjustments = [float(c) for c in generalization_adjustment.split(",")]
        assert len(adjustments) in (1, n_groups)
        if len(adjustments) == 1:
            adjustments = np.array(adjustments * n_groups)
        else:
            adjustments = np.array(adjustments)

        self.loss = None
        self.loss = get_loss(
            config_losses=config.trainer.loss,
            n_classes=self.n_classes,
            classes_loss_weights=config.trainer.classes_loss_weights,
            multi_loss_weights=config.trainer.multi_loss_weights,
            samples_per_cls=config.trainer.samples_per_class,
            reduction="none",
        )

        self.dro_loss = GroupDROLoss(
            criterion=self.loss,
            is_robust=is_robust,
            n_groups=n_groups,
            group_counts=np.array(group_counts),
            alpha=alpha,
            gamma=gamma,
            adj=adjustments,
            min_var_weight=0,
            step_size=0.01,
            normalize_loss=False,
            btl=False,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )
        # Prediction
        logits = self._forward(features)
        # Loss (on logits)
        loss = None
        if not is_predict:
            loss = self.dro_loss(logits, target, sensitive_attr)
        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }


class BaseMILModel_DomainIndependent(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        fairness_mode: str,
    ):
        super(BaseMILModel_DomainIndependent, self).__init__(
            config, n_classes=n_classes
        )
        # TODO: Log change of n_classes and n_groups
        if self.n_classes == 1:
            self.n_classes = 2
        self.n_groups = n_groups
        if self.n_groups == 1:
            self.n_groups = 2
        self.n_classes = self.n_classes * self.n_groups
        self.fairness_mode = fairness_mode
        assert self.fairness_mode in [
            "domain_independent",
            "domain_discriminative",
        ], "Fairness mode must be either 'domain_independent' or 'domain_discriminative'."
        assert isinstance(
            self.loss, torch.nn.CrossEntropyLoss
        ), "Only CE loss supported for DomainIndependent or DomainDiscriminative training."

    def _criterion(self, logits, target, sensitive_attr):
        domain_label = sensitive_attr.squeeze()
        class_num = self.n_classes // self.n_groups
        if self.fairness_mode == "domain_independent":
            _logits = []
            for idx in range(logits.shape[0]):
                _logits.append(
                    logits[
                        idx,
                        class_num
                        * domain_label[idx] : class_num
                        * (domain_label[idx] + 1),
                    ]
                )
            _logits = torch.vstack(_logits).to(logits.device)
            return self.loss.forward(_logits, target.squeeze())
        elif self.fairness_mode == "domain_discriminative":
            nd_way_target = torch.zeros_like(logits)
            nd_way_target[
                :, class_num * domain_label : class_num * (domain_label + 1)
            ] = 1
            nd_way_target = nd_way_target.reshape(class_num, self.n_groups)
            nd_way_target = nd_way_target * target
            nd_way_target = nd_way_target.reshape(-1)
            return self.loss.forward(logits, nd_way_target.float())

    def _inference(self, logits):
        class_num = self.n_classes // self.n_groups
        preds = torch.nn.functional.softmax(logits, dim=1)
        preds = preds.reshape(-1, self.n_groups, class_num)
        preds = preds.sum(dim=1)
        return preds

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )
        # Prediction
        logits = self._forward(features)
        # Loss (on logits)
        loss = None
        if not is_predict:
            loss = self._criterion(logits, target, sensitive_attr)
        # Sigmoid or Softmax activation
        preds = self._inference(logits)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }


class BaseMILModel_LNL(BaseModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        multires_aggregation: str,
        aux_lambda: float = 0.1,
        size: List[int] = None,
        n_resolutions: int = 1,
        gradient_clip_value: float = None,
        gradient_clip_algorithm: str = "norm",
    ):
        super(BaseMILModel_LNL, self).__init__(
            config,
            n_classes=n_classes,
            in_channels=None,
            segmentation=False,
            automatic_optimization=False,
        )
        self.automatic_optimization = False
        self.n_groups = n_groups
        if self.n_groups == 1:
            self.aux_act = torch.nn.Sigmoid()
            self.loss_aux = get_loss(["bce"])
        else:
            self.aux_act = torch.nn.Softmax(dim=1)
            self.loss_aux = get_loss(["ce"])
        self.aux_lambda = aux_lambda
        self.multires_aggregation = multires_aggregation
        self.n_resolutions = n_resolutions
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_value = gradient_clip_value

        if self.config.model.classifier != "clam":
            if self.multires_aggregation == "linear":
                assert size is not None
                self.linear_agg = LinearWeightedTransformationSum(
                    size[0], self.n_resolutions
                )
            elif self.multires_aggregation == "linear_2":
                assert size is not None
                self.linear_agg = LinearWeightedSum(size[0], self.n_resolutions)

    def forward(self, batch, is_predict=False, is_adv=True):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )
        # Prediction
        logits, logits_aux = self._forward(features, is_adv)

        loss = None
        _loss = None
        _loss_aux_adv = None
        _loss_aux_mi = None
        if not is_predict:
            if is_adv:
                # Loss (on logits)
                if self.n_classes == 2:
                    _loss = self.loss.forward(logits, target.reshape(-1))
                else:
                    _loss = self.loss.forward(logits, target.float())
                preds_aux = self.aux_act(logits_aux)
                _loss_aux_adv = torch.mean(
                    torch.sum(preds_aux * torch.log(preds_aux + 1e-5), 1)
                )

                loss = _loss + _loss_aux_adv * self.aux_lambda
            else:
                if self.n_classes == 2:
                    _loss_aux_mi = self.loss_aux.forward(
                        logits_aux, sensitive_attr.reshape(-1)
                    )
                else:
                    _loss_aux_mi = self.loss_aux.forward(logits_aux, sensitive_attr)
                loss = _loss_aux_mi

        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "main_loss": _loss,
            "aux_adv_loss": _loss_aux_adv,
            "aux_mi_loss": _loss_aux_mi,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch, is_adv=True):
        logits = []
        logits_aux = []
        for singlePatientFeatures in features_batch:
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits, _logits_aux = self.model.forward(h, is_adv=is_adv)
            logits.append(_logits)
            logits_aux.append(_logits_aux)
        return torch.vstack(logits), torch.vstack(logits_aux)

    def configure_optimizers(self):
        optimizer_main = self._get_optimizer(self.model.main_model.parameters())
        optimizer_aux = self._get_optimizer(self.model.aux_model.parameters())

        if self.config.trainer.lr_scheduler is not None:
            scheduler_main = self._get_scheduler(optimizer_main)
            scheduler_aux = self._get_scheduler(optimizer_aux)
            return [optimizer_main, optimizer_aux], [scheduler_main, scheduler_aux]
        else:
            return optimizer_main, optimizer_aux

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        opt = optimizers[0]
        opt_aux = optimizers[1]

        # ******************************** #
        # Main task + Adversarial AUX task #
        # ******************************** #
        output = self.forward(batch, is_adv=True)
        target, preds, loss, main_loss, aux_adv_loss = (
            output["target"],
            output["preds"],
            output["loss"],
            output["main_loss"],
            output["aux_adv_loss"],
        )

        opt.optimizer.zero_grad()
        opt_aux.optimizer.zero_grad()
        self.manual_backward(loss)
        if self.gradient_clip_value is not None:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_value,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )
            self.clip_gradients(
                opt_aux,
                gradient_clip_val=self.gradient_clip_value,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )
        opt.step()
        opt_aux.step()

        # *************************** #
        # Mutual Information AUX Task #
        # *************************** #
        output_mi = self.forward(batch, is_adv=False)
        aux_mi_loss = output_mi["aux_mi_loss"]
        # opt.optimizer.zero_grad()
        opt_aux.optimizer.zero_grad()
        self.manual_backward(aux_mi_loss)
        if self.gradient_clip_value is not None:
            self.clip_gradients(
                opt_aux,
                gradient_clip_val=self.gradient_clip_value,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
            )
        # opt.step()
        opt_aux.step()

        lr_schedulers = self.lr_schedulers()
        sch = lr_schedulers[0]
        sch_aux = lr_schedulers[1]
        sch.step()
        sch_aux.step()

        self._log_metrics(
            preds,
            target,
            dict(
                loss=loss,
                main_loss=main_loss,
                aux_adv_loss=aux_adv_loss,
                aux_mi_loss=aux_mi_loss,
            ),
            "train",
        )
        return {
            "loss": loss,
            "main_loss": main_loss,
            "aux_adv_loss": aux_adv_loss,
            "aux_mi_loss": aux_mi_loss,
            "preds": preds,
            "target": target,
        }

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch, is_adv=True)
        target, preds, loss, main_loss, aux_adv_loss = (
            output["target"],
            output["preds"],
            output["loss"],
            output["main_loss"],
            output["aux_adv_loss"],
        )
        output_mi = self.forward(batch, is_adv=False)
        aux_mi_loss = output_mi["aux_mi_loss"]

        self._log_metrics(
            preds,
            target,
            dict(
                loss=loss,
                main_loss=main_loss,
                aux_adv_loss=aux_adv_loss,
                aux_mi_loss=aux_mi_loss,
            ),
            "val",
        )
        return {
            "val_loss": loss,
            "val_main_loss": main_loss,
            "val_aux_adv_loss": aux_adv_loss,
            "val_aux_mi_loss": aux_mi_loss,
            "val_preds": preds,
            "val_target": target,
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch, is_adv=True)
        target, preds, loss, main_loss, aux_adv_loss = (
            output["target"],
            output["preds"],
            output["loss"],
            output["main_loss"],
            output["aux_adv_loss"],
        )
        output_mi = self.forward(batch, is_adv=False)
        aux_mi_loss = output_mi["aux_mi_loss"]

        self._log_metrics(
            preds,
            target,
            dict(
                loss=loss,
                main_loss=main_loss,
                aux_adv_loss=aux_adv_loss,
                aux_mi_loss=aux_mi_loss,
            ),
            "test",
        )

        return {
            "test_loss": loss,
            "test_main_loss": main_loss,
            "test_aux_adv_loss": aux_adv_loss,
            "test_aux_mi_loss": aux_mi_loss,
            "test_preds": preds,
            "test_target": target,
        }

    def predict_step(self, batch, batch_idx):
        output = self.forward(batch, is_predict=True, is_adv=True)
        return output["preds"], batch["labels"], batch["slide_name"]
