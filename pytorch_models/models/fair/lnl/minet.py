# https://github.com/yanyongluan/MINNs
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from dotmap import DotMap
from pytorch_models.models.base_fair import BaseMILModel_LNL
from pytorch_models.models.classification.minet import (
    ScorePooling,
    FeaturePooling,
    RC_Block,
    MI_Net_DS,
    MI_Net,
    mi_NET,
    MI_Net_RC,
)
from pytorch_models.models.fair.utils import grad_reverse
from pytorch_models.utils.tensor import aggregate_features


class mi_NET_LNL(nn.Module):
    def __init__(
        self, size=(384, 256, 128, 64), n_classes=1, n_groups=1, pooling_mode="max"
    ):
        super(mi_NET_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.pooling_mode = pooling_mode

        self.main_model = mi_NET(
            size, n_classes, pooling_mode, return_features=True, return_preds=True
        )
        self.aux_model = ScorePooling(
            size[3], self.n_groups, pooling_mode=self.pooling_mode
        )

    def forward(self, x, is_adv=True):
        preds, x = self.main_model(x)
        if not is_adv:
            x_aux = grad_reverse(x)
        else:
            x_aux = x
        preds_aux = self.aux_model(x_aux)
        return preds, preds_aux, None, None


class MI_Net_LNL(nn.Module):
    def __init__(
        self, size=(384, 256, 128, 64), n_classes=1, n_groups=1, pooling_mode="max"
    ):
        super(MI_Net_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.pooling_mode = pooling_mode

        self.main_model = MI_Net(
            size, n_classes, pooling_mode, return_preds=True, return_features=True
        )
        self.aux_model = FeaturePooling(
            size[3], self.n_groups, pooling_mode=self.pooling_mode
        )

    def forward(self, x, is_adv=True):
        preds, logits, feats = self.main_model(x)
        if not is_adv:
            x_aux = grad_reverse(feats)
        else:
            x_aux = feats
        preds_aux, logits_aux = self.aux_model(x_aux)
        return preds, logits, logits_aux


class MI_Net_DS_LNL(nn.Module):
    def __init__(
        self, size=(384, 256, 128, 64), n_classes=1, n_groups=1, pooling_mode="max"
    ):
        super(MI_Net_DS_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.pooling_mode = pooling_mode

        self.main_model = MI_Net_DS(
            size, n_classes, pooling_mode, return_features=True, return_preds=False
        )

        self.fp1_aux = FeaturePooling(
            size[1], self.n_groups, pooling_mode=self.pooling_mode
        )
        self.fp2_aux = FeaturePooling(
            size[2], self.n_groups, pooling_mode=self.pooling_mode
        )
        self.fp3_aux = FeaturePooling(
            size[3], self.n_groups, pooling_mode=self.pooling_mode
        )

    def forward(self, x, is_adv=True):
        preds, logits, (x1, x2, x3) = self.main_model(x)

        if not is_adv:
            x1_aux = grad_reverse(x1)
            x2_aux = grad_reverse(x2)
            x3_aux = grad_reverse(x3)
        else:
            x1_aux = x1
            x2_aux = x2
            x3_aux = x3

        preds1_aux, logits1_aux, feats1_aux = self.fp1_aux(x1_aux)
        preds2_aux, logits2_aux, feats2_aux = self.fp2_aux(x2_aux)
        preds3_aux, logits3_aux, feats3_aux = self.fp3_aux(x3_aux)

        return (
            preds,
            torch.mean(
                torch.stack([preds1_aux, preds2_aux, preds3_aux], dim=-1), dim=-1
            ),
            logits,
            torch.mean(
                torch.stack([logits1_aux, logits2_aux, logits3_aux], dim=-1), dim=-1
            ),
        )


class MI_Net_RC_LNL(nn.Module):
    def __init__(self, size=(384, 128), n_classes=1, n_groups=1, pooling_mode="max"):
        super(MI_Net_RC_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.pooling_mode = pooling_mode

        if len(size) > 2:
            p_size = size[:-1]
            size = size[-2:]
        else:
            p_size = size

        self.main_model = MI_Net_RC(
            size, n_classes, pooling_mode, return_features=True, return_preds=True
        )
        self.aux_model = nn.Linear(size[1], self.n_classes)
        self.act = nn.Sigmoid() if self.n_classes == 1 else nn.Softmax()

    def forward(self, x, is_adv=True):
        preds, logits, feats = self.main_model(x)
        if not is_adv:
            feats_aux = grad_reverse(feats)
        else:
            feats_aux = feats
        logits_aux = self.aux_model(feats_aux)

        return preds, self.act(logits_aux), logits, logits_aux


class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        return torch.log(x)


class MINet_LNL_PL(BaseMILModel_LNL):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        pooling_mode="max",
        multires_aggregation: Union[None, str] = None,
    ):
        if n_classes == 2:
            n_classes = 1
        if n_groups == 2:
            n_groups = 1
        super(MINet_LNL_PL, self).__init__(
            config, n_classes=n_classes, n_groups=n_groups
        )
        self.n_groups = n_groups
        if self.n_classes > 2:
            raise NotImplementedError

        if self.n_classes == 1:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        if self.n_groups == 1:
            self.loss_aux = nn.BCELoss()
        else:
            self.loss_aux = nn.NLLLoss()

        self.dropout = dropout
        self.pooling_mode = pooling_mode
        self.multires_aggregation = multires_aggregation

        if self.config.model.classifier == "minet_naive":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = mi_NET_LNL(
                size=size,
                n_classes=self.n_classes,
                n_groups=self.n_groups,
                pooling_mode=self.pooling_mode,
            )
        elif self.config.model.classifier == "minet":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = MI_Net_LNL(
                size=size,
                n_classes=self.n_classes,
                n_groups=self.n_groups,
                pooling_mode=self.pooling_mode,
            )
        elif self.config.model.classifier == "minet_ds":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = MI_Net_DS_LNL(
                size=size,
                n_classes=self.n_classes,
                n_groups=self.n_groups,
                pooling_mode=self.pooling_mode,
            )
        elif self.config.model.classifier == "minet_rc":
            assert len(size) >= 2, "size must be a list of at least 2 integers"
            self.model = MI_Net_RC_LNL(
                size=size,
                n_classes=self.n_classes,
                n_groups=self.n_groups,
                pooling_mode=self.pooling_mode,
            )

    def forward(self, batch, is_adv=True, is_predict=False):
        # Batch
        features, target, target_aux = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )

        # Prediction
        preds, preds_aux, _, _ = self._forward(features, is_adv)

        _loss = None
        _loss_aux_adv = None
        _loss_aux_mi = None
        if is_adv:
            if self.n_classes > 2:
                _loss = self.loss.forward(torch.log(preds), target)
            else:
                _loss = self.loss.forward(preds.float(), target.float())
            _loss_aux_adv = torch.mean(torch.sum(preds_aux * torch.log(preds_aux), 1))
            loss = _loss + _loss_aux_adv * self.aux_lambda
        else:
            if self.n_groups > 2:
                _loss_aux_mi = self.loss_aux.forward(torch.log(preds_aux), target_aux)
            else:
                _loss_aux_mi = self.loss_aux.forward(
                    preds_aux.float(), target_aux.float()
                )
            loss = _loss_aux_mi

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "main_loss": _loss,
            "aux_adv_loss": _loss_aux_adv,
            "aux_mi_loss": _loss_aux_mi,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features, is_adv=True):
        preds = []
        preds_aux = []
        logits = []
        logits_aux = []
        for patientFeatures in features:
            h: List[torch.Tensor] = [patientFeatures[key] for key in patientFeatures]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            o = self.model.forward(h, is_adv=is_adv)
            preds.append(o[0])
            preds_aux.append(o[1])
            logits.append(o[2])
            logits_aux.append(o[3])
        return (
            torch.vstack(preds),
            torch.vstack(preds_aux),
            torch.vstack(logits),
            torch.vstack(logits_aux),
        )

    def configure_optimizers(self):
        if not self.config.model.classifier == "minet_ds":
            return super().configure_optimizers()

        optimizer_main = self._get_optimizer(self.model.main_model.parameters())
        optimizer_aux1 = self._get_optimizer(self.model.fp1_aux.parameters())
        optimizer_aux2 = self._get_optimizer(self.model.fp2_aux.parameters())
        optimizer_aux3 = self._get_optimizer(self.model.fp3_aux.parameters())

        if self.config.trainer.lr_scheduler is not None:
            scheduler_main = self._get_scheduler(optimizer_main)
            scheduler_aux1 = self._get_scheduler(optimizer_aux1)
            scheduler_aux2 = self._get_scheduler(optimizer_aux2)
            scheduler_aux3 = self._get_scheduler(optimizer_aux3)

            return [optimizer_main, optimizer_aux1, optimizer_aux2, optimizer_aux3], [
                scheduler_main,
                scheduler_aux1,
                scheduler_aux2,
                scheduler_aux3,
            ]
        else:
            pass
            return optimizer_main, optimizer_aux1, optimizer_aux2, optimizer_aux3

    def training_step(self, batch, batch_idx):
        if not self.config.model.classifier == "minet_ds":
            return super().training_step(batch, batch_idx)

        optimizers = self.optimizers()
        opt = optimizers[0]
        opt_aux1 = optimizers[1]
        opt_aux2 = optimizers[2]
        opt_aux3 = optimizers[3]

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
        opt_aux1.optimizer.zero_grad()
        opt_aux2.optimizer.zero_grad()
        opt_aux3.optimizer.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # *************************** #
        # Mutual Information AUX Task #
        # *************************** #
        output_mi = self.forward(batch, is_adv=False)
        aux_mi_loss = output_mi["aux_mi_loss"]
        opt.optimizer.zero_grad()
        opt_aux1.optimizer.zero_grad()
        opt_aux2.optimizer.zero_grad()
        opt_aux3.optimizer.zero_grad()
        self.manual_backward(aux_mi_loss)
        opt.step()
        opt_aux1.step()
        opt_aux2.step()
        opt_aux3.step()

        self._log_metrics(
            preds, target, loss, main_loss, aux_adv_loss, aux_mi_loss, "train"
        )
        return {
            "loss": loss,
            "main_loss": main_loss,
            "aux_adv_loss": aux_adv_loss,
            "aux_mi_loss": aux_mi_loss,
            "preds": preds,
            "target": target,
        }


if __name__ == "__main__":
    config = DotMap(
        {
            "num_classes": 1,
            "model": {
                "input_shape": 256,
                "classifier": "minet_ds",
            },
            "trainer": {
                "optimizer_params": {"lr": 1e-3},
                "batch_size": 1,
                "loss": ["ce"],
                "classes_loss_weights": None,
                "multi_loss_weights": None,
                "samples_per_class": None,
                "sync_dist": False,
                "l1_reg_weight": None,
                "l2_reg_weight": None,
            },
            "devices": {
                "nodes": 1,
                "gpus": 1,
            },
            "metrics": {"threshold": 0.5},
        }
    )

    x = torch.rand(10, 384)
    y = torch.randint(0, 2, [32, 1])
    y_aux = torch.randint(0, 2, [32, 1])

    data = dict(
        features=[dict(target=x) for _ in range(32)],
        labels=y,
        labels_group=y_aux,
        slide_name="tmp",
    )

    model = MINet_LNL_PL(config, n_classes=1, n_groups=1, size=[384, 256, 128, 64])

    o = model.forward(data, is_adv=True)
    o2 = model.forward(data, is_adv=False)

    print(o)
