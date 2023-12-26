# https://github.com/yanyongluan/MINNs
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from dotmap import DotMap
from pytorch_models.models.base import BaseMILModel_LNL
from pytorch_models.models.fair.utils import grad_reverse
from pytorch_models.utils.tensor import aggregate_features


def max_pooling(x):
    """Max Pooling to obtain aggregation.
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do max-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of max-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = torch.max(x, dim=0, keepdim=True)[0]
    return output


def mean_pooling(x):
    """Mean Pooling to obtain aggregation.
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do mean-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of mean-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = torch.mean(x, dim=0, keepdim=True)
    return output


def LSE_pooling(x):
    """LSE Pooling to obtain aggregation.
    Do LSE(log-sum-exp) pooling, like LSE(x1, x2, x3) = log(exp(x1)+exp(x2)+exp(x3)).
    Parameters
    ---------------------
    x : Tensor (N x d)
        Input data to do LSE-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    Return
    ---------------------
    output : Tensor (1 x d)
        Output of LSE-pooling,
        where d is dimension of instance feature
        (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    output = torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))
    return output


def choice_pooling(x, pooling_mode):
    """Choice the pooling mode
    Parameters
    -------------------
    x : Tensor (N x d)
        Input data to do MIL-pooling,
        where N is number of instances in one bag,
        and d is dimension of instance feature
        (when d = 1, x means instance scores; when d > 1, x means instance representations).
    pooling_mode : string
        Choice the pooling mode for MIL pooling.
    Return
    --------------------
    output : Tensor (1 x d)
            Output of MIL-pooling,
            where d is dimension of instance feature
            (when d = 1, the output means bag score; when d > 1, the output means bag representation).
    """
    if pooling_mode == "max":
        return max_pooling(x)
    if pooling_mode == "lse":
        return LSE_pooling(x)
    if pooling_mode in ["ave", "mean"]:
        return mean_pooling(x)


class ScorePooling(nn.Module):
    def __init__(self, D, C, pooling_mode="max"):
        super(ScorePooling, self).__init__()
        self.D = D
        self.C = C
        self.fc = nn.Linear(D, C)
        if C == 1:
            self.act = nn.Sigmoid()
        elif C > 2:
            raise NotImplementedError
            self.act = nn.Softmax()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return choice_pooling(x, self.pooling_mode), None


class FeaturePooling(nn.Module):
    def __init__(self, D, C, pooling_mode="max"):
        super(FeaturePooling, self).__init__()
        self.D = D
        self.C = C
        self.fc = nn.Linear(D, C)
        if C == 1:
            self.act = nn.Sigmoid()
        elif C > 2:
            self.act = nn.Softmax()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        x = choice_pooling(x, self.pooling_mode)
        x = self.fc(x)
        output = self.act(x)
        return output, x


class RC_Block(nn.Module):
    def __init__(self, pooling_mode="max"):
        super(RC_Block, self).__init__()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        return choice_pooling(x, self.pooling_mode)


class _mi_NET_LNL(nn.Module):
    def __init__(self, size=(384, 256, 128, 64), n_classes=1, pooling_mode="max"):
        super(_mi_NET_LNL, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        self.sp = ScorePooling(size[3], n_classes, pooling_mode=pooling_mode)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        output, _ = self.sp(x)
        return output, x


class mi_NET_LNL(nn.Module):
    def __init__(
        self, size=(384, 256, 128, 64), n_classes=1, n_classes_aux=1, pooling_mode="max"
    ):
        super(mi_NET_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.n_classes_aux = n_classes_aux
        if self.n_classes_aux == 2:
            self.n_classes_aux = 1
        self.pooling_mode = pooling_mode

        self.main_model = _mi_NET_LNL(size, n_classes, pooling_mode)
        self.aux_model = ScorePooling(
            size[3], self.n_classes_aux, pooling_mode=self.pooling_mode
        )

    def forward(self, x, is_adv=True):
        output, x = self.main_model(x)
        if not is_adv:
            x_aux = grad_reverse(x)
        else:
            x_aux = x
        output_aux, _ = self.aux_model(x_aux)
        return output, output_aux, None, None


class _MI_Net_LNL(nn.Module):
    def __init__(self, size=(384, 256, 128, 64), n_classes=1, pooling_mode="max"):
        super(_MI_Net_LNL, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        self.fp = FeaturePooling(size[3], n_classes, pooling_mode=pooling_mode)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        output, logits = self.fp(x)
        return output, x, logits


class MI_Net_LNL(nn.Module):
    def __init__(
        self, size=(384, 256, 128, 64), n_classes=1, n_classes_aux=1, pooling_mode="max"
    ):
        super(MI_Net_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.n_classes_aux = n_classes_aux
        if self.n_classes_aux == 2:
            self.n_classes_aux = 1
        self.pooling_mode = pooling_mode

        self.main_model = _MI_Net_LNL(size, n_classes, pooling_mode)
        self.aux_model = FeaturePooling(
            size[3], self.n_classes_aux, pooling_mode=self.pooling_mode
        )

    def forward(self, x, is_adv=True):
        output, x, logits = self.main_model(x)
        if not is_adv:
            x_aux = grad_reverse(x)
        else:
            x_aux = x
        output_aux, logits_aux = self.aux_model(x_aux)
        return output, output_aux, logits, logits_aux


class _MI_Net_DS_LNL(nn.Module):
    def __init__(self, size=(384, 256, 128, 64), n_classes=1, pooling_mode="max"):
        super(_MI_Net_DS_LNL, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.fp1 = FeaturePooling(size[1], n_classes, pooling_mode=pooling_mode)
        self.fp2 = FeaturePooling(size[2], n_classes, pooling_mode=pooling_mode)
        self.fp3 = FeaturePooling(size[3], n_classes, pooling_mode=pooling_mode)

    def forward(self, x, is_adv=True):
        x1 = self.fc1(x)
        x1 = self.dropout1(x1)
        preds1, logits1 = self.fp1(x1)

        x2 = self.fc2(x1)
        x2 = self.dropout2(x2)
        preds2, logits2 = self.fp2(x2)

        x3 = self.fc3(x2)
        x3 = self.dropout3(x3)
        preds3, logits3 = self.fp3(x3)

        return preds1, preds2, preds3, x1, x2, x3, logits1, logits2, logits3


class MI_Net_DS_LNL(nn.Module):
    def __init__(
        self, size=(384, 256, 128, 64), n_classes=1, n_classes_aux=1, pooling_mode="max"
    ):
        super(MI_Net_DS_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.n_classes_aux = n_classes_aux
        if self.n_classes_aux == 2:
            self.n_classes_aux = 1
        self.pooling_mode = pooling_mode

        self.main_model = _MI_Net_DS_LNL(size, n_classes, pooling_mode)

        self.fp1_aux = FeaturePooling(
            size[1], self.n_classes_aux, pooling_mode=self.pooling_mode
        )
        self.fp2_aux = FeaturePooling(
            size[2], self.n_classes_aux, pooling_mode=self.pooling_mode
        )
        self.fp3_aux = FeaturePooling(
            size[3], self.n_classes_aux, pooling_mode=self.pooling_mode
        )

    def forward(self, x, is_adv=True):
        preds1, preds2, preds3, x1, x2, x3, logits1, logits2, logits3 = self.main_model(
            x
        )

        if not is_adv:
            x1_aux = grad_reverse(x1)
            x2_aux = grad_reverse(x2)
            x3_aux = grad_reverse(x3)
        else:
            x1_aux = x1
            x2_aux = x2
            x3_aux = x3

        preds1_aux, logits1_aux = self.fp1_aux(x1_aux)
        preds2_aux, logits2_aux = self.fp2_aux(x2_aux)
        preds3_aux, logits3_aux = self.fp3_aux(x3_aux)

        return (
            torch.mean(torch.stack([preds1, preds2, preds3], dim=-1), dim=-1),
            torch.mean(
                torch.stack([preds1_aux, preds2_aux, preds3_aux], dim=-1), dim=-1
            ),
            torch.mean(torch.stack([logits1, logits2, logits3], dim=-1), dim=-1),
            torch.mean(
                torch.stack([logits1_aux, logits2_aux, logits3_aux], dim=-1), dim=-1
            ),
        )


class _MI_Net_RC_LNL(nn.Module):
    def __init__(self, size=(384, 128), n_classes=1, pooling_mode="max"):
        super(_MI_Net_RC_LNL, self).__init__()
        if len(size) > 2:
            p_size = size[:-1]
            size = size[-2:]
        else:
            p_size = size

        self.fc1 = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(p_size[i], p_size[i + 1]), nn.ReLU())
                for i in range(len(p_size) - 1)
            ]
        )
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[1]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[1], size[1]), nn.ReLU())

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.rc1 = RC_Block(pooling_mode=pooling_mode)
        self.rc2 = RC_Block(pooling_mode=pooling_mode)
        self.rc3 = RC_Block(pooling_mode=pooling_mode)

        self.fc = nn.Linear(size[1], n_classes)
        self.act = nn.Sigmoid() if n_classes == 1 else nn.Softmax()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.dropout1(x1)
        output1 = self.rc1(x1)

        x2 = self.fc2(x1)
        x2 = self.dropout2(x2)
        output2 = self.rc2(x2)

        x3 = self.fc3(x2)
        x3 = self.dropout3(x3)
        output3 = self.rc3(x3)

        feats = torch.sum(torch.stack([output1, output2, output3], dim=-1), dim=-1)
        logits = self.fc(feats)

        return self.act(logits), feats, logits


class MI_Net_RC_LNL(nn.Module):
    def __init__(
        self, size=(384, 128), n_classes=1, n_classes_aux=1, pooling_mode="max"
    ):
        super(MI_Net_RC_LNL, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.n_classes_aux = n_classes_aux
        if self.n_classes_aux == 2:
            self.n_classes_aux = 1
        self.pooling_mode = pooling_mode

        if len(size) > 2:
            p_size = size[:-1]
            size = size[-2:]
        else:
            p_size = size

        self.main_model = _MI_Net_RC_LNL(size, n_classes, pooling_mode)
        self.aux_model = nn.Linear(size[1], self.n_classes)
        self.act = nn.Sigmoid() if self.n_classes == 1 else nn.Softmax()

    def forward(self, x, is_adv=True):
        preds, feats, logits = self.main_model(x)
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
        n_classes_aux: int,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        pooling_mode="max",
        multires_aggregation: Union[None, str] = None,
    ):
        super(MINet_LNL_PL, self).__init__(
            config, n_classes=n_classes, n_classes_aux=n_classes_aux
        )
        assert self.n_classes > 0, "n_classes must be greater than 0"
        if self.n_classes > 2:
            raise NotImplementedError
        if self.n_classes == 2:
            self.n_classes = 1

        self.n_classes_aux = n_classes_aux
        if self.n_classes_aux == 2:
            self.n_classes_aux = 1

        if self.n_classes == 1:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        if self.n_classes_aux == 1:
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
                n_classes_aux=self.n_classes_aux,
                pooling_mode=self.pooling_mode,
            )
        elif self.config.model.classifier == "minet":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = MI_Net_LNL(
                size=size,
                n_classes=self.n_classes,
                n_classes_aux=self.n_classes_aux,
                pooling_mode=self.pooling_mode,
            )
        elif self.config.model.classifier == "minet_ds":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = MI_Net_DS_LNL(
                size=size,
                n_classes=self.n_classes,
                n_classes_aux=self.n_classes_aux,
                pooling_mode=self.pooling_mode,
            )
        elif self.config.model.classifier == "minet_rc":
            assert len(size) >= 2, "size must be a list of at least 2 integers"
            self.model = MI_Net_RC_LNL(
                size=size,
                n_classes=self.n_classes,
                n_classes_aux=self.n_classes_aux,
                pooling_mode=self.pooling_mode,
            )

    def forward(self, batch, is_adv=True, is_predict=False):
        # Batch
        features, target, target_aux = (
            batch["features"],
            batch["labels"],
            batch["labels_aux"],
        )

        # Prediction
        preds, preds_aux, _, _ = self._forward(features, is_adv)
        preds = preds.squeeze(dim=1)
        preds_aux = preds_aux.squeeze(dim=1)
        target = target.squeeze(dim=1)
        target_aux = target_aux.squeeze(dim=1)

        _loss = None
        _loss_aux_adv = None
        _loss_aux_mi = None
        if is_adv:
            if self.n_classes > 2:
                _loss = self.loss.forward(torch.log(preds), target)
            else:
                _loss = self.loss.forward(preds.float(), target.float())
            if len(preds_aux.shape) == 1 and preds_aux.shape[0] == 1:
                _loss_aux_adv = preds_aux * torch.log(preds_aux)
            else:
                _loss_aux_adv = torch.mean(
                    torch.sum(preds_aux * torch.log(preds_aux), 1)
                )
            loss = _loss + _loss_aux_adv * self.aux_lambda
        else:
            if self.n_classes_aux > 2:
                _loss_aux_mi = self.loss.forward(torch.log(preds_aux), target_aux)
            else:
                _loss_aux_mi = self.loss.forward(preds_aux.float(), target_aux.float())
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
        h: List[torch.Tensor] = [features[key] for key in features]
        h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
        if len(h.shape) == 3:
            h = h.squeeze(dim=0)
        return self.model.forward(h, is_adv=is_adv)

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
    y = torch.randint(0, 2, [1, 1])
    y_aux = torch.randint(0, 2, [1, 1])

    data = dict(features=dict(target=x), labels=y, labels_aux=y_aux, slide_name="tmp")

    model = MINet_LNL_PL(config, n_classes=1, n_classes_aux=1, size=[384, 256, 128, 64])

    o = model.forward(data, is_adv=True)
    o2 = model.forward(data, is_adv=False)

    print(o)
