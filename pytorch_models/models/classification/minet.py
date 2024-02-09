# https://github.com/yanyongluan/MINNs
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from dotmap import DotMap
from pytorch_models.models.base import BaseMILModel
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
        elif C > 1:
            raise NotImplementedError
            self.act = nn.Softmax()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return choice_pooling(x, self.pooling_mode)


class FeaturePooling(nn.Module):
    def __init__(self, D, C, pooling_mode="max"):
        super(FeaturePooling, self).__init__()
        self.D = D
        self.C = C
        self.fc = nn.Linear(D, C)
        if C == 1:
            self.act = nn.Sigmoid()
        elif C > 1:
            self.act = nn.Softmax()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        feats = choice_pooling(x, self.pooling_mode)
        logits = self.fc(feats)
        preds = self.act(logits)
        return preds, logits, feats


class RC_Block(nn.Module):
    def __init__(self, pooling_mode="max"):
        super(RC_Block, self).__init__()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        return choice_pooling(x, self.pooling_mode)


class mi_NET(nn.Module):
    def __init__(
        self,
        size=(384, 256, 128, 64),
        n_classes=1,
        pooling_mode="max",
        return_features=False,
        return_preds=True,
    ):
        super(mi_NET, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.pooling_mode = pooling_mode
        self.return_features = return_features
        self.return_preds = return_preds
        assert return_preds

        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        self.sp = ScorePooling(size[3], self.n_classes, pooling_mode=self.pooling_mode)

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        preds = self.sp(x)
        if self.return_features:
            return preds, x
        else:
            return preds


class MI_Net(nn.Module):
    def __init__(
        self,
        size=(384, 256, 128, 64),
        n_classes=1,
        pooling_mode="max",
        return_features=False,
        return_preds=True,
    ):
        super(MI_Net, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.pooling_mode = pooling_mode
        self.return_features = return_features
        self.return_preds = return_preds

        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        self.fp = FeaturePooling(
            size[3], self.n_classes, pooling_mode=self.pooling_mode
        )

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        preds, logits, feats = self.fp(x)
        if self.return_features and self.return_preds:
            return preds, logits, feats
        elif self.return_features:
            return logits, feats
        elif self.return_preds:
            return preds, logits
        else:
            return logits


class MI_Net_DS(nn.Module):
    def __init__(
        self,
        size=(384, 256, 128, 64),
        n_classes=1,
        pooling_mode="max",
        return_features=False,
        return_preds=True,
    ):
        super(MI_Net_DS, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.pooling_mode = pooling_mode
        self.return_features = return_features
        self.return_preds = return_preds

        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.fp1 = FeaturePooling(size[1], n_classes, pooling_mode=self.pooling_mode)
        self.fp2 = FeaturePooling(size[2], n_classes, pooling_mode=self.pooling_mode)
        self.fp3 = FeaturePooling(size[3], n_classes, pooling_mode=self.pooling_mode)

    def forward(self, x, **kwargs):
        x1 = self.fc1(x)
        x1 = self.dropout1(x1)
        preds1, logits1, feats1 = self.fp1(x1)

        x2 = self.fc2(x1)
        x2 = self.dropout2(x2)
        preds2, logits2, feats2 = self.fp2(x2)

        x3 = self.fc3(x2)
        x3 = self.dropout3(x3)
        preds3, logits3, feats3 = self.fp3(x3)

        preds = torch.mean(torch.stack([preds1, preds2, preds3], dim=-1), dim=-1)
        logits = torch.mean(torch.stack([logits1, logits2, logits3], dim=-1), dim=-1)
        feats = feats1, feats2, feats3

        if self.return_features and self.return_preds:
            return preds, logits, feats
        elif self.return_features:
            return logits, feats
        elif self.return_preds:
            return preds, logits
        else:
            return logits


class MI_Net_RC(nn.Module):
    def __init__(
        self,
        size=(384, 128),
        n_classes=1,
        pooling_mode="max",
        return_features=False,
        return_preds=True,
    ):
        super(MI_Net_RC, self).__init__()
        self.size = size
        self.n_classes = n_classes
        self.pooling_mode = pooling_mode
        self.return_features = return_features
        self.return_preds = return_preds

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

        self.rc1 = RC_Block(pooling_mode=self.pooling_mode)
        self.rc2 = RC_Block(pooling_mode=self.pooling_mode)
        self.rc3 = RC_Block(pooling_mode=self.pooling_mode)

        self.fc = nn.Linear(size[1], self.n_classes)
        self.act = nn.Sigmoid() if self.n_classes == 1 else nn.Softmax()

    def forward(self, x, **kwargs):
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
        preds = self.act(logits)

        if self.return_features and self.return_preds:
            return preds, logits, feats
        elif self.return_features:
            return logits, feats
        elif self.return_preds:
            return preds, logits
        else:
            return logits


class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        return torch.log(x)


def get_minet_model(
    config, n_classes, size, pooling_mode, return_features=False, return_preds=True
):
    if config.model.classifier == "minet_naive":
        assert len(size) == 4, "size must be a list of 4 integers"
        model = mi_NET(
            size=size,
            n_classes=n_classes,
            pooling_mode=pooling_mode,
            return_features=return_features,
            return_preds=return_preds,
        )
    elif config.model.classifier == "minet":
        assert len(size) == 4, "size must be a list of 4 integers"
        model = MI_Net(
            size=size,
            n_classes=n_classes,
            pooling_mode=pooling_mode,
            return_features=return_features,
            return_preds=return_preds,
        )
    elif config.model.classifier == "minet_ds":
        assert len(size) == 4, "size must be a list of 4 integers"
        model = MI_Net_DS(
            size=size,
            n_classes=n_classes,
            pooling_mode=pooling_mode,
            return_features=return_features,
            return_preds=return_preds,
        )
    elif config.model.classifier == "minet_rc":
        assert len(size) >= 2, "size must be a list of at least 2 integers"
        model = MI_Net_RC(
            size=size,
            n_classes=n_classes,
            pooling_mode=pooling_mode,
            return_features=return_features,
            return_preds=return_preds,
        )
    return model


class MINet_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        pooling_mode="max",
        multires_aggregation: Union[None, str] = None,
    ):
        super(MINet_PL, self).__init__(config, n_classes=n_classes, size=size)
        if self.n_classes == 2:
            self.n_classes = 1

        if self.n_classes == 1:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

        self.dropout = dropout
        self.pooling_mode = pooling_mode
        self.multires_aggregation = multires_aggregation

        self.model = get_minet_model(
            config,
            self.n_classes,
            size,
            pooling_mode,
            return_features=False,
            return_preds=True,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch["features"], batch["labels"]

        # Prediction
        preds, logits = self._forward(features)
        preds = preds.squeeze(dim=1)
        target = target.squeeze(dim=1)
        logits = logits.squeeze(dim=1)

        loss = None
        if not is_predict:
            if self.n_classes > 1:
                loss = self.loss.forward(torch.log(preds), target)
            else:
                loss = self.loss.forward(preds.float(), target.float())

        return {
            "target": target,
            "preds": preds,
            "logits": logits,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch):
        preds = []
        logits = []
        for singlePatientFeatures in features_batch:
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation == "bilinear":
                assert len(h) == 2
                h = self.bilinear(h[0], h[1])
            elif self.multires_aggregation == "linear":
                assert len(h) == 2
                h = self.linear_agg_target(h[0]) + self.linear_agg_context(h[1])
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _preds, _logits = self.model.forward(h)
            preds.append(_preds)
            logits.append(_logits)
        return torch.vstack(logits), torch.vstack(preds)
