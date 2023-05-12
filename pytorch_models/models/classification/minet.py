# https://github.com/yanyongluan/MINNs
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    if pooling_mode == "ave":
        return mean_pooling(x)


class ScorePooling(nn.Module):
    def __init__(self, D, C, pooling_mode="max"):
        super(ScorePooling, self).__init__()
        self.fc = nn.Linear(D, C)
        self.sigmoid = nn.Sigmoid()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        output = choice_pooling(x, self.pooling_mode)
        return output


class FeaturePooling(nn.Module):
    def __init__(self, D, C, pooling_mode="max"):
        super(FeaturePooling, self).__init__()
        self.fc = nn.Linear(D, C)
        self.sigmoid = nn.Sigmoid()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        x = choice_pooling(x, self.pooling_mode)
        x = self.fc(x)
        output = self.sigmoid(x)
        return output


class RC_Block(nn.Module):
    def __init__(self, pooling_mode="max"):
        super(RC_Block, self).__init__()
        self.pooling_mode = pooling_mode

    def forward(self, x):
        return choice_pooling(x, self.pooling_mode)


class mi_NET(nn.Module):
    def __init__(self, size=(384, 256, 128, 64), n_classes=1, pooling_mode="max"):
        super(mi_NET, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.pooling_mode = pooling_mode

        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        self.sp = ScorePooling(size[3], self.n_classes, pooling_mode=self.pooling_mode)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        output = self.sp(x)
        return output


class MI_Net(nn.Module):
    def __init__(self, size=(384, 256, 128, 64), n_classes=1, pooling_mode="max"):
        super(MI_Net, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.pooling_mode = pooling_mode

        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        self.fp = FeaturePooling(
            size[3], self.n_classes, pooling_mode=self.pooling_mode
        )

    def forwrad(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        output = self.fp(x)
        return output


class MI_Net_DS(nn.Module):
    def __init__(self, size=(384, 256, 128, 64), n_classes=1, pooling_mode="max"):
        super(MI_Net_DS, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.pooling_mode = pooling_mode

        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(size[2], size[3]), nn.ReLU())

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.fp1 = FeaturePooling(
            size[1], self.n_classes, pooling_mode=self.pooling_mode
        )
        self.fp2 = FeaturePooling(
            size[2], self.n_classes, pooling_mode=self.pooling_mode
        )
        self.fp3 = FeaturePooling(
            size[3], self.n_classes, pooling_mode=self.pooling_mode
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.dropout1(x1)
        output1 = self.fp1(x1)

        x2 = self.fc2(x1)
        x2 = self.dropout2(x2)
        output2 = self.fp2(x2)

        x3 = self.fc3(x2)
        x3 = self.dropout3(x3)
        output3 = self.fp3(x3)

        return torch.mean(torch.stack([output1, output2, output3], dim=-1), dim=-1)


class MI_Net_RC(nn.Module):
    def __init__(self, size=(384, 128), n_classes=1, pooling_mode="max"):
        super(MI_Net_RC, self).__init__()
        self.size = size
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.n_classes = 1
        self.pooling_mode = pooling_mode

        self.fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
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

        output = torch.sum(torch.stack([output1, output2, output3], dim=-1), dim=-1)
        output = self.fc(output)
        output = self.act(output)

        return output


class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()

    def forward(self, x):
        return torch.log(x)


class MI_Net_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        pooling_mode="max",
        multires_aggregation: Union[None, str] = None,
    ):
        super(MI_Net_PL, self).__init__(config, n_classes=n_classes)
        assert self.n_classes > 0, "n_classes must be greater than 0"
        if self.n_classes == 2:
            self.n_classes = 1

        if self.n_classes == 1:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.Sequential(Log(), nn.NLLLoss())

        self.dropout = dropout
        self.pooling_mode = pooling_mode
        self.multires_aggregation = multires_aggregation

        if self.config.model == "minet_naive":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = mi_NET(
                size=size, n_classes=self.n_classes, pooling_mode=self.pooling_mode
            )
        elif self.config.model == "minet":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = MI_Net(
                size=size, n_classes=self.n_classes, pooling_mode=self.pooling_mode
            )
        elif self.config.model == "minet_ds":
            assert len(size) == 4, "size must be a list of 4 integers"
            self.model = MI_Net_DS(
                size=size, n_classes=self.n_classes, pooling_mode=self.pooling_mode
            )
        elif self.config.model == "minet_rc":
            assert len(size) == 2, "size must be a list of 2 integers"
            self.model = MI_Net_RC(
                size=size, n_classes=self.n_classes, pooling_mode=self.pooling_mode
            )

    def forward(self, batch, is_predict=False):
        raise NotImplementedError
        # Batch
        features, target = batch

        # Prediction
        preds = self._forward(features)
        preds = preds.squeeze(dim=1)
        target = target.squeeze(dim=1)

        loss = None
        if not is_predict:
            loss = self.loss.forward(preds.float(), target.float())

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
        }

    def _forward(self, features):
        h: List[torch.Tensor] = [features[key] for key in features]
        h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
        if len(h.shape) == 3:
            h = h.squeeze(dim=0)
        return self.model.forward(h)
