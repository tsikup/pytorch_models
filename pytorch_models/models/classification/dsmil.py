"""
DSMIL: Dual-stream multiple instance learning networks for tumor detection in Whole Slide Image
https://arxiv.org/abs/2011.08939
https://github.com/binli123/dsmil-wsi
"""
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features


class FCLayer(nn.Module):
    def __init__(self, size: List[int], nonlinear=False):
        super(FCLayer, self).__init__()
        assert isinstance(size, list) and len(size) > 1
        fc = []
        for idx in range(len(size) - 1):
            fc.append(nn.Sequential(nn.Linear(size[idx], size[idx + 1])))
            if nonlinear:
                fc.append(nn.ReLU())
        self.fc = nn.Sequential(*fc)

    def forward(self, feats):
        return self.fc(feats)


class FCClassifier(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(
        self, size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False
    ):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(size[0], size[1]),
                nn.ReLU(),
                nn.Linear(size[1], size[1]),
                nn.Tanh(),
            )
        else:
            self.q = nn.Linear(size[0], size[1])
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(size[0], size[0]), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=size[0])

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(
            c, 0, descending=True
        )  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(
            feats, dim=0, index=m_indices[0, :]
        )  # select critical instances, m_feats in shape C x K
        q_max = self.q(
            m_feats
        )  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(
            Q, q_max.transpose(0, 1)
        )  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(
            A
            / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
            0,
        )  # normalize attention scores, A in shape N x C,
        B = torch.mm(
            A.transpose(0, 1), V
        )  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x, return_features=False):
        feats, classes = self.i_classifier(x)
        logits, A, B = self.b_classifier(feats, classes)
        if return_features:
            return logits, feats
        return logits


class DSMIL(nn.Module):
    def __init__(
        self,
        size=(384, 128),
        n_classes=1,
        dropout=0.0,
        nonlinear=True,
        passing_v=False,
    ):
        super(DSMIL, self).__init__()
        if len(size) > 2:
            self.feature_processor = FCLayer(size[:-1], nonlinear=nonlinear)
        else:
            self.feature_processor = nn.Identity()
        i_classifier = FCClassifier(size[-2], n_classes)
        b_classifier = BClassifier(
            size=size[-2:],
            output_class=n_classes,
            dropout_v=dropout,
            nonlinear=nonlinear,
            passing_v=passing_v,
        )
        self.model = MILNet(i_classifier, b_classifier)

    def forward(self, features, return_features=False):
        return self.model.forward(
            self.feature_processor(features), return_features=return_features
        )


class DSMIL_PL(BaseMILModel):
    def __init__(
        self,
        config,
        size: Union[List[int], Tuple[int, int]] = (384, 128),
        n_classes=1,
        dropout=0.0,
        nonlinear=True,
        passing_v=False,
        multires_aggregation: Union[None, str] = None,
    ):
        if n_classes == 2:
            n_classes = 1
        self.multires_aggregation = multires_aggregation
        super(DSMIL_PL, self).__init__(config, n_classes=n_classes, size=size)
        assert len(size) >= 2, "size must be a tuple with 2 or more elements"
        self.model = DSMIL(
            size=size,
            n_classes=self.n_classes,
            dropout=dropout,
            nonlinear=nonlinear,
            passing_v=passing_v,
        )
