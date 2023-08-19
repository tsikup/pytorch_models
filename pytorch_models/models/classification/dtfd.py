# https://github.com/hrzhang1123/DTFD-MIL
import random
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0.0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x):  ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x)  ## K x L
        pred = self.classifier(afeat)  ## K x num_cls
        return pred


class DTFD(nn.Module):
    def __init__(self, size, n_classes, K=1, n_bags=3, dropout=0.25):
        super(DTFD, self).__init__()
        self.n_bags = n_bags
        self.n_classes = n_classes

        self.dim_reduction = DimReduction(n_channels=size[0], m_dim=size[1])
        self.attention = Attention_Gated(L=size[1], D=size[1], K=K)
        self.classifier = Classifier_1fc(size[1], n_classes)
        self.UClassifier = Attention_with_Classifier(
            L=size[1], D=size[2], K=K, num_cls=self.n_classes, droprate=dropout
        )

    def forward(self, h: torch.Tensor):
        feat_index = list(range(h.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), self.n_bags)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        slide_pseudo_feat = []
        slide_sub_logits = []

        for tindex in index_chunk_list:
            sub_h = torch.index_select(
                h, dim=0, index=torch.LongTensor(tindex).to(h.device)
            )
            sub_h = self.dim_reduction(sub_h)
            sub_A = self.attention(sub_h).squeeze(0)
            sub_M = torch.einsum("ns,n->ns", sub_h, sub_A)  ### n x fs
            sub_M = torch.sum(sub_M, dim=0).unsqueeze(0)  ## 1 x fs
            sub_logits = self.classifier(sub_M)  ### 1 x 2
            slide_sub_logits.append(sub_logits)
            slide_pseudo_feat.append(sub_M)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
        logits = self.UClassifier(slide_pseudo_feat)  ### 1 x num_cls

        if self.n_classes == 1:
            Y_hat = torch.ge(torch.sigmoid(logits), 0.5).float()
        else:
            Y_hat = torch.topk(logits, 1, dim=1)[1].float()

        return logits, slide_sub_logits, Y_hat


class DTFD_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        K: int = 1,
        n_bags=3,
        dropout=0.25,
        multires_aggregation: Union[None, str] = None,
    ):
        super(DTFD_PL, self).__init__(config, n_classes=n_classes)

        assert len(size) == 3, "size must be a tuple of size 3"
        assert self.n_classes > 0, "n_classes must be greater than 0"
        if self.n_classes == 2:
            self.n_classes = 1

        self.multires_aggregation = multires_aggregation

        self.model = DTFD(
            size=size,
            n_classes=self.n_classes,
            K=K,
            n_bags=n_bags,
            dropout=dropout,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch["features"], batch["labels"]

        # Prediction
        logits, sub_logits, _ = self._forward(features)
        if self.n_classes == 1:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)

        logits = logits.squeeze(dim=1)
        target = target.squeeze(dim=1)

        sub_logits = torch.concatenate(sub_logits, dim=0)  ### numGroup x fs
        sub_labels = target.repeat(sub_logits.shape[0])  ### numGroup

        loss = None
        if not is_predict:
            loss = self.loss.forward(
                logits, target.float() if self.n_classes == 1 else target
            )
            loss += self.loss.forward(
                sub_logits,
                sub_labels.unsqueeze(1).float() if self.n_classes == 1 else sub_labels,
            )

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features):
        h: List[torch.Tensor] = [features[key] for key in features]
        h: torch.Tensor = aggregate_features(h, method=self.multires_aggregation)
        if len(h.shape) == 3:
            h = h.squeeze(dim=0)
        return self.model.forward(h)


if __name__ == "__main__":
    # create test data and model
    n_features = 1024
    n_classes = 3
    n_samples = 100

    loss = nn.CrossEntropyLoss()
    # loss = nn.BCEWithLogitsLoss()
    loss2 = nn.CrossEntropyLoss(reduction="none")
    target = torch.from_numpy(np.array([[1]]))
    features = torch.rand(n_samples, n_features)

    model = DTFD(
        size=(n_features, 512, 128),
        n_classes=n_classes,
        K=1,
        n_bags=3,
        dropout=0.25,
    )

    # test forward
    logits, sub_logits, preds = model.forward(features)

    logits = logits.squeeze(dim=1)
    target = target.squeeze(dim=1)

    sub_logits = torch.concatenate(sub_logits, dim=0)  ### numGroup x fs
    sub_labels = target.repeat(sub_logits.shape[0])  ### numGroup

    _loss = loss.forward(logits, target.float() if n_classes == 1 else target)
    _loss += loss.forward(
        sub_logits, sub_labels.unsqueeze(1).float() if n_classes == 1 else sub_labels
    )

    print(_loss)
