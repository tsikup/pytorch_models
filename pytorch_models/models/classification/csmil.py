# https://github.com/hrlblab/CS-MIL
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from dotmap import DotMap
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features

"""
Model definition of DeepAttnMISL

If this work is useful for your research, please consider to cite our papers:

[1] "Whole Slide Images based Cancer Survival Prediction using Attention Guided Deep Multiple Instance Learning Networks"
Jiawen Yao, XinliangZhu, Jitendra Jonnagaddala, NicholasHawkins, Junzhou Huang,
Medical Image Analysis, Available online 19 July 2020, 101789

[2] "Deep Multi-instance Learning for Survival Prediction from Whole Slide Images", In MICCAI 2019

"""


class CSMIL(nn.Module):
    """
    Deep AttnMISL Model definition
    """

    def __init__(self, cluster_num=1, feature_size=1024, n_classes=1):
        super(CSMIL, self).__init__()
        self.n_classes = n_classes

        self.embedding_net = nn.Sequential(
            nn.Conv2d(feature_size, 64, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )

        self.res_attention = nn.Sequential(
            nn.Conv2d(64, 32, 1),  # V
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)  # V  # W
        )

        self.fc6 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, n_classes),
        )
        self.cluster_num = cluster_num

    def masked_softmax(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / (mask + 1e-5))
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)

    def forward(self, x, mask=None):

        "x is a tensor list"
        res = []
        for i in range(self.cluster_num):
            hh = x[i].type(torch.FloatTensor)
            n_scales = hh.size(2)
            output = []
            for j in range(n_scales):
                output.append(self.embedding_net(hh[:, :, j : j + 1, :]))
            output = torch.cat(output, 2)
            res_attention = self.res_attention(output).squeeze(-1)

            final_output = torch.matmul(
                output.squeeze(-1), torch.transpose(res_attention, 2, 1)
            ).squeeze(-1)
            res.append(final_output)

        h = torch.cat(res)

        b = h.size(0)
        c = h.size(1)

        h = h.view(b, c)

        A = self.attention(h)
        A = torch.transpose(A, 1, 0)  # KxN

        A = self.masked_softmax(A, mask)

        M = torch.mm(A, h)  # KxL

        logits = self.fc6(M)

        if self.n_classes == 1:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=1)

        return logits, preds


class CSMIL_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: int = 1024,
        cluster_num: int = 1,
    ):
        super(CSMIL_PL, self).__init__(config, n_classes=n_classes)

        assert self.n_classes > 0, "n_classes must be greater than 0"
        if self.n_classes == 2:
            self.n_classes = 1

        self.model = CSMIL(
            cluster_num=cluster_num, feature_size=size, n_classes=self.n_classes
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch["features"], batch["labels"]

        # Prediction
        logits, preds = self._forward(features)
        logits = logits.squeeze(dim=1)
        target = target.squeeze(dim=1)

        loss = None
        if not is_predict:
            loss = self.loss.forward(
                logits, target.float() if self.n_classes == 1 else target
            )

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features):
        h = [features[key] for key in features]
        h = torch.stack(h, dim=-1)
        h = h.unsqueeze(dim=-1)
        if len(h.shape) == 4:
            h = h.unsqueeze(dim=0)
        # h -> [n_clusters, n_patches, n_features, n_resolutions, 1]
        return self.model.forward(h)


if __name__ == "__main__":
    # create test data and model
    n_features = 1024
    n_classes = 3
    n_samples = 100

    loss = nn.CrossEntropyLoss()
    # loss = nn.BCEWithLogitsLoss()
    target = torch.from_numpy(np.array([[1]]))
    features = torch.rand(1, n_samples, n_features, 2, 1)

    model = CSMIL(cluster_num=1, feature_size=n_features, n_classes=n_classes)

    # test forward
    logits, preds = model.forward(features, None)

    logits = logits.squeeze(dim=1)
    target = target.squeeze(dim=1)

    _loss = loss.forward(logits, target.float() if n_classes == 1 else target)

    print(_loss)
