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

    def forward(self, x, mask=None, return_features=False):

        "x is a tensor list"
        res = []
        for i in range(self.cluster_num):
            hh = x[i]
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
        # h -> [n_clusters, n_patches, n_features, n_resolutions, 1] for concat
        # h -> [n_clusters, n_patches, n_features, 1, 1] for others

        b = h.size(0)
        c = h.size(1)

        h = h.view(b, c)

        A = self.attention(h)
        A = torch.transpose(A, 1, 0)  # KxN

        A = self.masked_softmax(A, mask)

        M = torch.mm(A, h)  # KxL

        logits = self.fc6(M)

        if return_features:
            return logits, M
        return logits


class CSMIL_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: int = 1024,
        cluster_num: int = 1,
        multires_aggregation: Union[None, str] = None,
    ):
        super(CSMIL_PL, self).__init__(config, n_classes=n_classes, multires_aggregation=multires_aggregation, size=[size])

        if self.n_classes == 2:
            self.n_classes = 1

        self.model = CSMIL(
            cluster_num=cluster_num, feature_size=size, n_classes=self.n_classes
        )

    def _forward(self, features_batch):
        logits = []
        for singlePatientFeatures in features_batch:
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation == "concat":
                h = torch.stack(h, dim=-1)
            elif self.multires_aggregation == "bilinear":
                assert len(h) == 2
                h = self.bilinear(h[0], h[1])
                h = h.unsqueeze(dim=-1)
            elif self.multires_aggregation == "linear":
                assert len(h) == 2
                h = self.linear_agg_target(h[0]) + self.linear_agg_context(h[1])
                h = h.unsqueeze(dim=-1)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
                h = h.unsqueeze(dim=-1)
            h = h.unsqueeze(dim=-1)
            if len(h.shape) == 4:
                h = h.unsqueeze(dim=0)
            # h -> [n_clusters, n_patches, n_features, n_resolutions, 1] here n_clusters = 1, since we don't apply clustering as the original csmil paper
            _logits = self.model.forward(h)
            logits.append(_logits)
        return torch.vstack(logits)
