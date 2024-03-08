# https://github.com/hustvl/MMIL-Transformer

import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from einops import rearrange
from nystrom_attention import NystromAttention
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.clustering import FeatureClustering, cat_msg2cluster_group
from pytorch_models.utils.tensor import aggregate_features


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        # x = x.squeeze(dim=0)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class AttenLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1, attn_mode="normal"):
        super(AttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.mode = attn_mode
        self.attn = Attention(
            self.dim, heads=self.heads, dim_head=self.dim_head, dropout=self.dropout
        )

    def forward(self, x):
        return x + self.attn(x)


class NyAttenLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super(NyAttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        return x + self.attn(x)


class GroupsAttenLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1, attn_mode="normal"):
        super(GroupsAttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        if attn_mode == "nystrom":
            self.AttenLayer = NyAttenLayer(
                dim=self.dim,
                heads=self.heads,
                dim_head=self.dim_head,
                dropout=self.dropout,
            )
        else:
            self.AttenLayer = AttenLayer(
                dim=self.dim,
                heads=self.heads,
                dim_head=self.dim_head,
                dropout=self.dropout,
            )

    def forward(self, x_groups, mask_ratio=0):
        group_after_attn = []
        r = int(len(x_groups) * (1 - mask_ratio))
        x_groups_masked = random.sample(x_groups, k=r)
        for x in x_groups_masked:
            x = x.squeeze(dim=0)
            temp = self.AttenLayer(x).unsqueeze(dim=0)
            group_after_attn.append(temp)
        return group_after_attn


class GroupsMSGAttenLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.AttenLayer = AttenLayer(
            dim=self.dim, heads=self.heads, dim_head=self.dim_head, dropout=self.dropout
        )

    def forward(self, data):
        msg_cls, x_groups, msg_tokens_num = data
        groups_num = len(x_groups)
        msges = torch.zeros(size=(1, 1, groups_num * msg_tokens_num, self.dim)).to(
            msg_cls.device
        )
        for i in range(groups_num):
            msges[:, :, i * msg_tokens_num : (i + 1) * msg_tokens_num, :] = x_groups[i][
                :, :, 0:msg_tokens_num
            ]
        msges = torch.cat((msg_cls, msges), dim=2).squeeze(dim=0)
        msges = self.AttenLayer(msges).unsqueeze(dim=0)
        msg_cls = msges[:, :, 0].unsqueeze(dim=0)
        msges = msges[:, :, 1:]
        for i in range(groups_num):
            x_groups[i] = torch.cat(
                (
                    msges[:, :, i * msg_tokens_num : (i + 1) * msg_tokens_num],
                    x_groups[i][:, :, msg_tokens_num:],
                ),
                dim=2,
            )
        data = msg_cls, x_groups, msg_tokens_num
        return data


class BasicLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.GroupsAttenLayer = GroupsAttenLayer(dim=dim)
        self.GroupsMSGAttenLayer = GroupsMSGAttenLayer(dim=dim)

    def forward(self, data, mask_ratio):
        msg_cls, x_groups, msg_tokens_num = data
        x_groups = self.GroupsAttenLayer(x_groups, mask_ratio)
        data = (msg_cls, x_groups, msg_tokens_num)
        data = self.GroupsMSGAttenLayer(data)
        return data


class MultipleMILTransformer(nn.Module):
    def __init__(
        self,
        in_chans: int = 1024,
        embed_dim: int = 512,
        n_classes: int = 2,
        num_msg: int = 1,
        num_subbags: int = 16,
        mode: str = "random",
        ape: bool = True,
        num_layers: int = 2,
        max_size: int = int(5e4),
    ):
        super(MultipleMILTransformer, self).__init__()

        self.embed_dim = embed_dim
        self.ape = ape
        self.mode = mode

        self.fc1 = nn.Linear(in_chans, embed_dim)
        self.fc2 = nn.Linear(embed_dim, n_classes)
        self.msg_tokens_num = num_msg
        self.msgcls_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        # ---> make sub-bags
        print("try to grouping seq to ", num_subbags)
        self.grouping = FeatureClustering(num_subbags, max_size=max_size)
        if mode == "random":
            self.grouping_features = self.grouping.random_grouping
        elif mode == "coords":
            self.grouping_features = self.grouping.coords_grouping
        elif mode == "seq":
            self.grouping_features = self.grouping.seqential_grouping
        elif mode == "embed":
            self.grouping_features = self.grouping.embedding_grouping
        else:
            raise ValueError(
                f"Grouping function requested (`{mode}`) should be one of [`random`, `coords`, `seq`, `embed`]"
            )

        self.msg_tokens = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.cat_msg2cluster_group = cat_msg2cluster_group
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --->build layers
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = BasicLayer(dim=embed_dim)
            self.layers.append(layer)

    def head(self, x):
        return self.fc2(x)

    def forward(self, x, coords=None, mask_ratio=0, return_features=False):
        # ---> init
        x = self.fc1(x)
        if self.ape:
            x = x + self.absolute_pos_embed.expand(1, x.shape[0], self.embed_dim)

        if self.mode != "coords":
            x_groups = self.grouping_features(x)
        else:
            assert (
                coords is not None
            ), "Centroid coordinates of each patch must be given, since clustering mode is `coords`"
            x_groups = self.grouping_features(x, coords)

        msg_tokens = self.msg_tokens.expand(1, 1, self.msg_tokens_num, -1)
        msg_cls = self.msgcls_token
        x_groups = self.cat_msg2cluster_group(x_groups, msg_tokens)
        data = (msg_cls, x_groups, self.msg_tokens_num)
        # ---> feature forward
        for i in range(len(self.layers)):
            if i == 0:
                mr = mask_ratio
                data = self.layers[i](data, mr)
            else:
                mr = 0
                data = self.layers[i](data, mr)
        # ---> head
        msg_cls, _, _ = data
        msg_cls = msg_cls.view(1, self.embed_dim)
        logits = self.head(msg_cls)

        if return_features:
            return logits, msg_cls
        return logits


class MMIL_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: Union[List[int], Tuple[int, int]] = None,
        num_msg: int = 1,
        num_subbags: int = 16,
        mode: str = "random",
        ape: bool = True,
        num_layers: int = 2,
        multires_aggregation: Union[None, str] = None,
        n_resolutions: int = 1,
    ):
        if n_classes == 1:
            n_classes = 2
        super(MMIL_PL, self).__init__(
            config,
            n_classes=n_classes,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
        )
        assert len(size) == 2, "size must be a tuple of size 2"

        self.size = size
        self.num_msg = num_msg
        self.num_subbags = num_subbags
        self.grouping_mode = mode
        self.ape = ape
        self.num_layers = num_layers

        self.model = MultipleMILTransformer(
            in_chans=self.size[0],
            embed_dim=self.size[1],
            n_classes=self.n_classes,
            num_msg=self.num_msg,
            num_subbags=self.num_subbags,
            mode=self.grouping_mode,
            ape=self.ape,
            num_layers=self.num_layers,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target, coords = batch["features"], batch["labels"], None
        if self.grouping_mode == "coords":
            coords = batch["coords"]

        # Prediction
        logits = self._forward(features, coords)
        preds = F.softmax(logits, dim=1)

        loss = None
        if not is_predict:
            loss = self.loss.forward(logits, target.squeeze(dim=1))

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch, coords_batch=None):
        logits = []
        for idx, singlePatientFeatures in enumerate(features_batch):
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation == "linear":
                h = [self.linear_agg[i](h[i]) for i in range(len(h))]
                h = self._aggregate_multires_features(
                    h,
                    method="sum",
                    is_attention=False,
                )
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits = self.model.forward(h, coords_batch[idx] if coords_batch else None)
            logits.append(_logits)
        return torch.vstack(logits)

    def _compute_metrics(self, preds, target, mode):
        if mode == "val":
            metrics = self.val_metrics
        elif mode == "train":
            metrics = self.train_metrics
        elif mode in ["eval", "test"]:
            metrics = self.test_metrics
        if self.n_classes in [1, 2]:
            metrics(
                preds,
                nn.functional.one_hot(target.view(-1), num_classes=self.n_classes),
            )
        else:
            metrics(preds, target.view(-1))
