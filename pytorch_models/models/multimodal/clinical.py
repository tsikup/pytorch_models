from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
from typing import List, Union

NormType = Enum("NormType", "Batch BatchZero Weight Spectral Instance InstanceZero")


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low


def SigmoidRange(self, x):
    "Sigmoid module with range `(low, high)`"
    return sigmoid_range(x, self.low, self.high)


def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm(
        "BatchNorm", nf, ndim, zero=norm_type == NormType.BatchZero, **kwargs
    )


class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"

    def __init__(self, n_in, n_out, bn=True, p=0.0, act=None, lin_first=False):
        layers = [BatchNorm(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None:
            lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)


class TabularModel(nn.Module):
    "Basic model for tabular data."

    def __init__(
        self,
        emb_szs: list,  # Sequence of (num_embeddings, embedding_dim) for each categorical variable
        n_cont: int,  # Number of continuous variables
        layers: list,  # Sequence of ints used to specify the input and output size of each `LinBnDrop` layer
        out_sz: int = None,  # Number of outputs for final `LinBnDrop` layer
        ps: Union[
            float, List[float]
        ] = None,  # Sequence of dropout probabilities for `LinBnDrop`
        embed_p: float = 0.0,  # Dropout probability for `Embedding` layer
        y_range=None,  # Low and high for `SigmoidRange` activation
        use_bn: bool = True,  # Use `BatchNorm1d` in `LinBnDrop` layers
        bn_final: bool = False,  # Use `BatchNorm1d` on final layer
        bn_cont: bool = True,  # Use `BatchNorm1d` on continuous variables
        act_cls=nn.ReLU(inplace=True),  # Activation type for `LinBnDrop` layers
        lin_first: bool = True,  # Linear layer is first or last in `LinBnDrop` layers
    ):
        super().__init__()
        if not ps:
            ps = [0] * len(layers)
        if ps and not (isinstance(ps, list) or isinstance(ps, tuple)):
            ps = [ps] * len(layers)
        ps = list(ps)
        layers = list(layers)
        assert len(ps) == len(layers)
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = n_emb, n_cont
        sizes = (
            [n_emb + n_cont] + layers + [out_sz]
            if out_sz
            else [n_emb + n_cont] + layers
        )
        actns = (
            [act_cls for _ in range(len(sizes) - 2)] + [None]
            if out_sz is not None
            else [act_cls for _ in range(len(sizes) - 1)]
        )
        _layers = [
            LinBnDrop(
                sizes[i],
                sizes[i + 1],
                bn=use_bn and (i != len(actns) - 1 or bn_final or out_sz is None),
                p=p,
                act=a,
                lin_first=lin_first,
            )
            for i, (p, a) in enumerate(
                zip(ps + [0.0] if out_sz is not None else ps, actns)
            )
        ]
        if y_range is not None and out_sz is not None:
            _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:, i].int()) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None:
                x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)


class ClinicalModel(nn.Module):
    def __init__(
        self,
        size_cat: int,
        size_cont: int,
        layers: List[int],
        embed_size: list = None,
        dropout: float = 0.5,
        batch_norm=True,
    ):
        super(ClinicalModel, self).__init__()

        self.embed_size = embed_size
        ps = [dropout] * len(layers)

        if embed_size:
            self.clinical_model = TabularModel(
                emb_szs=embed_size,
                n_cont=size_cont,
                layers=layers,
                out_sz=None,
                ps=ps,
                use_bn=batch_norm,
                bn_cont=batch_norm,
            )
        else:
            size_clinical = [size_cat + size_cont] + layers
            _acts = [nn.ReLU(inplace=True) for _ in range(len(layers))]
            _layers = [
                LinBnDrop(
                    size_clinical[i],
                    size_clinical[i + 1],
                    bn=batch_norm,
                    p=p,
                    act=a,
                    lin_first=True,
                )
                for i, (p, a) in enumerate(zip(ps, _acts))
            ]
            self.clinical_model = nn.Sequential(*_layers)

    def forward(self, categorical, continuous):
        if self.embed_size:
            return self.clinical_model(categorical, continuous)
        else:
            x = torch.cat((categorical, continuous), dim=-1)
            return self.clinical_model(x)
