"""
Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images. Nature Biomedical Engineering
"""
import copy
from typing import Union, List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from topk.svm import SmoothTop1SVM
from torch import Tensor

from pytorch_models.models.base_fair import BaseMILModel_LNL
from pytorch_models.models.fair.lnl.base import _BaseLNL
from pytorch_models.models.fair.utils import grad_reverse
from pytorch_models.models.utils import (
    LinearWeightedTransformationSum,
    LinearWeightedSum,
)
from pytorch_models.utils.tensor import aggregate_features


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if "bias" in m.state_dict().keys():
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB_LNL(nn.Module):
    """
    args:
        gate: whether to use gated attention network
        size_arg: config for network size
        dropout: whether to use dropout
        k_sample: number of positive/neg patches to sample for instance-level training
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
        instance_loss_fn: loss function to supervise instance-level training
        subtyping: whether it's a subtyping problem
        multires_aggregation: whether to use multiresolution aggregation
        attention_depth: attention module starting size as an index of the size array
        classifier_depth: classification module starting size as an index of the size array
    """

    def __init__(
        self,
        gate=True,
        size=None,
        dropout=True,
        k_sample=8,
        n_classes=2,
        n_groups=2,
        instance_loss_fn="svm",
        subtyping=False,
        linear_feature=False,
        # bilinear=None,
        multires_aggregation=None,
        attention_depth: Union[List[int], int] = 1,
        classifier_depth: Union[List[int], int] = 1,
        n_resolutions: int = 1,
    ):
        super(CLAM_SB_LNL, self).__init__()

        if size is None:
            size = [384, 256, 128]
        assert isinstance(size, list), "Please give the size array as a list"

        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.multires_aggregation = multires_aggregation
        self.use_multires = (
            self.multires_aggregation is not None
            and self.multires_aggregation["features"] is not None
        )
        self.attention_depth = attention_depth
        self.classifier_depth = classifier_depth
        self.linear_feature = linear_feature
        self.n_resolutions = n_resolutions

        if instance_loss_fn == "svm":
            self.instance_loss_fn = SmoothTop1SVM(n_classes=n_classes)
            self.instance_loss_on_gpu = False
        else:
            self.instance_loss_fn = nn.CrossEntropyLoss()
            self.instance_loss_on_gpu = True

        assert self.attention_depth is not None and self.classifier_depth is not None

        if self.linear_feature:
            if self.use_multires:
                if (
                    self.multires_aggregation["features"] == "concat"
                    and self.multires_aggregation["attention"] != "late"
                ):
                    _size = int(size[0] // self.n_resolutions)
                else:
                    _size = size[0]
                self.dense_layers = []
                for _ in range(self.n_resolutions):
                    _layer = nn.Linear(_size, _size)
                    if self.linear_feature == "relu":
                        _layer = nn.Sequential(_layer, nn.ReLU())
                    elif self.linear_feature == "leakyrelu":
                        _layer = nn.Sequential(_layer, nn.LeakyReLU())
                    elif self.linear_feature == "prelu":
                        _layer = nn.Sequential(_layer, nn.PReLU(num_parameters=_size))
                    elif self.linear_feature == "gelu":
                        _layer = nn.Sequential(_layer, nn.GELU())
                    self.dense_layers.append(_layer)
            else:
                _layer = nn.Linear(size[0], size[0])
                if self.linear_feature == "relu":
                    _layer = nn.Sequential(_layer, nn.ReLU())
                elif self.linear_feature == "leakyrelu":
                    _layer = nn.Sequential(_layer, nn.LeakyReLU())
                elif self.linear_feature == "prelu":
                    _layer = nn.Sequential(_layer, nn.PReLU(num_parameters=size[0]))
                elif self.linear_feature == "gelu":
                    _layer = nn.Sequential(_layer, nn.GELU())
                self.dense_layers = [_layer]

            self.dense_layers = nn.ModuleList(self.dense_layers)

        if isinstance(self.classifier_depth, int):
            self.classifier_size = [size[self.classifier_depth]]
        elif isinstance(self.classifier_depth, list):
            assert (
                len(self.classifier_depth) == 2
            ), "Please give the classifier depth indices as [first, last] for multilayer or int (only one layer)"
            self.classifier_size = size[
                self.classifier_depth[0] : self.classifier_depth[1] + 1
            ]
        else:
            raise TypeError(
                "Please give the classifier depth indices as [first, last] for multilayer or int (only one layer)"
            )

        assert (
            self.classifier_size[0] == size[self.attention_depth]
        ), "Mismatch between attention module output feature size and classifiers' input feature size"

        if self.use_multires and self.multires_aggregation["attention"] == "late":
            __size = self.classifier_size[0]
        else:
            __size = size[0]

        if self.use_multires and self.multires_aggregation["features"] == "linear":
            self.linear_agg = LinearWeightedTransformationSum(
                __size, self.n_resolutions
            )
        elif self.use_multires and self.multires_aggregation["features"] == "linear_2":
            self.linear_agg = LinearWeightedSum(__size, self.n_resolutions)

        if self.use_multires and self.multires_aggregation["attention"] == "linear":
            self.linear_agg_attention = LinearWeightedTransformationSum(
                __size, self.n_resolutions
            )
        elif self.use_multires and self.multires_aggregation["attention"] == "linear_2":
            self.linear_agg_attention = LinearWeightedSum(__size, self.n_resolutions)

        if (
            self.use_multires
            and self.multires_aggregation["attention"] == "late"
            and self.multires_aggregation["features"] == "concat"
        ):
            last_layer = self.classifier_size[-1]
            self.classifier_size = [
                self.n_resolutions * l for l in self.classifier_size
            ]
            self.classifier_size.append(last_layer)

        self.attention_nets = []
        if self.use_multires and self.multires_aggregation["attention"] is not None:
            assert (
                self.multires_aggregation["attention"] != "concat"
            ), "Multiresolution integration at the attention level is enabled.. The aggregation function must not be concat for the attention vectors, because each tile feature vector (either integrated or not) should have a single attention score."
            for _ in range(self.n_resolutions):
                self.attention_nets.append(
                    self._create_attention_model(size, dropout, gate, n_classes=1)
                )
        else:
            self.attention_nets = [
                self._create_attention_model(size, dropout, gate, n_classes=1)
            ]
        self.attention_nets = nn.ModuleList(self.attention_nets)

        if (
            self.use_multires
            and self.multires_aggregation["features"] == "concat"
            and self.multires_aggregation["attention"] == "late"
        ):
            _downsample = self.n_resolutions
        else:
            _downsample = 1

        if len(self.classifier_size) > 1:
            _classifiers = []
            _aux_classifiers = []
            _instance_classifiers = dict()

            for idx, _ in enumerate(self.classifier_size[:-1]):
                _classifiers.append(
                    nn.Linear(self.classifier_size[idx], self.classifier_size[idx + 1])
                )
            _classifiers.append(nn.Linear(self.classifier_size[-1], n_classes))
            self.classifiers = nn.Sequential(*_classifiers)

            for idx, _ in enumerate(self.classifier_size[:-1]):
                _aux_classifiers.append(
                    nn.Linear(self.classifier_size[idx], self.classifier_size[idx + 1])
                )
            _aux_classifiers.append(nn.Linear(self.classifier_size[-1], n_groups))
            self.aux_classifiers = nn.Sequential(*_aux_classifiers)

            for c in range(n_classes):
                _tmp_instance_classifier = []
                for idx, _ in enumerate(self.classifier_size[:-1]):
                    _tmp_instance_classifier.append(
                        nn.Linear(
                            self.classifier_size[idx] // _downsample,
                            self.classifier_size[idx + 1] // _downsample,
                        )
                    )
                _tmp_instance_classifier.append(
                    nn.Linear(self.classifier_size[-1] // _downsample, 2)
                )
                _instance_classifiers[c] = nn.Sequential(*_tmp_instance_classifier)
            self.instance_classifiers = nn.ModuleList(
                [_instance_classifiers[c] for c in range(n_classes)]
            )
        else:
            self.classifiers = nn.Linear(self.classifier_size[0], n_classes)
            self.aux_classifiers = nn.Linear(self.classifier_size[0], n_groups)

            instance_classifiers = [
                nn.Linear(self.classifier_size[0] // _downsample, 2)
                for i in range(n_classes)
            ]
            self.instance_classifiers = nn.ModuleList(instance_classifiers)

        initialize_weights(self)

    def _create_attention_model(self, size, dropout, gate, n_classes):
        depth = self.attention_depth
        fc = []
        for i in range(depth):
            fc.append(nn.Linear(size[i], size[i + 1]))
            fc.append(nn.ReLU())
            if dropout:
                fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(
                L=size[depth], D=size[depth + 1], dropout=dropout, n_classes=n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[depth], D=size[depth + 1], dropout=dropout, n_classes=n_classes
            )
        fc.append(attention_net)
        return nn.Sequential(*fc)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if not self.instance_loss_on_gpu:
            self.instance_loss_fn = self.instance_loss_fn.cuda(device.index)
            self.instance_loss_on_gpu = True
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, min(A.shape[1], self.k_sample))[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, min(A.shape[1], self.k_sample), dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(min(A.shape[1], self.k_sample), device)
        n_targets = self.create_negative_targets(min(A.shape[1], self.k_sample), device)
        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if not self.instance_loss_on_gpu:
            self.instance_loss_fn = self.instance_loss_fn.cuda(device.index)
            self.instance_loss_on_gpu = True
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, min(A.shape[1], self.k_sample))[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(min(A.shape[1], self.k_sample), device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    @staticmethod
    def _aggregate_multires_features(
        features: List[torch.Tensor], method, is_attention=False
    ):
        assert not (
            method == "concat" and is_attention
        ), "Attention vectors cannot be integrated with concat method."
        return aggregate_features(features=features, method=method)

    def forward_imaging(
        self,
        features: List[Tensor],
    ):
        if self.linear_feature:
            for idx, _layer in enumerate(self.dense_layers):
                features[idx] = _layer(features[idx])

        if self.use_multires:
            assert (
                len(features) > 1
            ), "Multiresolution is enabled.. h_context features should not be None."
            if self.multires_aggregation["attention"] is None:
                if self.multires_aggregation["features"] != "linear":
                    h = self._aggregate_multires_features(
                        features,
                        method=self.multires_aggregation["features"],
                        is_attention=False,
                    )
                elif self.multires_aggregation["features"] in ["linear", "linear_2"]:
                    h = self.linear_agg(features)
                A, h = self.attention_nets[0](h)  # NxK
                A = torch.transpose(A, 1, 0)  # KxN
            else:
                A = []
                h = []
                for i in range(len(features)):
                    _A, _h = self.attention_nets[i](features[i])
                    _A = torch.transpose(_A, 1, 0)  # KxN
                    A.append(_A)
                    h.append(_h)

                if self.multires_aggregation["attention"] != "late":
                    if self.multires_aggregation["attention"] != "linear":
                        A = self._aggregate_multires_features(
                            A,
                            method=self.multires_aggregation["attention"],
                            is_attention=True,
                        )
                    elif self.multires_aggregation["attention"] in [
                        "linear",
                        "linear_2",
                    ]:
                        A = self.linear_agg_attention(A)
                    if self.multires_aggregation["features"] != "linear":
                        h = self._aggregate_multires_features(
                            h,
                            method=self.multires_aggregation["features"],
                            is_attention=False,
                        )
                    elif self.multires_aggregation["features"] in [
                        "linear",
                        "linear_2",
                    ]:
                        h = self.linear_agg(h)
        else:
            assert (
                len(features) == 1
            ), "Single resolution is enabled but more than 1 res features vector were supplied."
            h = features[0]
            A, h = self.attention_nets[0](h)
            A = torch.transpose(A, 1, 0)  # KxN

        return h, A

    def forward_instance_eval(
        self,
        h,
        A,
        label,
        instance_eval=False,
    ):
        results_dict = {}
        if self.use_multires and self.multires_aggregation["attention"] == "late":

            for idx in range(self.n_resolutions):
                A[idx] = F.softmax(A[idx], dim=1)  # softmax over N

            if instance_eval:
                is_target = [
                    [idx == 0 for idx in range(self.n_resolutions)]
                    for _ in range(len(h))
                ]
                total_inst_loss = 0.0
                all_preds = []
                all_targets = []
                all_context_preds = []
                inst_labels = F.one_hot(
                    label.to(torch.int64), num_classes=self.n_classes
                ).view(
                    -1, len(self.instance_classifiers)
                )  # binarize label
                for _tmp in zip(h, A, is_target):
                    _h, _A, is_target = _tmp
                    for i in range(len(self.instance_classifiers)):
                        inst_label = inst_labels[:, i].item()
                        classifier = self.instance_classifiers[i]
                        if inst_label == 1:  # in-the-class:
                            instance_loss, preds, targets = self.inst_eval(
                                _A,
                                _h,
                                classifier,
                            )
                        else:  # out-of-the-class
                            if self.subtyping:
                                instance_loss, preds, targets = self.inst_eval_out(
                                    A,
                                    h,
                                    classifier,
                                )
                            else:
                                continue
                        if is_target:
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            all_context_preds.extend(preds.cpu().numpy())
                        total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

                total_inst_loss /= 2

                results_dict = {
                    "instance_loss": total_inst_loss,
                    "inst_labels": np.array(all_targets),
                    "inst_preds": np.array(all_preds),
                    "inst_context_preds": np.array(all_context_preds),
                }

            M = []
            for idx in range(self.n_resolutions):
                M.append(torch.mm(A[idx], h[idx]))
            if self.multires_aggregation["features"] != "linear":
                M = self._aggregate_multires_features(
                    M,
                    method=self.multires_aggregation["features"],
                    is_attention=False,
                )
            elif self.multires_aggregation["features"] in ["linear", "linear_2"]:
                M = self.linear_agg(M)

        else:
            A = F.softmax(A, dim=1)  # softmax over N
            if instance_eval:
                total_inst_loss = 0.0
                all_preds = []
                all_targets = []
                inst_labels = F.one_hot(
                    label.to(torch.int64), num_classes=self.n_classes
                ).squeeze()  # binarize label
                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1:  # in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:  # out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(
                                A,
                                h,
                                classifier,
                            )
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

                results_dict = {
                    "instance_loss": total_inst_loss,
                    "inst_labels": np.array(all_targets),
                    "inst_preds": np.array(all_preds),
                }

            M = torch.mm(A, h)

        return M, results_dict

    def forward(self, feats, label, instance_eval=False):
        h, A = self.forward_imaging(feats)
        M, results_dict = self.forward_instance_eval(
            h,
            A,
            label,
            instance_eval=instance_eval,
        )
        logits = self.classifiers(M)
        return logits, A, M, results_dict


class CLAM_LNL(_BaseLNL):
    def __init__(
        self,
        gate,
        size,
        dropout,
        k_sample,
        n_classes,
        n_groups,
        instance_loss_fn,
        subtyping,
        linear_feature,
        multires_aggregation,
        attention_depth,
        classifier_depth,
        n_resolutions,
    ):
        super(CLAM_LNL, self).__init__()
        self.main_model = CLAM_SB_LNL(
            gate=gate,
            size=size,
            dropout=dropout,
            k_sample=k_sample,
            n_classes=n_classes,
            n_groups=n_groups,
            instance_loss_fn=instance_loss_fn,
            subtyping=subtyping,
            linear_feature=linear_feature,
            multires_aggregation=multires_aggregation,
            attention_depth=attention_depth,
            classifier_depth=classifier_depth,
            n_resolutions=n_resolutions,
        )
        self.aux_model = copy.deepcopy(self.main_model.aux_classifiers)
        del self.main_model.aux_classifiers

    def forward(self, feats, label, instance_eval=False, is_adv=True):
        logits, A, M, results_dict = self.main_model.forward(
            feats, label, instance_eval=instance_eval
        )
        if not is_adv:
            M_aux = grad_reverse(M)
        else:
            M_aux = M
        logits_aux = self.aux_model(M_aux)
        return logits, logits_aux, A, results_dict


class CLAM_PL_LNL(BaseMILModel_LNL):
    def __init__(
        self,
        config,
        n_classes,
        n_groups,
        aux_lambda=0.1,
        size=None,
        gate: bool = True,
        dropout=False,
        k_sample: int = 8,
        instance_eval: bool = False,
        instance_loss: str = "ce",
        instance_loss_weight: float = 0.3,
        subtyping: bool = False,
        multibranch=False,
        multires_aggregation=None,
        linear_feature: bool = False,
        attention_depth=None,
        classifier_depth=None,
        n_resolutions: int = 1,
        gradient_clip_value: float = 0.5,
        gradient_clip_algorithm: str = "norm",
    ):
        if n_classes == 1:
            n_classes = 2
        if n_groups == 1:
            n_groups = 2
        super(CLAM_PL_LNL, self).__init__(
            config,
            n_classes=n_classes,
            n_groups=n_groups,
            aux_lambda=aux_lambda,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
            size=size,
            gradient_clip_value=gradient_clip_value,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        self.size = size
        self.dropout = dropout
        self.gate = gate
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.instance_eval = instance_eval
        self.instance_loss_weight = instance_loss_weight
        self.multibranch = multibranch
        self.attention_depth = attention_depth
        self.classifier_depth = classifier_depth
        self.linear_feature = linear_feature

        if not self.multibranch:
            self.model = CLAM_LNL(
                gate=gate,
                size=size,
                dropout=dropout,
                k_sample=k_sample,
                n_classes=n_classes,
                n_groups=n_groups,
                instance_loss_fn=instance_loss,
                subtyping=subtyping,
                linear_feature=linear_feature,
                multires_aggregation=multires_aggregation,
                attention_depth=attention_depth,
                classifier_depth=classifier_depth,
                n_resolutions=n_resolutions,
            )
        else:
            raise NotImplementedError

    def forward(self, batch, is_predict=False, is_adv=True):
        # Batch
        features, target, sensitive_attr = (
            batch["features"],
            batch["labels"],
            batch["labels_group"],
        )

        # Prediction
        logits, preds, logits_aux, _, A, results_dict = self._forward(
            features,
            labels=target,
            instance_eval=self.instance_eval and not is_predict,
            is_adv=is_adv,
        )

        loss = None
        _loss = None
        _loss_aux_adv = None
        _loss_aux_mi = None
        if not is_predict:
            if is_adv:
                # Loss (on logits)
                _loss = self.loss.forward(logits, target.reshape(-1))
                if self.instance_eval:
                    _instance_loss = torch.mean(
                        torch.stack([r["instance_loss"] for r in results_dict])
                    )
                    _loss = (
                        1 - self.instance_loss_weight
                    ) * _loss + self.instance_loss_weight * _instance_loss

                preds_aux = self.aux_act(logits_aux)
                _loss_aux_adv = torch.mean(
                    torch.sum(preds_aux * torch.log(preds_aux + 1e-5), 1)
                )

                loss = _loss + _loss_aux_adv * self.aux_lambda
            else:
                if self.n_classes == 2:
                    _loss_aux_mi = self.loss_aux.forward(
                        logits_aux, sensitive_attr.reshape(-1)
                    )
                else:
                    _loss_aux_mi = self.loss_aux.forward(logits_aux, sensitive_attr)
                loss = _loss_aux_mi

        # Sigmoid or Softmax activation
        if self.n_classes in [1, 2]:
            preds = preds[:, 1]
            preds = torch.unsqueeze(preds, dim=1)

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "main_loss": _loss,
            "aux_adv_loss": _loss_aux_adv,
            "aux_mi_loss": _loss_aux_mi,
            "attention": A,
            "slide_name": batch["slide_name"],
        }

    def _forward(
        self,
        features_batch,
        labels=None,
        instance_eval=False,
        is_adv=True,
    ):
        logits = []
        logits_aux = []
        preds = []
        preds_aux = []
        A = []
        results_dict = []
        for idx, singlePatientFeatures in enumerate(features_batch):
            feats = [
                singlePatientFeatures[f].squeeze() for f in singlePatientFeatures.keys()
            ]

            if labels is not None:
                label = labels[idx]
                label = label.squeeze(dim=0) if label is not None else None
            else:
                label = None

            _logits, _logits_aux, _A, _results_dict = self.model.forward(
                feats, label, instance_eval=instance_eval, is_adv=is_adv
            )
            _preds = torch.nn.functional.softmax(_logits, dim=1)
            _preds_aux = torch.nn.functional.softmax(_logits_aux, dim=1)

            logits.append(_logits)
            logits_aux.append(_logits_aux)
            preds.append(_preds)
            preds_aux.append(_preds_aux)
            A.append(_A)
            results_dict.append(_results_dict)

        return (
            torch.vstack(logits),
            torch.vstack(preds),
            torch.vstack(logits_aux),
            torch.vstack(preds_aux),
            A,
            results_dict,
        )


if __name__ == "__main__":
    from dotmap import DotMap

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

    y = torch.randint(0, 2, [32, 1])

    data = dict(
        features=[dict(features=torch.rand(10, 384)) for _ in range(32)],
        labels=y,
        slide_name="tmp",
    )

    model = CLAM_PL(
        config,
        n_classes=2,
        size=[384, 256, 128],
        dropout=False,
        instance_eval=True,
        attention_depth=1,
        classifier_depth=1,
    )

    o = model.forward(data)

    print(o)
