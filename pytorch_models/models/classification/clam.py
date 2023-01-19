"""
Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images. Nature Biomedical Engineering
"""
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from topk.svm import SmoothTop1SVM
from ..base import BaseMILModel
from ..ssl_features.vit import ViT
from ..ssl_features.resnets import ResNet50_SimCLR


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


class CLAM_SB(nn.Module):
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
    """

    size_dict = {
        "small": [1024, 512, 256],
        "big": [1024, 512, 384],
        "resnet": [2048, 1024, 512],
        "vit": [384, 256, 128],
    }

    def __init__(
        self,
        gate=True,
        size_arg="vit",
        dropout=False,
        k_sample=8,
        n_classes=2,
        instance_loss_fn="ce",
        subtyping=False,
        multires_aggregation=None,  # dict(mode, features, attention)
        autoscale_network=True,
        attention_depth=None,
        classifier_depth=None,
        legacy=True,
    ):
        super(CLAM_SB, self).__init__()
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.multires_aggregation = multires_aggregation
        self.concat_attentions = False
        self.legacy = legacy

        if legacy and self.multires_aggregation is not None:
            self.multires_aggregation["feature_level"] = 0

        if instance_loss_fn == "svm":
            self.instance_loss_fn = SmoothTop1SVM(n_classes=n_classes)
            self.instance_loss_on_gpu = False
            # if torch.cuda.is_available():
            #     self.instance_loss_fn = self.instance_loss_fn.cuda()
        else:
            self.instance_loss_fn = nn.CrossEntropyLoss()

        if isinstance(size_arg, list):
            size = size_arg
        else:
            size = size = self.size_dict[size_arg]

        assert (
            attention_depth is not None
        ), "Please set the attention module starting size as an index of the size array."
        assert (
            classifier_depth is not None
        ), "Please set the classification module starting size as an index of the size array."

        self.attention_depth = attention_depth

        if isinstance(classifier_depth, int):
            self.classifier_size = [size[classifier_depth]]
        elif isinstance(classifier_depth, list):
            assert (
                len(classifier_depth) == 2
            ), "Please give the classifier depth indices as [first, last] for multilayer or int (only one layer)"
            self.classifier_size = size[classifier_depth[0] : classifier_depth[1] + 1]
        else:
            raise TypeError(
                "Please give the classifier depth indices as [first, last] for multilayer or int (only one layer)"
            )

        if not isinstance(self.classifier_size, list):
            self.classifier_size = [self.classifier_size]
        assert (
            self.classifier_size[0] == size[self.attention_depth]
        ), "Mismatch between attention module output feature size and classifiers' input feature size"

        if (
            autoscale_network
            and self.multires_aggregation is not None
            and self.multires_aggregation["features"] == "concat"
        ):
            size[0] = 2 * size[0]
            size[1] = 2 * size[1]

        self.target_net, self.attention_net = self._create_attention_model(
            size, dropout, gate, n_classes=1
        )

        if self.multires_aggregation is not None and self.multires_aggregation[
            "features"
        ] in ["linear", "nonlinear"]:
            self.linear_features_target = nn.Linear(size[0], size[0], bias=False)
            self.linear_features_context = nn.Linear(size[0], size[0], bias=False)

        if (
            self.multires_aggregation is not None
            and self.multires_aggregation["attention"] is not None
        ):
            self.context_net, self.attention_context_net = self._create_attention_model(
                size, dropout, gate, n_classes=1
            )
            assert self.multires_aggregation["attention"] not in [
                "linear",
                "nonlinear",
            ], "Multiresolution integration at the attention level is enabled.. The aggregation function must not be linear for the attention vectors."
            assert (
                self.multires_aggregation["attention"] != "concat"
            ), "Multiresolution integration at the attention level is enabled.. The aggregation function must not be concat for the attention vectors."
        elif (
            self.multires_aggregation is not None
            and self.multires_aggregation["feature_level"] > 0
        ):
            self.context_net, _ = self._create_attention_model(
                size, dropout, gate, n_classes=1
            )

        if len(self.classifier_size) > 1:
            _classifiers = []
            _instance_classifiers = dict()
            for idx, _ in enumerate(self.classifier_size[:-1]):
                _classifiers.append(
                    nn.Linear(self.classifier_size[idx], self.classifier_size[idx + 1])
                )
            _classifiers.append(nn.Linear(self.classifier_size[-1], n_classes))
            self.classifiers = nn.Sequential(*_classifiers)

            for c in range(n_classes):
                _tmp_instance_classifier = []
                for idx, _ in enumerate(self.classifier_size[:-1]):
                    _tmp_instance_classifier.append(
                        nn.Linear(
                            self.classifier_size[idx], self.classifier_size[idx + 1]
                        )
                    )
                _tmp_instance_classifier.append(nn.Linear(self.classifier_size[-1], 2))
                _instance_classifiers[c] = nn.Sequential(*_tmp_instance_classifier)
            self.instance_classifiers = nn.ModuleList(
                [_instance_classifiers[c] for c in range(n_classes)]
            )
        else:
            self.classifiers = nn.Linear(self.classifier_size[0], n_classes)

            instance_classifiers = [
                nn.Linear(self.classifier_size[0], 2) for i in range(n_classes)
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
        if self.legacy:
            fc.append(attention_net)
            return None, nn.Sequential(*fc)
        return nn.Sequential(*fc), attention_net

    # def relocate(self):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.attention_net = self.attention_net.to(device)
    #     self.classifiers = self.classifiers.to(device)
    #     self.instance_classifiers = self.instance_classifiers.to(device)

    # @staticmethod
    # def create_positive_targets(length):
    #     return torch.full((length,), 1).long()

    # @staticmethod
    # def create_negative_targets(length):
    #     return torch.full((length,), 0).long()

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
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)
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
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def _aggregate_multires_features(self, h, h_context, method, is_attention=False):
        if method == "concat":
            h = torch.cat([h, h_context], dim=1)
        elif method == "average" or method == "mean":
            h = torch.dstack((h, h_context))
            h = torch.mean(h, dim=-1)
        elif method == "max":
            h = torch.dstack((h, h_context))
            h = torch.max(h, dim=-1)
        elif method == "min":
            h = torch.dstack((h, h_context))
            h = torch.min(h, dim=-1)
        elif method == "mul":
            h = torch.mul(h, h_context)
        elif method == "add":
            h = torch.add(h, h_context)
        elif method in ["linear", "nonlinear"]:
            if not is_attention:
                h = torch.add(
                    self.linear_features_target(h),
                    self.linear_features_context(h_context),
                )
                if method == "nonlinear":
                    h = nn.functional.relu(h)
            else:
                raise Exception(
                    "Attention vectors cannot be integrated with linear layer yet."
                )
        else:
            pass
        return h

    def apply_attention_net(self, h, process_net, attention_net):
        if self.legacy:
            return attention_net(h)
        h = process_net(h)
        return attention_net(h)

    def forward(
        self,
        h,
        h_context=None,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        if self.multires_aggregation is not None:
            assert (
                h_context is not None
            ), "Multiresolution is enabled.. h_context features should not be None."
            if self.multires_aggregation["attention"] is None:
                if self.multires_aggregation["feature_level"] <= 0:
                    h = self._aggregate_multires_features(
                        h,
                        h_context,
                        method=self.multires_aggregation["features"],
                        is_attention=False,
                    )
                else:
                    h_context = self.context_net(h_context)

                A, h = self.apply_attention_net(
                    h, self.target_net, self.attention_net
                )  # NxK
                A = torch.transpose(A, 1, 0)  # KxN

                if self.multires_aggregation["feature_level"] > 0:
                    h = self._aggregate_multires_features(
                        h,
                        h_context,
                        method=self.multires_aggregation["features"],
                        is_attention=False,
                    )
            else:
                A, h = self.apply_attention_net(h, self.target_net, self.attention_net)
                A = torch.transpose(A, 1, 0)  # KxN
                A_context, h_context = self.apply_attention_net(
                    h_context, self.context_net, self.attention_context_net
                )
                A_context = torch.transpose(A_context, 1, 0)  # KxN

                if self.multires_aggregation["attention"] != "late":
                    A = self._aggregate_multires_features(
                        A,
                        A_context,
                        method=self.multires_aggregation["attention"],
                        is_attention=True,
                    )
                    h = self._aggregate_multires_features(
                        h,
                        h_context,
                        method=self.multires_aggregation["features"],
                        is_attention=False,
                    )
        else:
            A, h = self.apply_attention_net(h, self.target_net, self.attention_net)
            A = torch.transpose(A, 1, 0)  # KxN

        if self.multires_aggregation["attention"] == "late":
            if attention_only:
                return A, A_context
            A_raw = A
            A_context_raw = A_context
            A = F.softmax(A, dim=1)  # softmax over N
            A_context = F.softmax(A_context, dim=1)  # softmax over N

            M = torch.mm(A, h)
            M_context = torch.mm(A_context, h_context)
            M = self._aggregate_multires_features(
                M,
                M_context,
                method=self.multires_aggregation["features"],
                is_attention=False,
            )

            logits = self.classifiers(M)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            Y_prob = F.softmax(logits, dim=1)
            if instance_eval:
                pass
            else:
                results_dict = {}
            if return_features:
                results_dict.update({"features": M})
            else:
                results_dict.update({"features": None})
            return logits, Y_prob, Y_hat, A_raw, results_dict
        else:
            if attention_only:
                return A
            A_raw = A
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
                                A, h, classifier
                            )
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)

            M = torch.mm(A, h)
            logits = self.classifiers(M)
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            Y_prob = F.softmax(logits, dim=1)
            if instance_eval:
                results_dict = {
                    "instance_loss": total_inst_loss,
                    "inst_labels": np.array(all_targets),
                    "inst_preds": np.array(all_preds),
                }
            else:
                results_dict = {}
            if return_features:
                results_dict.update({"features": M})
            else:
                results_dict.update({"features": None})
            return logits, Y_prob, Y_hat, A_raw, results_dict


# class CLAM_MB(CLAM_SB):
#     def __init__(
#         self,
#         gate=True,
#         size_arg="vit",
#         dropout=False,
#         k_sample=8,
#         n_classes=2,
#         instance_loss_fn=nn.CrossEntropyLoss(),
#         subtyping=False,
#         multires_aggregation=None,
#     ):
#         super().__init__(
#             gate,
#             size_arg,
#             dropout,
#             k_sample,
#             n_classes,
#             instance_loss_fn,
#             subtyping,
#             multires_aggregation,
#         )

#         size = self.size_dict[size_arg]

#         if self.multires_aggregation is not None and self.multires_aggregation['mode'] != 'attention' and self.multires_aggregation['features'] == 'concat':
#             size[0] = 2 * size[0]
#             size[1] = 2 * size[1]

#         self.attention_net = self._create_attention_model(size, dropout, gate, n_classes=self.n_classes)

#         if self.multires_aggregation is not None and self.multires_aggregation['mode'] == 'attention':
#             self.attention_context_net = self._create_attention_model(size, dropout, gate, n_classes=self.n_classes)
#             assert self.multires_aggregation['attention'] != 'conat', "Multiresolution integration at the attention level is enabled.. The aggregation function must not be concat for the attention vectors."

#         bag_classifiers = [
#             nn.Linear(size[1], 1) for i in range(n_classes)
#         ]  # use an indepdent linear layer to predict each class
#         self.classifiers = nn.ModuleList(bag_classifiers)
#         instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
#         self.instance_classifiers = nn.ModuleList(instance_classifiers)

#         initialize_weights(self)

#     def forward(
#         self,
#         h,
#         h_context=None,
#         label=None,
#         instance_eval=False,
#         return_features=False,
#         attention_only=False,
#     ):
#         device = h.device

#         if self.multires_aggregation is not None:
#             assert h_context is not None, "Multiresolution is enabled.. h_context features should not be None."
#             if self.multires_aggregation['mode'] != 'attention':
#                 h = self._aggregate_multires_features(h, h_context, method=self.multires_aggregation['features'])
#                 A, h = self.attention_net(h)  # NxK
#                 A = torch.transpose(A, 1, 0)  # KxN
#             else:
#                 A, h = self.attention_net(h)
#                 A = torch.transpose(A, 1, 0)  # KxN
#                 A_context, h_context = self.attention_context_net(h_context)
#                 A_context = torch.transpose(A_context, 1, 0) # KxN
#                 A = self._aggregate_multires_features(A, A_context, method=self.multires_aggregation['attention'])
#                 h = self._aggregate_multires_features(h, h_context, method=self.multires_aggregation['features'])
#         else:
#             A, h = self.attention_net(h)
#             A = torch.transpose(A, 1, 0)  # KxN

#         if attention_only:
#             return A
#         A_raw = A
#         A = F.softmax(A, dim=1)  # softmax over N

#         if instance_eval:
#             total_inst_loss = 0.0
#             all_preds = []
#             all_targets = []
#             inst_labels = F.one_hot(
#                 label.to(torch.int64), num_classes=self.n_classes
#             ).squeeze()  # binarize label
#             for i in range(len(self.instance_classifiers)):
#                 inst_label = inst_labels[i].item()
#                 classifier = self.instance_classifiers[i]
#                 if inst_label == 1:  # in-the-class:
#                     instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
#                     all_preds.extend(preds.cpu().numpy())
#                     all_targets.extend(targets.cpu().numpy())
#                 else:  # out-of-the-class
#                     if self.subtyping:
#                         instance_loss, preds, targets = self.inst_eval_out(
#                             A[i], h, classifier
#                         )
#                         all_preds.extend(preds.cpu().numpy())
#                         all_targets.extend(targets.cpu().numpy())
#                     else:
#                         continue
#                 total_inst_loss += instance_loss

#             if self.subtyping:
#                 total_inst_loss /= len(self.instance_classifiers)

#         M = torch.mm(A, h)
#         logits = torch.empty(1, self.n_classes).float().to(device)
#         for c in range(self.n_classes):
#             logits[0, c] = self.classifiers[c](M[c])
#         Y_hat = torch.topk(logits, 1, dim=1)[1]
#         Y_prob = F.softmax(logits, dim=1)
#         if instance_eval:
#             results_dict = {
#                 "instance_loss": total_inst_loss,
#                 "inst_labels": np.array(all_targets),
#                 "inst_preds": np.array(all_preds),
#             }
#         else:
#             results_dict = {}
#         if return_features:
#             results_dict.update({"features": M})
#         else:
#             results_dict.update({"features": None})
#         return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_Image_PL(BaseMILModel):
    def __init__(
        self,
        config,
        n_classes,
        size_arg="resnet",
        gate: bool = True,
        dropout=False,
        k_sample: int = 8,
        instance_eval: bool = False,
        instance_loss: str = "ce",
        instance_loss_weight: float = 0.3,
        subtyping: bool = False,
        multires_aggregation=None,
        multibranch=False,
        feature_extractor="vit",
        ckpt_dir=None,
        processing_batch_size=100,
        autoscale_network=True,
        attention_depth=None,
        classifier_depth=None,
    ):
        super(CLAM_Image_PL, self).__init__(config, n_classes=n_classes)

        self.size_arg = size_arg
        self.feature_extractor = feature_extractor
        self.dropout = dropout
        self.gate = gate
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.instance_eval = instance_eval
        self.instance_loss_weight = instance_loss_weight
        self.multires_aggregation = multires_aggregation
        self.multibranch = multibranch
        self.processing_batch_size = processing_batch_size

        if self.feature_extractor == "vit" and ckpt_dir is not None:
            self.backbone = ViT(
                arch="small", ckpt=os.path.join(ckpt_dir, "vits_tcga_brca_dino.pt")
            )
            self.backbone.freeze()
        elif self.feature_extractor == "resnet" and ckpt_dir is not None:
            self.backbone = ResNet50_SimCLR(
                ckpt=os.path.join(ckpt_dir, "resnet50_tcga_brca_simclr.pt")
            )
            self.backbone.freeze()

        if not self.multibranch:
            self.model = CLAM_SB(
                gate=self.gate,
                size_arg=size_arg,
                dropout=dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                subtyping=self.subtyping,
                instance_loss_fn=instance_loss,
                multires_aggregation=self.multires_aggregation,
                autoscale_network=autoscale_network,
                attention_depth=attention_depth,
                classifier_depth=classifier_depth,
            )
        # else:
        #     self.model = CLAM_MB(
        #         gate=self.gate,
        #         size_arg=size_arg,
        #         dropout=dropout,
        #         k_sample=self.k_sample,
        #         n_classes=self.n_classes,
        #         subtyping=self.subtyping,
        #         instance_loss_fn=self.instance_loss,
        #         multires_aggregation=self.multires_aggregation,
        #     )

    def compute_features(self, patches):
        if patches is None:
            return None
        assert (
            patches.shape[0] == 1
        ), "Image-CLAM works only with a batch size of 1 at the moment."
        patches = patches[0]
        features = []
        _batch_size = self.processing_batch_size
        if _batch_size > 0:
            for i in range(0, len(patches), _batch_size):
                _patches = patches[i : i + _batch_size]
                features.append(self.backbone.forward(_patches))
            features = torch.vstack(features)  # .to(patches.device)
        else:
            features = self.backbone.forward(patches)
        return features

    def forward_shared(self, batch, is_predict=False):
        # Batch
        patches, target = batch

        images = dict()
        images["images"] = self.compute_features(patches["images"])
        if self.multires_aggregation is not None and "images_context" in patches:
            images["images_context"] = self.compute_features(patches["images_context"])
        else:
            images["images_context"] = None

        # Prediction
        logits, preds, _, _, results_dict = self.forward(
            h=images["images"],
            h_context=images["images_context"],
            label=target,
            instance_eval=self.instance_eval and not is_predict,
            return_features=False,
            attention_only=False,
        )

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss = self.loss.forward(logits, target.squeeze(dim=1))
            if self.instance_eval:
                loss = (
                    1 - self.instance_loss_weight
                ) * loss + self.instance_loss_weight * results_dict["instance_loss"]

        preds = preds[:, 1]
        preds = torch.unsqueeze(preds, dim=1)

        return {
            "features": results_dict["features"],
            "target": target,
            "preds": preds,
            "loss": loss,
        }

    def forward(
        self,
        h,
        h_context=None,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        h = h.squeeze()
        if h_context is not None:
            h_context = h_context.squeeze()
        if label is not None:
            label = label.squeeze(dim=0)

        return self.model.forward(
            h=h,
            h_context=h_context,
            label=label,
            instance_eval=instance_eval,
            return_features=return_features,
            attention_only=attention_only,
        )


class CLAM_Features_PL(BaseMILModel):
    def __init__(
        self,
        config,
        n_classes,
        size_arg="resnet",
        gate: bool = True,
        dropout=False,
        k_sample: int = 8,
        instance_eval: bool = False,
        instance_loss: str = "ce",
        instance_loss_weight: float = 0.3,
        subtyping: bool = False,
        multires_aggregation=None,
        multibranch=False,
        autoscale_network=True,
        attention_depth=None,
        classifier_depth=None,
    ):
        super(CLAM_Features_PL, self).__init__(config, n_classes=n_classes)

        self.size_arg = size_arg
        self.dropout = dropout
        self.gate = gate
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.instance_eval = instance_eval
        self.instance_loss_weight = instance_loss_weight
        self.multires_aggregation = multires_aggregation
        self.multibranch = multibranch

        if not self.multibranch:
            self.model = CLAM_SB(
                gate=self.gate,
                size_arg=size_arg,
                dropout=dropout,
                k_sample=self.k_sample,
                n_classes=self.n_classes,
                subtyping=self.subtyping,
                instance_loss_fn=instance_loss,
                multires_aggregation=self.multires_aggregation,
                autoscale_network=autoscale_network,
                attention_depth=attention_depth,
                classifier_depth=classifier_depth,
            )
        # else:
        #     self.model = CLAM_MB(
        #         gate=self.gate,
        #         size_arg=size_arg,
        #         dropout=dropout,
        #         k_sample=self.k_sample,
        #         n_classes=self.n_classes,
        #         subtyping=self.subtyping,
        #         instance_loss_fn=self.instance_loss,
        #         multires_aggregation=self.multires_aggregation,
        #     )

    def forward_shared(self, batch, is_predict=False):
        # Batch
        features, target = batch

        # Prediction
        logits, preds, _, _, results_dict = self.forward(
            h=features["features"],
            h_context=features["features_context"]
            if self.multires_aggregation is not None and "features_context" in features
            else None,
            label=target,
            instance_eval=self.instance_eval and not is_predict,
            return_features=False,
            attention_only=False,
        )

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss = self.loss.forward(logits, target.squeeze(dim=1))
            if self.instance_eval:
                loss = (
                    1 - self.instance_loss_weight
                ) * loss + self.instance_loss_weight * results_dict["instance_loss"]

        if self.n_classes in [1, 2]:
            preds = preds[:, 1]
            preds = torch.unsqueeze(preds, dim=1)

        return {
            "features": results_dict["features"],
            "target": target,
            "preds": preds,
            "loss": loss,
        }

    def forward(
        self,
        h,
        h_context=None,
        label=None,
        instance_eval=False,
        return_features=False,
        attention_only=False,
    ):
        h = h.squeeze()
        if h_context is not None:
            h_context = h_context.squeeze()
        if label is not None:
            label = label.squeeze(dim=0)

        return self.model.forward(
            h=h,
            h_context=h_context,
            label=label,
            instance_eval=instance_eval,
            return_features=return_features,
            attention_only=attention_only,
        )
