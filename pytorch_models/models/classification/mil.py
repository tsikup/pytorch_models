"""
Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images. Nature Biomedical Engineering
"""
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseMILModel


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MIL(nn.Module):
    def __init__(
        self,
        size=[1024, 512],
        dropout=False,
        n_classes=1,
        aggregates: Union[str, List[str]] = "mean",
        top_k=1,
    ):
        super(MIL, self).__init__()
        self.size = size
        self.top_k = top_k
        self.n_classes = n_classes
        self.aggregates = self._define_aggregate(aggregates)

        _classifier = list()

        for i in range(len(self.size) - 1):
            fc = [nn.Linear(self.size[i], self.size[i + 1]), nn.ReLU()]
            if dropout:
                fc.append(nn.Dropout(0.25))
            _classifier.extend(fc)

        _classifier.append(nn.Linear(size[-1], n_classes))
        self.classifier = nn.Sequential(*_classifier)

        initialize_weights(self)

    def _define_aggregate(self, aggregates):
        if aggregates is None:
            if self.top_k is None or self.top_k < 1:
                self.top_k = 1
            aggregates = ["top_k"]

        if not isinstance(aggregates, list):
            aggregates = [aggregates]
        if "top_k" in aggregates:
            assert len(aggregates) == 1, (
                "If aggregate includes `top_k`, then it should be the only aggregate to consider. "
                "Please delete other aggregate strategies."
            )

        print(
            "Calculating new input number of features based on aggregation functions. "
            "e.g. if two or more aggregation functions are defined, "
            "then the input size will be double, one for each function."
        )
        self.size[0] = (
            self.size[0] * len(aggregates)
            if "top_k" not in aggregates
            else self.size[0]
        )
        return aggregates

    def forward(self, x, return_features=False):
        assert isinstance(x, list), (
            "Please use `FeatureDatasetHDF5.collate` function to load dataset "
            "because each sample may have different number of features."
        )

        batch_logits = []
        batch_top_instance_logits = []
        batch_Y_prob = []
        batch_Y_hat = []
        batch_y_probs = []
        batch_results_dict = []

        for h in x:
            _h = []
            for aggregate in self.aggregates:
                if aggregate == "mean":
                    _h.append(torch.mean(h, dim=0))
                elif aggregate == "max":
                    _h.append(torch.max(h, dim=0)[0])
                elif aggregate == "min":
                    _h.append(torch.min(h, dim=0)[0])
                elif aggregate == "topk":
                    _h = h
                else:
                    raise KeyError(f"Aggregate function `{aggregate}` not known.")

            h = torch.cat(_h)

            logits = self.classifier(h)  # K x n_classes
            batch_logits.append(logits)

            if not "top_k" in self.aggregates:
                continue

            if self.n_classes == 1:
                y_probs = logits.sigmoid()
                top_instance_idx = torch.topk(y_probs, self.top_k, dim=0)[1].view(
                    1,
                )
                top_instance_logits = torch.index_select(
                    logits, dim=0, index=top_instance_idx
                )
                Y_hat = torch.topk(top_instance_logits, 1, dim=1)[1]
                Y_prob = top_instance_logits.sigmoid()
            else:
                y_probs = F.softmax(logits, dim=1)
                m = y_probs.view(1, -1).argmax(1)
                top_indices = torch.cat(
                    (
                        (m // self.n_classes).view(-1, 1),
                        (m % self.n_classes).view(-1, 1),
                    ),
                    dim=1,
                ).view(-1, 1)
                top_instance_logits = logits[top_indices[0]]
                Y_hat = top_indices[1]
                Y_prob = y_probs[top_indices[0]]
                top_instance_idx = top_indices[0]

            batch_top_instance_logits.append(top_instance_logits)
            batch_Y_prob.append(Y_prob)
            batch_Y_hat.append(Y_hat)
            batch_y_probs.append(y_probs)

            if return_features:
                top_features = torch.index_select(h, dim=0, index=top_instance_idx)
                batch_results_dict.append(top_features)

        if not "top_k" in self.aggregates:
            logits = torch.vstack(batch_logits)
            return logits
        else:
            top_instance_logits = torch.vstack(batch_top_instance_logits)
            Y_prob = torch.vstack(batch_Y_prob)
            Y_hat = torch.vstack(batch_Y_hat)
            y_probs = torch.vstack(batch_y_probs)
            results_dict = dict(results_dict=torch.vstack(batch_results_dict))
            return top_instance_logits, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc(nn.Module):
    def __init__(self, size, dropout=False, n_classes=2, top_k=1):
        super(MIL_fc, self).__init__()
        assert n_classes == 2
        assert len(size) == 2
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(size[1], n_classes))
        self.classifier = nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k = top_k

    def forward(self, x, return_features=False):
        assert isinstance(x, list), (
            "Please use `FeatureDatasetHDF5.collate` function to load dataset "
            "because each sample may have different number of features."
        )

        batch_logits = []
        batch_top_instance_logits = []
        batch_Y_prob = []
        batch_Y_hat = []
        batch_y_probs = []
        batch_results_dict = []

        results_dict = dict(features=None)

        for _h in x:
            h = _h
            if return_features:
                h = self.classifier.module[:3](h)
                logits = self.classifier.module[3](h)
            else:
                logits = self.classifier(h)  # K x 1

            y_probs = F.softmax(logits, dim=1)
            top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(
                1,
            )
            top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
            Y_hat = torch.topk(top_instance, 1, dim=1)[1]
            Y_prob = F.softmax(top_instance, dim=1)

            batch_logits.append(logits)
            batch_top_instance_logits.append(top_instance)
            batch_Y_prob.append(Y_prob)
            batch_Y_hat.append(Y_hat)
            batch_y_probs.append(y_probs)

            if return_features:
                top_features = torch.index_select(h, dim=0, index=top_instance_idx)
                batch_results_dict.append(top_features)

        top_instance = torch.vstack(batch_top_instance_logits)
        Y_prob = torch.vstack(batch_Y_prob)
        Y_hat = torch.vstack(batch_Y_hat)
        y_probs = torch.vstack(batch_y_probs)
        if return_features:
            results_dict = dict(features=torch.vstack(batch_results_dict))

        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    def __init__(self, size, dropout=False, n_classes=2, top_k=1):
        super(MIL_fc_mc, self).__init__()
        assert n_classes > 2
        assert len(size) == 2
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList(
            [nn.Linear(size[1], 1) for i in range(n_classes)]
        )
        initialize_weights(self)
        self.top_k = top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def forward(self, x, return_features=False):
        assert isinstance(x, list), (
            "Please use `FeatureDatasetHDF5.collate` function to load dataset "
            "because each sample may have different number of features."
        )

        batch_logits = []
        batch_top_instance_logits = []
        batch_Y_prob = []
        batch_Y_hat = []
        batch_y_probs = []
        batch_results_dict = []

        for _h in x:
            h = _h

            h = self.fc(h)
            logits = torch.empty(h.size(0), self.n_classes).float()

            for c in range(self.n_classes):
                if isinstance(self.classifiers, nn.DataParallel):
                    logits[:, c] = self.classifiers.module[c](h).squeeze(1)
                else:
                    logits[:, c] = self.classifiers[c](h).squeeze(1)

            y_probs = F.softmax(logits, dim=1)
            m = y_probs.view(1, -1).argmax(1)
            top_indices = torch.cat(
                ((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)),
                dim=1,
            ).view(-1, 1)
            top_instance = logits[top_indices[0]]

            Y_hat = top_indices[1]
            Y_prob = y_probs[top_indices[0]]

            batch_logits.append(logits)
            batch_top_instance_logits.append(top_instance)
            batch_Y_prob.append(Y_prob)
            batch_Y_hat.append(Y_hat)
            batch_y_probs.append(y_probs)

            if return_features:
                top_features = torch.index_select(h, dim=0, index=top_indices[0])
                batch_results_dict.append(top_features)

        top_instance = torch.vstack(batch_top_instance_logits)
        Y_prob = torch.vstack(batch_Y_prob)
        Y_hat = torch.vstack(batch_Y_hat)
        y_probs = torch.vstack(batch_y_probs)
        results_dict = dict(results_dict=torch.vstack(batch_results_dict))

        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_PL(BaseMILModel):
    def __init__(
        self,
        config,
        n_classes,
        size=[1024, 512],
        aggregates: Union[str, List[str]] = "mean",
        top_k: int = 1,
        dropout=False,
    ):
        super(MIL_PL, self).__init__(config, n_classes=n_classes, in_features=size[0])
        self.size = size
        self.top_k = top_k
        self.dropout = dropout
        self.aggregates = aggregates

        self.loss = nn.CrossEntropyLoss()

        if self.aggregates == "clam_mil" or "clam_mil" in self.aggregates:
            print("Using CLAM's MIL model")
            if self.n_classes in [1, 2]:
                self.model = MIL_fc(
                    size=size, dropout=dropout, n_classes=2, top_k=self.top_k
                )
            else:
                self.model = MIL_fc_mc(
                    size=size,
                    dropout=dropout,
                    n_classes=self.n_classes,
                    top_k=self.top_k,
                )
        else:
            self.model = MIL(
                size=self.size,
                dropout=self.dropout,
                n_classes=self.n_classes,
                aggregates=self.aggregates,
                top_k=self.top_k,
            )

    def forward_shared(self, batch):
        if (
            "top_k" in self.aggregates
            or "clam" in self.aggregates
            or "clam_mil" in self.aggregates
        ):
            # Batch
            features, target = batch
            # Prediction
            logits, preds, _, _, results_dict = self.forward(features)
            # Loss (on logits)
            loss = self.loss.forward(logits, target.squeeze())
            # loss = self.loss.forward(logits, target.float())

            preds = preds[:, 1]

            return {
                "features": results_dict["features"],
                "target": target,
                "preds": preds,
                "loss": loss,
            }
        else:
            return super(MIL_PL, self).forward_shared(batch)

    def forward(self, x):
        return self.model.forward(x)
