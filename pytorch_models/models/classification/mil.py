"""
Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images. Nature Biomedical Engineering
"""
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class PredMIL(nn.Module):
    def __init__(
        self,
        size=None,
        dropout=False,
        n_classes=1,
        aggregate: Union[str, float] = "mean",
        agg_level="preds",
        top_k: int = None,
    ):
        super(PredMIL, self).__init__()
        if size is None:
            size = [1024, 512]

        self.size = size
        self.top_k = top_k if top_k is not None else 0
        self.n_classes = n_classes
        self.aggregate = aggregate
        self.agg_level = agg_level

        if isinstance(self.aggregate, float):
            assert 0.0 < self.aggregate < 1.0, "Aggregate must be between 0 and 1"

        _classifier = list()

        for i in range(len(self.size) - 1):
            fc = [nn.Linear(self.size[i], self.size[i + 1]), nn.ReLU()]
            if dropout:
                fc.append(nn.Dropout(0.25))
            _classifier.extend(fc)

        _classifier.append(nn.Linear(size[-1], n_classes))
        self.classifier = nn.Sequential(*_classifier)

        initialize_weights(self)

    def forward(self, x: List[torch.Tensor], return_features=False):
        assert isinstance(x, list), (
            "Please use `FeatureDatasetHDF5.collate` function to load dataset "
            "because each sample may have different number of features."
        )

        batch_logits = []
        batch_probs = []
        features_list = []

        for h in x:
            logits = self.classifier(h)  # K x n_classes
            probs = F.softmax(logits, dim=1) if self.n_classes > 1 else logits.sigmoid()

            if self.top_k != 0:
                if self.agg_level == "logits":
                    top_instance_idx = logits.topk(
                        self.top_k, dim=0, largest=self.top_k > 0, sorted=False
                    )[1].squeeze()
                else:
                    top_instance_idx = probs.topk(
                        self.top_k, dim=0, largest=self.top_k > 0, sorted=False
                    )[1].squeeze()
                logits = torch.index_select(logits, dim=0, index=top_instance_idx)
                probs = torch.index_select(probs, dim=0, index=top_instance_idx)

            if self.aggregate == "mean":
                logits = torch.mean(logits, dim=0)
                probs = torch.mean(probs, dim=0)
            elif self.aggregate == "max":
                logits = torch.max(logits, dim=0)[0]
                probs = torch.max(probs, dim=0)[0]
            elif self.aggregate == "min":
                logits = torch.min(logits, dim=0)[0]
                probs = torch.min(probs, dim=0)[0]
            elif isinstance(self.aggregate, float):
                logits = logits.kthvalue(
                    int(logits.shape[0] * self.aggregate), dim=0, keepdim=False
                )[0]
                probs = probs.kthvalue(
                    int(probs.shape[0] * self.aggregate), dim=0, keepdim=False
                )[0]
            else:
                raise KeyError(f"Aggregate function `{self.aggregate}` not known.")

            batch_logits.append(logits)
            batch_probs.append(probs)

            if return_features and self.top_k != 0:
                top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            else:
                top_features = h
            features_list.append(top_features)

        logits = torch.vstack(batch_logits)
        probs = torch.vstack(batch_probs)
        if return_features:
            results_dict = dict(features=torch.stack(features_list, dim=0))
            return logits, probs, results_dict
        return logits, probs


class FeatureMIL(nn.Module):
    def __init__(
        self,
        size=None,
        dropout=False,
        n_classes=1,
        aggregates: Union[str, List[str]] = "mean",
        top_k=1,
    ):
        super(FeatureMIL, self).__init__()
        if size is None:
            size = [1024, 512]

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
                top_instance_idx = torch.topk(y_probs, self.top_k, dim=0)[1].squeeze()
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
            results_dict = dict(features=torch.vstack(batch_results_dict))
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
        mil_type="pred",
        agg_level="preds",
        aggregates: Union[str, List[str]] = "mean",
        top_k: int = 1,
        dropout=False,
        multires_aggregation=None,
    ):
        self.size = size
        self.top_k = top_k
        self.dropout = dropout
        self.aggregates = aggregates
        self.mil_type = mil_type
        self.agg_level = agg_level
        self.multires_aggregation = multires_aggregation
        super(MIL_PL, self).__init__(config, n_classes=n_classes)

        if n_classes == 2:
            self.n_classes = 1
            n_classes = 1

        assert self.mil_type in ["pred", "features", "clam_mil"]

        if self.mil_type == "pred":
            assert self.agg_level in ["preds", "logits"]

        self.loss = nn.CrossEntropyLoss()

        if self.mil_type == "clam_mil":
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
        elif self.mil_type == "features":
            self.model = FeatureMIL(
                size=self.size,
                dropout=self.dropout,
                n_classes=self.n_classes,
                aggregates=self.aggregates,
                top_k=self.top_k,
            )
        elif self.mil_type == "pred":
            self.model = PredMIL(
                size=self.size,
                dropout=self.dropout,
                n_classes=self.n_classes,
                aggregate=self.aggregates,
                agg_level=self.agg_level,
                top_k=self.top_k,
            )

    def forward(self, batch: dict[str, torch.Tensor]):
        if (
            "top_k" in self.aggregates
            or "clam" in self.aggregates
            or "clam_mil" in self.aggregates
        ):
            # Batch
            features, target = batch
            # Prediction
            logits, preds, _, _, results_dict = self._forward(features)
            # Loss (on logits)
            loss = self.loss.forward(logits, target.squeeze())

            preds = preds[:, 1]

            return {
                "features": results_dict["features"],
                "target": target,
                "preds": preds,
                "loss": loss,
            }
        else:
            # Batch
            features, target = batch
            # Prediction
            logits, preds = self._forward(features)
            # Loss (on logits)
            loss = self.loss.forward(logits.float(), target.float())

            return {
                "target": target,
                "preds": preds,
                "loss": loss,
            }

    def _forward(self, features):
        _data = []
        for data in features:
            h = [data[key] for key in data]
            h = aggregate_features(h, method=self.multires_aggregation)
            _data.append(h)
        return self.model.forward(_data)

    def _log_metrics(self, preds, target, loss, mode):
        on_step = False if mode != "train" else True
        # https://github.com/Lightning-AI/lightning/issues/13210
        sync_dist = self.sync_dist and (
            mode == "val" or mode == "test" or mode == "eval"
        )
        if mode == "val":
            metrics = self.val_metrics
        elif mode == "train":
            metrics = self.train_metrics
        elif mode == "test":
            metrics = self.test_metrics

        self._compute_metrics(preds, target, mode)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{mode}_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
            batch_size=self.batch_size,
        )


if __name__ == "__main__":
    # create test data and model
    _batch = 32
    _n_features = 384
    _n_classes = 1
    _n_samples = 100

    _features = [torch.rand(_n_samples, _n_features) for _ in range(_batch)]

    for aggregate in ["mean", "max", "min", 0.25, 0.5, 0.75]:
        model = PredMIL(
            size=[384, 256, 128],
            dropout=True,
            n_classes=1,
            aggregate=aggregate,
            agg_level="probs",
            top_k=10,
        )

        # test forward pass
        _logits, _probs, _results_dict = model.forward(_features, return_features=True)
        print("------------------------")
        print("aggregate:", aggregate)
        print(_logits.shape)
        print(_probs.shape)
        print(_results_dict["features"].shape)
