from typing import Union

import torch
from pytorch_models.models.base import BaseModel
from torch import nn

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "hardswish": nn.Hardswish,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "hardsigmoid": nn.Hardsigmoid,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "softmin": nn.Softmin,
}


class NNClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        depth: Union[int, None] = 4,
        activation: str = "relu",
        dropout=0.5,
    ):
        super(NNClassifier, self).__init__()

        features = dict()

        if isinstance(in_features, list):
            assert (
                depth is None
            ), "When you set in_features as a list, depth must be None."
            depth = len(in_features) - 1
            for d in range(depth):
                features[d + 1] = dict()
                features[d + 1]["in"] = in_features[d]
                features[d + 1]["out"] = in_features[d + 1]
            features[depth + 1] = dict()
            features[depth + 1]["in"] = in_features[-1]
        else:
            assert depth is not None
            for d in range(1, depth + 1):
                features[d] = dict()
                features[d]["in"] = in_features
                features[d]["out"] = in_features // 2
                in_features = in_features // 2
            features[depth + 1] = dict()
            features[depth + 1]["in"] = in_features

        for d in range(1, depth + 1):
            self.add_module(
                name=f"linear_{d-1}",
                module=nn.Linear(
                    in_features=features[d]["in"], out_features=features[d]["out"]
                ),
            )
            self.add_module(
                name=f"activation_{d-1}", module=ACTIVATION_FUNCTIONS[activation]()
            )
            if dropout:
                self.add_module(name=f"dropout_{d-1}", module=nn.Dropout(p=dropout))

        if n_classes is not None:
            self.add_module(
                name="linear_out",
                module=nn.Linear(features[depth + 1]["in"], n_classes),
            )

    def forward(self, x):
        y = x
        for layer in self.children():
            y = layer(y)
        return y


class NNClassifierPL(BaseModel):
    def __init__(self, config, size, n_classes, dropout=True):
        if n_classes == 2:
            n_classes = 1
        super(NNClassifierPL, self).__init__(
            config=config, n_classes=n_classes, in_channels=size[0]
        )

        self.model = NNClassifier(
            in_features=size,
            n_classes=n_classes,
            depth=None,
            dropout=dropout,
            activation="relu",
        )

    def _forward(self, x):
        return self.model.forward(x)


class NNClassifierMILPL(BaseModel):
    def __init__(
        self,
        config,
        size,
        n_classes,
        dropout=True,
        aggregation="mean",
        agg_level="probs",
        top_k=0,
    ):
        if n_classes == 2:
            n_classes = 1
        super(NNClassifierMILPL, self).__init__(
            config=config, n_classes=n_classes, in_channels=size[0]
        )

        self.top_k = top_k
        self.agg_level = agg_level
        self.aggregation = aggregation

        self.model = NNClassifier(
            in_features=size,
            n_classes=n_classes,
            depth=None,
            dropout=dropout,
            activation="relu",
        )

    def aggregate_predictions(self, preds: torch.Tensor):
        if self.top_k != 0:
            preds = preds.topk(self.top_k, dim=0, largest=self.top_k > 0, sorted=False)[
                0
            ]  # view(1,)

        if self.aggregation == "mean":
            pred = preds.mean(dim=0, keepdim=False)
        elif self.aggregation == "max":
            pred = preds.max(dim=0, keepdim=False)
        elif self.aggregation == "min":
            pred = preds.min(dim=0, keepdim=False)
        elif isinstance(self.aggregation, float):
            pred = preds.kthvalue(
                int(preds.shape[0] * self.aggregation), dim=0, keepdim=False
            )
        else:
            raise NotImplementedError(
                f"Aggregation {self.aggregation} not implemented."
            )
        return pred

    def _forward(self, x):
        return self.model.forward(x)

    def calculate_preds(self, logits):
        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)

        return preds

    def forward(self, batch):
        # Batch
        features, target = batch

        # Prediction
        logits = self._forward(features)

        if self.agg_level == "logits":
            # Aggregation
            logits = self.aggregate_predictions(logits)

            # Loss (on logits)
            loss = self.loss.forward(logits, target.float())

            # Prediction
            preds = self.calculate_preds(logits)
        else:
            # Loss (on logits)
            loss = 0.0

            # Prediction
            preds = self.calculate_preds(logits)

            # Aggregation
            preds = self.aggregate_predictions(preds)

        return {"target": target, "preds": preds, "loss": loss}

    def training_step(self, batch, batch_idx):
        raise NotImplementedError(
            "Training is not implemented for NNClassifierMIL models."
        )

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        target, preds, loss = (
            output["target"],
            output["preds"],
            output["loss"],
        )
        self._log_metrics(preds, target, loss, "val")
        return {"val_loss": loss, "val_preds": preds, "val_target": target}

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)

        target, preds, loss = (
            output["target"],
            output["preds"],
            output["loss"],
        )

        self._log_metrics(preds, target, loss, "test")

        return {"test_loss": loss, "test_preds": preds, "test_target": target}


if __name__ == "__main__":
    # create test data and model
    batch = 32
    n_features = 384
    n_classes = 1

    features = torch.rand(batch, n_features)

    model = NNClassifier(
        in_features=n_features,
        n_classes=n_classes,
        depth=3,
        activation="relu",
        dropout=True,
    )

    # test forward
    logits = model.forward(features)
    print(logits.shape)
