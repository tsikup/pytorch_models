from typing import Union

from torch import nn
from ..base import BaseModel

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
        dropout=True,
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
            features[depth + 1]["out"] = n_classes
        else:
            assert depth is not None
            for d in range(1, depth + 1):
                features[d] = dict()
                features[d]["in"] = in_features
                features[d]["out"] = in_features // 2
                in_features = in_features // 2
            features[depth + 1] = dict()
            features[depth + 1]["in"] = in_features
            features[depth + 1]["out"] = n_classes

        print(features)
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
                self.add_module(name=f"dropout_{d-1}", module=nn.Dropout(p=0.5))

        self.add_module(
            name="linear_out", module=nn.Linear(features[depth + 1]["in"], n_classes)
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
