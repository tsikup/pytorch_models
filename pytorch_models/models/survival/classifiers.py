from pytorch_models.models.base import BaseSurvModel
from pytorch_models.models.classification.classifiers import NNClassifier


class NNClassifierSurvPL(BaseSurvModel):
    def __init__(self, config, size, n_classes, dropout=True):
        if n_classes == 2:
            n_classes = 1
        assert n_classes == 1, "Survival model should have 1 output class (i.e. hazard)"
        super(NNClassifierSurvPL, self).__init__(
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
