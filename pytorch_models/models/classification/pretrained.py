import os
import timm
import torch
from torch import nn
from typing import List
from dotmap import DotMap
from torchvision import models
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from .classifiers import NNClassifier
from ..base import BaseModel


MODELS = {
    "inception": models.inception_v3,
    "inceptionv3": models.inception_v3,
    "resnet": models.resnet50,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50": models.resnext50_32x4d,
    "resnext101": models.resnext101_32x8d,
    "resnext101_32": models.resnext101_32x8d,
    "resnext101_64": models.resnext101_64x4d,
    "dense121": models.densenet121,
    "dense161": models.densenet161,
    "dense169": models.densenet169,
    "dense201": models.densenet201,
}

WEIGHTS = {
    "inception": models.Inception_V3_Weights.IMAGENET1K_V1,
    "inceptionv3": models.Inception_V3_Weights.IMAGENET1K_V1,
    "resnet": models.ResNet50_Weights.IMAGENET1K_V2,
    "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
    "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
    "resnext50": models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
    "resnext101": models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
    "resnext101_32": models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
    "resnext101_64": models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1,
    "dense121": models.DenseNet121_Weights.IMAGENET1K_V1,
    "dense161": models.DenseNet161_Weights.IMAGENET1K_V1,
    "dense169": models.DenseNet169_Weights.IMAGENET1K_V1,
    "dense201": models.DenseNet201_Weights.IMAGENET1K_V1,
}

CLASSIFIERS = {"tsik": NNClassifier}


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


class PretrainedModel(BaseModel):
    def __init__(
        self,
        config: DotMap,
        model_name: str,
        classifier_name: str,
        n_classes: int = 1,
        in_channels: int = 3,
        classifier_depth: int = 4,
        classifier_activation: str = "relu",
        classifier_dropout: bool = True,
        freeze: bool = False,
        pretrained: bool = True,
    ):
        super(PretrainedModel, self).__init__(
            config=config,
            n_classes=n_classes,
            in_channels=in_channels,
            segmentation=False,
        )

        self.freeze = freeze

        assert (
            model_name in MODELS.keys() and model_name in WEIGHTS.keys()
        ), "Pretrained model name not known..."

        assert classifier_name in CLASSIFIERS.keys(), "Classifier name not known..."

        pretrained_model = MODELS[model_name](
            weights=WEIGHTS[model_name] if pretrained else None
        )
        train_nodes, eval_nodes = get_graph_node_names(pretrained_model)
        return_nodes = train_nodes[train_nodes.index("flatten")]
        self.output_layer_name = return_nodes
        self.feature_extractor = create_feature_extractor(
            pretrained_model, return_nodes=[return_nodes]
        )

        if self.freeze:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            freeze_params(self.feature_extractor)

        num_features = self._get_conv_output(self.dim)
        self.classifier = CLASSIFIERS[classifier_name](
            in_features=num_features,
            n_classes=self.n_classes,
            depth=classifier_depth,
            activation=classifier_activation,
            dropout=classifier_dropout,
        )

    def _get_conv_output(self, shape):
        """
        Returns the size of the output tensor going into the Linear layer from the conv block.
        Args:
            shape: shape of the input tensor, e.g. [3, 512, 512]
        """
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        features = self.feature_extractor(x)[self.output_layer_name]
        features = features
        return features

    def forward(self, x):
        y = self._forward_features(x)
        y = self.classifier(y)
        return y
