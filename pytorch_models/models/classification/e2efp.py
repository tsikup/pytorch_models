#!/usr/bin/env python
# coding=utf-8
"""
Model architecture
Author: Lei Cao
"""
import glob
import os
import random
import warnings
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features
from torch.jit.annotations import Dict
from torch.utils.data import Dataset
from torchvision.models import resnet34, resnet50


class IntermediateLayerGetter(nn.ModuleDict):
    __annotations__ = {"return_layers": Dict[str, str]}

    def __init__(self, model, return_layers):
        ori_return_layers = return_layers.copy()
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = ori_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class MaxPool(nn.Module):
    def __init__(self, n_classes=1, k=50, dropout=False, pretrained=True):
        super(MaxPool, self).__init__()
        self.k = k
        self.n_classes = n_classes

        # ResNet34 Backbone
        m1 = resnet34(pretrained=pretrained)
        m_list = []
        for m in m1.children():
            if isinstance(m, nn.AdaptiveAvgPool2d):
                break
            m_list.append(m)
        self.feature_extractor = nn.Sequential(*m_list)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, self.n_classes)

    def forward(self, x):
        x = x.view(-1, 3, x.shape[-2], x.shape[-3])
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = x.view(-1, self.k, 512)
        x = torch.amax(x, dim=1)
        out = self.classifier(x)
        return out


class AttFPNMIL(nn.Module):
    def __init__(self, n_classes=1, k=50, pretrained=True):
        super(AttFPNMIL, self).__init__()
        self.k = k
        self.n_classes = n_classes

        # ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2", "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(
            self.feature_extractor, self.return_layers
        )
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attbranch = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 4), nn.Softmax(dim=-1)
        )
        self.classifier = nn.Linear(512, self.n_classes)

    def forward(self, x):
        x = x.view(-1, 3, x.shape[-2], x.shape[-3])
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        scores = self.attbranch(feat4)
        merged_feat = (
            scores[:, 0].unsqueeze(1) * feat1
            + scores[:, 1].unsqueeze(1) * feat2
            + scores[:, 2].unsqueeze(1) * feat3
            + scores[:, 3].unsqueeze(1) * feat4
        )
        merged_feat = merged_feat.view(-1, self.k, 512)
        x = torch.amax(merged_feat, dim=1)
        out = self.classifier(x)
        return out


class Att2FPNMIL(nn.Module):
    def __init__(self, n_classes=1, k=50, pretrained=True):
        super(Att2FPNMIL, self).__init__()
        self.k = k
        self.n_classes = n_classes

        # ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2", "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(
            self.feature_extractor, self.return_layers
        )
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attbranch = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 4), nn.Softmax(dim=-1)
        )
        self.attbranch2 = nn.Sequential(nn.Linear(512, 128), nn.Tanh())
        self.attbranch3 = nn.Sequential(nn.Linear(512, 128), nn.Sigmoid())
        self.classifier = nn.Linear(128, self.n_classes)

    def forward(self, x):
        x = x.view(-1, 3, x.shape[-2], x.shape[-3])
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        scores = self.attbranch(feat4)
        merged_feat = (
            scores[:, 0].unsqueeze(1) * feat1
            + scores[:, 1].unsqueeze(1) * feat2
            + scores[:, 2].unsqueeze(1) * feat3
            + scores[:, 3].unsqueeze(1) * feat4
        )
        merged_feat = merged_feat.view(-1, self.k, 512)
        x = self.attbranch2(merged_feat) * self.attbranch3(merged_feat)
        out = self.classifier(x)
        return out


class FPNMIL(nn.Module):
    def __init__(self, n_classes=1, k=50, pretrained=True):
        super(FPNMIL, self).__init__()
        self.k = k
        self.n_classes = n_classes

        # ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2", "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(
            self.feature_extractor, self.return_layers
        )
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, self.n_classes)

    def forward(self, x):
        x = x.view(-1, 3, x.shape[-2], x.shape[-3])
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        merged_feat = merged_feat.view(-1, self.k, 512)
        # _, top_index = merged_feat.max(dim=1)
        x, _ = torch.max(merged_feat, dim=1)
        out = self.classifier(x)
        return out  # , top_index


class FPNMIL50(nn.Module):
    def __init__(self, n_classes=1, k=50, pretrained=True):
        super(FPNMIL50, self).__init__()
        self.k = k
        self.n_classes = n_classes

        # ResNet50 Backbone
        model = resnet50(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2", "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(
            self.feature_extractor, self.return_layers
        )
        self.conv64_512 = nn.Conv2d(256, 2048, kernel_size=1)
        self.conv128_512 = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv256_512 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, self.n_classes)

    def forward(self, x):
        x = x.view(-1, 3, x.shape[-2], x.shape[-3])
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        merged_feat = merged_feat.view(-1, self.k, 2048)
        x, _ = torch.max(merged_feat, dim=1)
        out = self.classifier(x)
        return out


class FPNMIL_Mean(nn.Module):
    def __init__(self, n_classes=1, k=50, pretrained=True):
        super(FPNMIL_Mean, self).__init__()
        self.k = k
        self.n_class = n_classes

        # ResNet34 Backbone
        model = resnet34(pretrained=pretrained)
        self.return_layers = {"4": "feat1", "5": "feat2", "6": "feat3", "7": "feat4"}
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.layergetter = IntermediateLayerGetter(
            self.feature_extractor, self.return_layers
        )
        self.conv64_512 = nn.Conv2d(64, 512, kernel_size=1)
        self.conv128_512 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv256_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, self.n_class)

    def forward(self, x):
        x = x.view(-1, 3, x.shape[-2], x.shape[-3])
        inter_feats = self.layergetter(x)
        feat1 = self.gap(F.relu(self.conv64_512(inter_feats["feat1"])))
        feat2 = self.gap(F.relu(self.conv128_512(inter_feats["feat2"])))
        feat3 = self.gap(F.relu(self.conv256_512(inter_feats["feat3"])))
        feat4 = self.gap(inter_feats["feat4"])

        feat1 = feat1.view(feat1.shape[0], feat1.shape[1])
        feat2 = feat2.view(feat2.shape[0], feat2.shape[1])
        feat3 = feat3.view(feat3.shape[0], feat3.shape[1])
        feat4 = feat4.view(feat4.shape[0], feat4.shape[1])
        merged_feat = feat1 + feat2 + feat3 + feat4
        merged_feat = merged_feat.view(-1, self.k, 512)
        # x = torch.amax(merged_feat, dim=1)
        x = torch.mean(merged_feat, dim=1)
        out = self.classifier(x)
        return out


class FPNMIL_PL(BaseMILModel):
    def __init__(
        self,
        config,
        n_classes,
        multires_aggregation=None,
    ):
        self.multires_aggregation = multires_aggregation
        super(FPNMIL_PL, self).__init__(config, n_classes=n_classes)

        if config.model.e2efp.classifier == "fpnmil":
            self.model = FPNMIL(n_classes=n_classes, k=config.model.e2efp.k)
        elif config.model.e2efp.classifier == "fpnmil_mean":
            self.model = FPNMIL_Mean(n_classes=n_classes, k=config.model.e2efp.k)
        elif config.model.e2efp.classifier == "fpnmil_att":
            self.model = AttFPNMIL(n_classes=n_classes, k=config.model.e2efp.k)
        elif config.model.e2efp.classifier == "fpnmil_att2":
            self.model = Att2FPNMIL(n_classes=n_classes, k=config.model.e2efp.k)

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch["images"], batch["labels"]

        # Prediction
        results_dict = self._forward(data=features)
        logits = results_dict["logits"]
        preds = results_dict["Y_prob"]

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss = self.loss.forward(logits, target.squeeze(dim=1))

        if self.n_classes in [1, 2]:
            preds = preds[:, 1]
            preds = torch.unsqueeze(preds, dim=1)

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, data: Dict[str, torch.Tensor]):
        if self.multires_aggregation is None:
            h = data["images_target"]
        else:
            raise NotImplementedError
        logits = self.model(h)
        if self.n_classes in [1, 2]:
            preds = torch.sigmoid(logits)
        else:
            preds = torch.softmax(logits, dim=-1)
        return {"logits": logits, "Y_prob": preds}


class FPNMILDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        data_dir: str,
        data_cols: Dict[str, str],
        k: int = 50,
        base_label: int = 0,
        transform=None,
    ):
        """
        :param data_dir: hdf5 folder
        :param data_cols: hdf5 dataset name, e.g.
                {
                    "images": "images_target",
                    "features_context": "features_context",
                    "labels": "labels"
                }
        :param base_label: label offset
        """
        assert data_cols is not None

        self.data_dir = data_dir
        self.data_cols = data_cols
        self.base_label = base_label
        self.k = k
        self.transform = transform

        self.multiresolution = (
            len(self.data_cols) > 2
            if "labels" in self.data_cols
            else len(self.data_cols) > 1
        )

        assert (
            "images_target" in self.data_cols
            and self.data_cols["images_target"] is not None
        ), "`images_target` is required in `data_cols`"

        assert os.path.isdir(data_dir), f"{data_dir} is not a directory"

        # self.h5_dataset = None
        # self.labels = None
        self.slides = glob.glob(os.path.join(data_dir, "*.h5"))

        if len(self.slides) == 0:
            warnings.warn(f"No hdf5 files found in {data_dir}")

        # determine dataset length and shape
        self.dataset_size = len(self.slides)
        if self.dataset_size > 0:
            with h5py.File(self.slides[0], "r") as f:
                # Total number of datapoints
                self.img_shape = f[self.data_cols["images_target"]].shape[-3:]
                self.labels_shape = 1
        else:
            self.img_shape = None
            self.labels_shape = None

    @staticmethod
    def collate(batch):
        data = [item["images"] for item in batch]
        target = [item["labels"] for item in batch]
        target = torch.vstack(target)
        return [data, target]

    def read_hdf5(self, h5_path, load_ram=False):
        assert os.path.exists(h5_path), f"{h5_path} does not exist"

        # Open hdf5 file where images and labels are stored
        with h5py.File(h5_path, "r") as h5_dataset:
            images_dict = dict()

            for key in self.data_cols:
                if key != "labels":
                    images = h5_dataset[self.data_cols[key]]
                    indices = np.sort(
                        np.random.choice(
                            len(images),
                            size=min(self.k, len(images)),
                            replace=len(images) < self.k,
                        )
                    )
                    images = images[indices]
                    if self.transform is not None:
                        images = self.transform(images)
                    else:
                        images = torch.from_numpy(images)

                    images_dict[key] = images

            if "labels" in self.data_cols:
                label = h5_dataset[self.data_cols["labels"]][0] - self.base_label
                label = torch.from_numpy(np.array([label], dtype=label.dtype))
            else:
                label = -100
                label = torch.from_numpy(np.array([label], dtype=np.uint8))

        return images, label

    def __len__(self):
        return self.dataset_size

    @property
    def shape(self):
        return [self.dataset_size, self.img_shape]


class PixelReg(object):
    """Randomly replace tiles by an image with all pixel values set to
    the mean pixel value of the dataset with a probability of 0.75.
    """

    def __init__(self, mean_pixels, p=0.25):
        self.p = p
        assert mean_pixels is not None
        self.mean_pixels = mean_pixels

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        if torch.rand(1) < self.p:
            mask = np.zeros((3, h, w), np.float32)
            mask[0, ...] = self.mean_pixels[0]
            mask[1, ...] = self.mean_pixels[1]
            mask[2, ...] = self.mean_pixels[2]
            _range = np.max(mask) - np.min(mask)
            mask = (mask - np.min(mask)) / _range
            mask = torch.from_numpy(mask)
            return mask
        else:
            return img


class Cutout(object):
    """Randomly mask out one or more patches from an image"""

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (tensor): Tensor image of size (C, H, W)
        Returns:
            Image with n_holes of dimension lengthxlength cut out of it
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class MyRotationTrans:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


if __name__ == "__main__":
    model = Att2FPNMIL(pretrained=False)
    data = torch.randn(2, 50, 3, 384, 384)
    model.forward(data)
