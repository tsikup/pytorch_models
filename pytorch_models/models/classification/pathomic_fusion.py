# https://github.com/mahmoodlab/PathomicFusion
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from pytorch_models.models.base import BaseMILModel
from pytorch_models.models.multimodal.two_modalities import BilinearFusion
from pytorch_models.models.multimodal.three_modalities import (
    TrilinearFusion_A,
    TrilinearFusion_B,
    TrilinearFusion_C,
)
from pytorch_models.utils.tensor import aggregate_features
from pytorch_models.utils.weight_init import init_max_weights


##############
# Omic Model #
##############
class MaxNet(nn.Module):
    def __init__(self, size: List[int], dropout_rate=0.25, init_max=True):
        super(MaxNet, self).__init__()
        encoder = []
        for i in range(len(size) - 1):
            encoder.append(
                nn.Sequential(
                    nn.Linear(size[i], size[i + 1]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout_rate, inplace=False),
                )
            )
        self.encoder = nn.Sequential(*encoder)
        # self.classifier = nn.Sequential(nn.Linear(omic_dim, n_classes))
        if init_max:
            init_max_weights(self)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


######################
# Imaging MIL Models #
######################
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


class ADMIL_NoClassifier(nn.Module):
    def __init__(
        self,
        gate=True,
        size=None,
        dropout=True,
        n_classes=2,
        linear_feature=False,
        multires=False,
    ):
        super(ADMIL_NoClassifier, self).__init__()
        assert isinstance(size, list), "Please give the size array as a list"
        assert len(size) == 4, "Please give the size array as a list of length 4"

        self.n_classes = n_classes
        self.multires = multires
        self.linear_feature = linear_feature

        if linear_feature:
            if self.multires:
                _size = int(size[0] / 2)
                self.linear_context = nn.Linear(_size, _size)
                self.linear_context = nn.Sequential(self.linear_context, nn.ReLU())
            else:
                _size = size[0]
            self.linear_target = nn.Linear(_size, _size)
            self.linear_target = nn.Sequential(self.linear_target, nn.ReLU())

        self.attention_net = self._create_attention_model(
            size, dropout, gate, n_classes=1
        )

    @staticmethod
    def _create_attention_model(size, dropout, gate, n_classes):
        fc = []
        for i in range(2):
            fc.append(nn.Linear(size[i], size[i + 1]))
            fc.append(nn.ReLU())
            if dropout:
                fc.append(nn.Dropout(0.25))

        if gate:
            attention_net = Attn_Net_Gated(
                L=size[2], D=size[3], dropout=dropout, n_classes=n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[2], D=size[3], dropout=dropout, n_classes=n_classes
            )
        fc.append(attention_net)
        return nn.Sequential(*fc)

    def forward(
        self,
        h: torch.Tensor,
        h_context: torch.Tensor = None,
    ):
        if self.multires:
            assert h_context is not None

        if self.linear_feature:
            h = self.linear_target(h)
            if self.multires:
                h_context = self.linear_context(h_context)

        if self.multires:
            h = aggregate_features([h, h_context], method="concat")
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        return M, A


class PathDnaRna_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        multires=False,
    ):
        if n_classes == 2:
            n_classes = 1
        super().__init__(config=config, n_classes=n_classes)
        self.multires = multires

        # Imaging model
        if config.model.classifier.startswith("clam"):
            self.model = ADMIL_NoClassifier(
                gate=config.model.clam.gated,
                size=config.model.clam.size,
                dropout=config.model.clam.dropout,
                n_classes=self.n_classes,
                linear_feature=config.model.clam.linear_feature,
                multires=self.multires,
            )

        self.dna_model = MaxNet(
            size=config.model.dna.size,
            dropout_rate=config.model.dna.dropout,
            init_max=True,
        )

        self.rna_model = MaxNet(
            size=config.model.rna.size,
            dropout_rate=config.model.rna.dropout,
            init_max=True,
        )

        # Fusion model
        if config.model.classifier.endswith("pathomic_tri_a"):
            self.fusion_model = TrilinearFusion_A(**config.model.pathomic)
        elif config.model.classifier.endswith("pathomic_tri_b"):
            self.fusion_model = TrilinearFusion_B(**config.model.pathomic)
        elif config.model.classifier.endswith("pathomic_tri_c"):
            self.fusion_model = TrilinearFusion_C(**config.model.pathomic)
        elif config.model.classifier.endswith("pathomic_bi"):
            self.fusion_model = BilinearFusion(**config.model.pathomic)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.model.pathomic.mmhid, self.n_classes),
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features_keys = (
            ["features"] if not self.multires else ["features", "features_context"]
        )
        features, dna, rna, clinical, target = (
            dict((k, batch["features"][k]) for k in features_keys),
            batch["features"]["dna"].float(),
            batch["features"]["rna"].float(),
            batch["features"]["clinical"].float(),
            batch["labels"],
        )

        """
        CLAM
        """
        # Prediction
        M, A = self._forward_imaging(features)

        """
        DNA
        """
        dna = self.dna_model(dna)

        """
        RNA
        """
        rna = self.rna_model(rna)

        """
        Pathomic Fusion
        """
        # vec1: path, vec2: dna, vec3: rna
        feats = self.fusion_model({"imaging": M, "dna": dna, "rna": rna})

        """
        Classifier
        """
        logits = self.classifier(feats).float()

        target = target.float()
        if self.n_classes == 1:
            logits = logits.squeeze(dim=1)

        loss = None
        if not is_predict:
            # Loss (on logits)
            loss = self.loss.forward(logits, target.squeeze(dim=1))

        if self.n_classes == 1:
            preds = torch.sigmoid(logits)
        elif self.n_classes > 2:
            preds = torch.softmax(logits, dim=1)
        else:
            raise ValueError(f"Invalid number of classes, `{self.n_classes}`")

        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward_imaging(
        self,
        features: Dict[str, torch.Tensor],
    ):
        h = features["features"] if "features" in features else None
        h_context = features["features_context"] if self.multires else None

        if h is not None and len(h.shape) == 3:
            h = h.squeeze(dim=0)
        if h_context is not None and len(h_context.shape) == 3:
            h_context = h_context.squeeze(dim=0)

        return self.model(h, h_context)
