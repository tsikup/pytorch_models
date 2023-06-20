# https://github.com/mahmoodlab/PathomicFusion
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from pytorch_models.models.base import BaseMILModel
from pytorch_models.utils.tensor import aggregate_features


def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


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


#####################
# Multimodal Fusion #
#####################
class BilinearFusion(nn.Module):
    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        dim1=32,
        dim2=32,
        scale_dim1=1,
        scale_dim2=1,
        mmhid=64,
        dropout_rate=0.25,
    ):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = (
            dim1,
            dim2,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
        )
        skip_dim = dim1 + dim2 + 2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        init_max_weights(self)

    def forward(self, data):
        if "imaging" in data and "dna" in data:
            vec1 = data["imaging"]
            vec2 = data["dna"]
        elif "imaging" in data and "rna" in data:
            vec1 = data["imaging"]
            vec2 = data["rna"]
        elif "rna" in data and "dna" in data:
            vec1 = data["rna"]
            vec2 = data["dna"]

        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec2)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            )
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec1, vec2)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            )
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        if torch.cuda.is_available():
            o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        else:
            o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(
            start_dim=1
        )  # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out


class TrilinearFusion_A(nn.Module):
    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        gate3=1,
        dim1=32,
        dim2=32,
        dim3=32,
        scale_dim1=1,
        scale_dim2=1,
        scale_dim3=1,
        mmhid=96,
        dropout_rate=0.25,
    ):
        super(TrilinearFusion_A, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = (
            dim1,
            dim2,
            dim3,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
            dim3 // scale_dim3,
        )
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim3_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        ### DNA
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim3_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim3_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        ### RNA
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = (
            nn.Bilinear(dim1_og, dim3_og, dim3)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        )
        self.linear_o3 = nn.Sequential(
            nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        init_max_weights(self)

    def forward(self, data):
        vec1, vec2, vec3 = data["imaging"], data["dna"], data["rna"]

        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec3)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            )  # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec3)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec3), dim=1))
            )  # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec1, vec3)
                if self.use_bilinear
                else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            )  # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        if torch.cuda.is_available():
            o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        else:
            o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out


class TrilinearFusion_B(nn.Module):
    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        gate3=1,
        dim1=32,
        dim2=32,
        dim3=32,
        scale_dim1=1,
        scale_dim2=1,
        scale_dim3=1,
        mmhid=96,
        dropout_rate=0.25,
    ):
        super(TrilinearFusion_B, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = (
            dim1,
            dim2,
            dim3,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
            dim3 // scale_dim3,
        )
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim3_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        ### DNA
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim1_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim1_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        ### RNA
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = (
            nn.Bilinear(dim1_og, dim3_og, dim3)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        )
        self.linear_o3 = nn.Sequential(
            nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        init_max_weights(self)

    def forward(self, data):
        vec1, vec2, vec3 = data["imaging"], data["dna"], data["rna"]

        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec3)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            )  # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec1)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec1), dim=1))
            )  # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec1, vec3)
                if self.use_bilinear
                else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            )  # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        if torch.cuda.is_available():
            o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        else:
            o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out


class TrilinearFusion_C(nn.Module):
    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        gate3=1,
        dim1=32,
        dim2=32,
        dim3=32,
        scale_dim1=1,
        scale_dim2=1,
        scale_dim3=1,
        mmhid=96,
        dropout_rate=0.25,
    ):
        super(TrilinearFusion_C, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = (
            dim1,
            dim2,
            dim3,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
            dim3 // scale_dim3,
        )
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        ### Path + RNA
        if gate1:
            self.linear_h13 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            # bilinear: path + rna
            self.linear_z13 = (
                nn.Bilinear(dim1_og, dim3_og, dim1)
                if use_bilinear
                else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
            )
            self.linear_o13 = nn.Sequential(
                nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
            )

            ### Path + DNA
            self.linear_h12 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            # bilinear: path + DNA
            self.linear_z12 = (
                nn.Bilinear(dim1_og, dim2_og, dim1)
                if use_bilinear
                else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
            )
            self.linear_o12 = nn.Sequential(
                nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
            )

            self.linear_o1 = nn.Sequential(
                nn.Linear(2 * dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
            )
        else:
            self.linear_o1 = nn.Sequential(
                nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
            )

        ### DNA + RNA
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        # bilinear: dna + rna
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim3_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim3_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        ### RNA + DNA
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        # bilinear rna + dna
        self.linear_z3 = (
            nn.Bilinear(dim3_og, dim2_og, dim3)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim3_og + dim2_og, dim3))
        )
        self.linear_o3 = nn.Sequential(
            nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        init_max_weights(self)

    def forward(self, data):
        # vec1: path, vec2: dna, vec3: rna
        vec1, vec2, vec3 = data["imaging"], data["dna"], data["rna"]

        ### Gated Multimodal Units
        if self.gate1:
            h12 = self.linear_h12(vec1)
            z12 = (
                self.linear_z12(vec1, vec2)
                if self.use_bilinear
                else self.linear_z12(torch.cat((vec1, vec2), dim=1))
            )  # Gate Path with DNA
            o12 = self.linear_o12(nn.Sigmoid()(z12) * h12)

            h13 = self.linear_h13(vec1)
            z13 = (
                self.linear_z13(vec1, vec3)
                if self.use_bilinear
                else self.linear_z13(torch.cat((vec1, vec3), dim=1))
            )  # Gate Path with DNA
            o13 = self.linear_o13(nn.Sigmoid()(z13) * h13)

            o1 = torch.cat((o12, o13), dim=1)
            o1 = self.linear_o1(o1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec3)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec3), dim=1))
            )  # Gate DNA with Path
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec3, vec2)
                if self.use_bilinear
                else self.linear_z3(torch.cat((vec3, vec2), dim=1))
            )  # Gate RNA with DNA
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        if torch.cuda.is_available():
            o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        else:
            o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
            o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
            o3 = torch.cat((o3, torch.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out


class PathDnaRna_PL(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        multires=None,
    ):
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
        features, dna, rna, clinical, target = (
            batch["features"],
            batch["dna"],
            batch["rna"],
            batch["clinical"],
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
        logits = self.classifier(feats)

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


if __name__ == "__main__":
    config = DotMap(
        {
            "mode": "train",
            "model": {
                "backbone": "vit",
                "classifier": "clam-pathomic_tri_c",
                "initializer": None,
                "compile": False,
                "checkpoint": None,
                "clam": {
                    "size": [2048, 1024, 512, 256],
                    "attention_depth": 2,
                    "classifier_depth": 2,
                    "gated": True,
                    "dropout": True,
                    "linear_feature": True,
                    "multires": True,
                },
                "dna": {"size": [200, 128, 64, 32, 32], "dropout": 0.25},
                "rna": {"size": [200, 128, 64, 32, 32], "dropout": 0.25},
                "pathomic": {
                    "skip": 1,
                    "use_bilinear": 1,
                    "gate1": 1,
                    "gate2": 1,
                    "gate3": 1,
                    "dim1": 512,
                    "dim2": 32,
                    "dim3": 32,
                    "scale_dim1": 16,
                    "scale_dim2": 1,
                    "scale_dim3": 1,
                    "mmhid": 96,
                    "dropout_rate": 0.25,
                },
            },
            "trainer": {
                "seed": 42,
                "precision": 16,
                "epochs": 50,
                "batch_size": 1,
                "accumulate_grad_batches": 32,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "num_workers": 1,
                "shuffle": True,
                "check_val_every_n_epoch": 1,
                "reload_dataloaders_every_n_epochs": 1,
                "callbacks": True,
                "sync_dist": False,
                "optimizer": "swats",
                "lookahead": False,
                "optimizer_params": {"lr": 0.0001},
                "lr_scheduler": None,
                "lr_scheduler_params": {
                    "first_cycle_steps": 250,
                    "cycle_mult": 1,
                    "max_lr": 0.0001,
                    "min_lr": 0.00001,
                    "warmup_steps": 0,
                    "gamma": 0.9,
                },
                "class_mode": "binary",
                "loss": ["ce"],
                "multi_loss_weights": [1],
                "classes_loss_weights": None,
            },
            "metrics": {
                "mdmc_reduce_comment": "`global` or `samplewise`",
                "mdmc_reduce": "global",
                "threshold": None,
            },
            "callbacks": {
                "early_stopping": True,
                "es_patience": 10,
                "es_min_delta": 0.001,
                "checkpoint_top_k": 5,
                "stochastic_weight_averaging": False,
            },
            "dataset": {
                "mil": True,
                "precomputed": True,
                "num_tiles": -1,
                "processing_batch_size": -1,
                "train_folder": "/mimer/NOBACKUP/groups/foukakis_ai/niktsi/data/PREDIX/Multiomics/Pathomic-Fusion/data/kfold",
                "val_folder": "/mimer/NOBACKUP/groups/foukakis_ai/niktsi/data/PREDIX/Multiomics/Pathomic-Fusion/data/kfold",
                "test_folder": "/mimer/NOBACKUP/groups/foukakis_ai/niktsi/data/PREDIX/Multiomics/Pathomic-Fusion/data/test",
                "data_cols": {
                    "features_target": "embeddings_densenet121_imagenet_x_target",
                    "features_context": "embeddings_densenet121_imagenet_x_x10",
                    "dna": "dna",
                    "rna": "rna",
                    "clinical": "clinical",
                    "labels": "pCR",
                },
                "base_label": 0,
                "classes": [0, 1],
                "target_names": ["noPCR", "PCR"],
                "num_classes": 2,
            },
            "comet": {
                "enable": False,
                "api_key": "API_KEY",
                "project": "PROJECT_NAME",
                "workspace": "WORKSPACE",
                "experiment_key": None,
            },
            "telegram": {"token": None, "chat_id": None},
        }
    )
    vec1 = torch.randn(100, 1024)
    vec2 = torch.randn(1, 200)
    vec3 = torch.randn(1, 200)
    vec4 = torch.randn(1, 4)
    labels = torch.randint(0, 2, (1, 1))
    model = PathDnaRna_PL(config, n_classes=1, multires=True)

    out = model(
        {
            "features": {
                "features": vec1,
                "features_context": vec1,
            },
            "dna": vec2,
            "rna": vec3,
            "clinical": vec4,
            "labels": labels,
        }
    )
