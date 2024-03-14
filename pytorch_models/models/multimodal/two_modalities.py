import torch
from pytorch_models.utils.tensor import kronecker_product_einsum_batched
from pytorch_models.utils.weight_init import init_max_weights
from torch import nn


class IntegrateTwoModalities(nn.Module):
    def __init__(self, dim1, dim2, odim, method="concat", dropout=0.5):
        super(IntegrateTwoModalities, self).__init__()
        assert method in [
            "concat",
            "kron",
            "bilinear",
            "bilinear_pathomic",
            "cmmmf",
        ], f"Method {method} not recognized."
        self.method = method

        if method == "bilinear_pathomic":
            self.bilinear = BilinearFusion(
                dim1=dim1, dim2=dim2, mmhid=odim, dropout_rate=dropout
            )
        elif method == "bilinear":
            self.bilinear = nn.Bilinear(dim1, dim2, odim, bias=True)
        elif method == "cmmmf":
            self.cmmmf = CMMMF(dim1, dim2, odim)

    def forward(self, imaging, clinical):
        clinical = clinical.view(*imaging.shape[:-1], -1)
        if self.method == "concat":
            return torch.cat((imaging, clinical), dim=-1)
        elif self.method == "kron":
            assert imaging.dim() in [2, 3] and clinical.dim() in [
                2,
                3,
            ], "Kronecker product only works for 2D or 3D tensors"
            return kronecker_product_einsum_batched(imaging, clinical)
        elif self.method.startswith("bilinear"):
            return self.bilinear(imaging, clinical)
        elif self.method == "cmmmf":
            return self.cmmmf(imaging, clinical)


class CMMMF(nn.Module):
    """
    From CS-MIL paper
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        odim: int,
    ):
        super(CMMMF, self).__init__()

        self.dense1 = nn.Sequential(nn.Linear(dim1, odim), nn.ReLU(), nn.Dropout(p=0.5))
        self.dense2 = nn.Sequential(nn.Linear(dim2, odim), nn.ReLU(), nn.Dropout(p=0.5))

        self.multimodal_attention = nn.Sequential(
            nn.Conv2d(odim, odim // 2, 1),  # V
            nn.ReLU(),
            nn.Conv2d(odim // 2, 1, 1),
        )

    def forward(
        self,
        vec1,
        vec2,
    ):
        if vec1.dim() == 1:
            vec1 = vec1.view(1, -1)
        if vec2.dim() == 1:
            vec2 = vec2.view(1, -1)
        bsize = vec1.shape[0]
        vec1 = self.dense1(vec1)
        vec2 = self.dense2(vec2)
        feats = torch.stack([vec1, vec2], dim=-1).view(bsize, -1, 2, 1)
        ma = self.multimodal_attention(feats)
        h = torch.matmul(feats.squeeze(-1), ma.squeeze(1)).squeeze(-1)
        h = h.view(bsize, -1)
        return h


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

    def forward(self, vec1, vec2):
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
