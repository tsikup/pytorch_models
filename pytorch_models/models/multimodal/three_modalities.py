import torch
import torch.nn as nn

from pytorch_models.utils.weight_init import init_max_weights


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

    def forward(self, vec1, vec2, vec3):
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
        # Path + RNA
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
        # DNA + Path
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
        # RNA + Path
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

    def forward(self, vec1, vec2, vec3):
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

    def forward(self, vec1, vec2, vec3):
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
