"""
https://huggingface.co/MahmoodLab/UNI2-h
"""

import os
import torch
import timm


class UniModel(torch.nn.Module):
    def __init__(self, ckpt_dir, v2=True):
        super(UniModel, self).__init__()
        assert v2, "Only Uni-v2 is supported"

        timm_kwargs = {
            "model_name": "vit_giant_patch14_224",
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        self.model = timm.create_model(pretrained=False, **timm_kwargs)
        self.model.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "uni_v2_mahmood.bin"), map_location="gpu"
            ),
            strict=True,
        )

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 1536) shape
        return features


def uni(v2=True):
    return UniModel(v2=v2)
