"""
https://huggingface.co/MahmoodLab/UNI2-h
"""

import torch
import timm


class UniModel(torch.nn.Module):
    def __init__(self, v2=True):
        super(UniModel, self).__init__()
        assert v2, "Only Uni-v2 is supported"

        timm_kwargs = {
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

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 1536) shape
        return features


def uni(v2=True):
    return UniModel(v2=v2)
