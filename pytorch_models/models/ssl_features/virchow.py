"""
https://huggingface.co/paige-ai/Virchow2
"""

import torch
import timm
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class VirchowModel(torch.nn.Module):
    def __init__(self):
        super(VirchowModel, self).__init__()
        # need to specify MLP layer and activation function for proper init
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    def get_transforms(self):
        transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )
        return transforms

    def forward(self, x, with_class_token=True):
        output = self.model(x)  # size: 1 x 261 x 1280

        class_token = output[:, 0]  # size: 1 x 1280
        patch_tokens = output[
            :, 5:
        ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        if with_class_token:
            # concatenate class token and average pool of patch tokens
            embedding = torch.cat(
                [class_token, patch_tokens.mean(1)], dim=-1
            )  # size: 1 x 2560
        else:
            embedding = patch_tokens.mean(1)

        return embedding


def virchow():
    return VirchowModel()
