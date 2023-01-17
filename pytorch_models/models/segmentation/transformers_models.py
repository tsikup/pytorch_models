import os
import numpy as np
from dotmap import DotMap
from torch import nn
from torchvision.models import resnet50
from bottleneck_transformer_pytorch import BottleStack
from ..base import BaseModel

# from submodules.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
# from submodules.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# TODO: https://github.com/The-AI-Summer/self-attention-cv
# TODO: https://theaisummer.com/transformers-computer-vision/
# TODO: https://github.com/lucidrains/vit-pytorch
# TODO: https://huggingface.co/docs/transformers/model_doc/segformer#transformers.SegformerForSemanticSegmentation


class MyTransUnet(BaseModel):
    def __init__(
        self,
        config: DotMap,
        vit_name="R50-ViT-B_16",
        n_skip=3,
        vit_patches_size=16,
        pretrained=True,
    ):
        super(MyTransUnet, self).__init__(config)
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = self.num_classes
        config_vit.n_skip = n_skip
        img_size = self.config.model.input_shape[1]
        if vit_name.find("R50") != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size),
                int(img_size / vit_patches_size),
            )
        self.model = ViT_seg(
            config_vit, img_size=img_size, num_classes=config_vit.n_classes
        )
        if pretrained:
            self.model.load_from(weights=np.load(config_vit.pretrained_path))

    def forward(self, x):
        return self.model(x)


class BotNet(BaseModel):
    """
    Srinivas et al. "Bottleneck Transformers for Visual Recognition"
    https://github.com/lucidrains/bottleneck-transformer-pytorch
    """

    def __init__(self, config: DotMap):
        super(BotNet, self).__init__(config)
        # TODO: Fix args
        layer = BottleStack(
            dim=256,
            fmap_size=56,
            dim_out=2048,
            proj_factor=4,
            downsample=True,
            heads=4,
            dim_head=128,
            rel_pos_emb=True,
            activation=nn.ReLU(),
        )

        resnet = resnet50()
        # model surgery
        backbone = list(resnet.children())

        self.model = nn.Sequential(
            *backbone[:5],
            layer,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(2048, 1000)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    import json
    from dotmap import DotMap
    from definitions import ROOT_DIR

    with open(
        os.path.join(ROOT_DIR, "assets/configs/training_seg_config.json"), "r"
    ) as f:
        config = DotMap(json.load(f))

    a = torch.rand(2, 3, 512, 512)
    model = MyTransUnet(config, pretrained=False)
    y = model.forward(a)  # [2, 1, 512, 512]
    print(y.shape)
