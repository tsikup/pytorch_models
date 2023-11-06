import segmentation_models_pytorch as smp
import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseSegmentationModel
from pytorch_models.models.segmentation.unet.layers import unetConv2, unetUp
from pytorch_models.utils.weight_init import init_weights
from torch import nn


class UNet(nn.Module):
    """
    Original UNet

    Ronneberger, O., Fischer, P. and Brox, T., 2015, October. U-net: Convolutional networks for biomedical image
    segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp.
    234-241). Springer, Cham.
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        num_filters=(64, 128, 256, 512, 1024),
        is_deconv=True,
        is_batchnorm=True,
        apply_last_layer=True,
    ):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.apply_last_layer = apply_last_layer
        self.in_channels = in_channels
        self.n_classes = n_classes
        filters = num_filters
        depth = len(filters)

        # Encoder
        self.encoder = []
        for d in range(depth - 1):
            self.encoder.append(
                dict(
                    conv=unetConv2(in_channels, filters[d], is_batchnorm),
                    maxpool=nn.MaxPool2d(kernel_size=2),
                )
            )
            in_channels = filters[d]

        # Center
        self.center = unetConv2(filters[depth - 2], filters[depth - 1], is_batchnorm)

        # Decoder
        self.decoder = []
        for d in reversed(range(1, depth)):
            self.decoder.append(
                unetUp(
                    filters[d],
                    filters[d - 1],
                    is_deconv,
                    n_concat=2,
                )
            )

        if self.apply_last_layer:
            self.outconv1 = nn.Conv2d(filters[0], n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming", init_bias=True)
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming", init_bias=True)

    def dotProduct(self, seg, cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs):
        i = inputs
        convs = []
        for d in range(len(self.encoder)):
            i = self.encoder[d]["conv"](i)
            convs.append(i)
            i = self.encoder[d]["maxpool"](i)

        out = self.center(i)  # 256*32*64

        for d in range(len(self.decoder)):
            out = self.decoder[d](out, convs[-(d + 1)])

        out = self.outconv1(out)  # 256
        return out


class UNet_PL(BaseSegmentationModel):
    def __init__(
        self,
        config,
        in_channels=3,
        n_classes=1,
        num_filters=(64, 128, 256, 512, 1024),
        is_deconv=True,
        is_batchnorm=True,
        apply_last_layer=True,
    ):
        super().__init__(config, in_channels=in_channels, n_classes=n_classes)
        self.model = UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            num_filters=num_filters,
            is_deconv=is_deconv,
            is_batchnorm=is_batchnorm,
            apply_last_layer=apply_last_layer,
        )

    def _forward(self, img_batch):
        return self.model(img_batch)


class ResNetUnet(BaseSegmentationModel):
    def __init__(
        self, config: DotMap, in_channels: int, n_classes: int, pretrained=True
    ):
        super(ResNetUnet, self).__init__(
            config, in_channels=in_channels, n_classes=n_classes
        )
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=self.in_channels,
            classes=self.n_classes,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    from dotmap import DotMap

    config = DotMap(
        {
            "num_classes": 1,
            "model": {"input_shape": 256},
            "trainer": {
                "optimizer_params": {"lr": 1e-3},
                "batch_size": 1,
                "loss": ["ce"],
                "classes_loss_weights": None,
                "multi_loss_weights": None,
                "samples_per_class": None,
                "sync_dist": False,
                "l1_reg_weight": None,
                "l2_reg_weight": None,
            },
            "devices": {
                "nodes": 1,
                "gpus": 1,
            },
            "metrics": {"threshold": 0.5},
        }
    )

    a = torch.rand(2, 3, 224, 224)
    b = torch.rand(2, 1, 224, 224)
    model = UNet_PL(config, num_filters=(64, 128, 256, 512, 1024), is_deconv=True)
    y = model.forward((a, b))  # [2, 1, 224, 224]
    print(y["logits"].shape)
