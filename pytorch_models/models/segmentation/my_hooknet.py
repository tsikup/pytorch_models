from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseSegmentationModel
from torch import nn
from torchvision.transforms.functional import center_crop


class MyHookNet(nn.Module):
    """
    PyTorch implementation of HookNet (Includes custom hooks)
    args:
        n_classes: number of classes
        depth: depth of network: default 4
        n_convs: number of conv2d for each conv block: default 2
        n_filters: number of initial filters: default 2
        batch_norm: apply batch norm layer: default True
        hooks: list of tuples (hook_from_index, hook_to_index)
            for each branch (order: low-res, mid-res and high-res):
            default ((0, None), (1, 2), (None, 3))
    """

    def __init__(
        self,
        n_classes,
        depth=4,
        n_convs=2,
        n_filters=2,
        batch_norm=True,
        hooks=({"from": 0, "to": None}, {"from": 2, "to": 1}, {"from": None, "to": 3}),
    ):
        super().__init__()

        self.hooks_low_res = hooks[0]
        self.hooks_mid_res = hooks[1]
        self.hooks_high_res = hooks[2]

        self.low_mag_branch = Branch(
            n_classes=n_classes,
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            batch_norm=batch_norm,
            hook_from_index=self.hooks_low_res["from"],
        )

        low_channels = self.low_mag_branch.decoder._out_channels[0]
        self.mid_mag_branch = Branch(
            n_classes=n_classes,
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            batch_norm=batch_norm,
            hook_channels=low_channels,
            hook_to_index=self.hooks_mid_res["to"],
            hook_from_index=self.hooks_mid_res["from"],
        )

        mid_channels = self.mid_mag_branch.decoder._out_channels[2]
        self.high_mag_branch = Branch(
            n_classes=n_classes,
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            batch_norm=batch_norm,
            hook_channels=mid_channels,
            hook_to_index=self.hooks_high_res["to"],
        )
        self.last_conv = nn.Conv2d(
            self.high_mag_branch.decoder._out_channels[0], n_classes, 1
        )

    def forward(self, high_input, mid_input, low_input):
        low_out = self.low_mag_branch(low_input)
        mid_out = self.mid_mag_branch(mid_input, low_out)
        high_out = self.high_mag_branch(high_input, mid_out)
        return {"out": self.last_conv(high_out)}


class Branch(nn.Module):
    def __init__(
        self,
        n_classes,
        n_filters,
        depth,
        n_convs,
        batch_norm,
        hook_channels=0,
        hook_from_index=None,
        hook_to_index=None,
    ):

        super().__init__()
        self.encoder = Encoder(3, n_filters, depth, n_convs, batch_norm)
        self.mid_conv_block = ConvBlock(
            self.encoder._out_channels[depth - 1],
            n_filters * 2 * (depth + 1),
            n_convs,
            batch_norm,
        )
        self.decoder = Decoder(
            n_filters * 2 * (depth + 1),
            hook_channels,
            self.encoder._out_channels,
            n_filters,
            depth,
            n_convs,
            batch_norm,
            hook_from_index,
            hook_to_index,
        )

    def forward(self, x, hook_in=None):
        out, residuals = self.encoder(x)
        out = self.mid_conv_block(out)
        out = self.decoder(out, residuals, hook_in)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, n_filters, depth, n_convs, batch_norm):
        super().__init__()
        self._out_channels = {}
        self._encode_path = nn.ModuleDict()
        self._depth = depth
        for d in range(self._depth):
            self._out_channels[d] = n_filters + in_channels
            self._encode_path[f"convblock{d}"] = ConvBlock(
                in_channels, n_filters, n_convs, batch_norm, residual=True
            )
            self._encode_path[f"pool{d}"] = nn.MaxPool2d((2, 2))
            in_channels += n_filters
            n_filters *= 2

    def forward(self, x):
        residuals = []
        for d in range(self._depth):
            x = self._encode_path[f"convblock{d}"](x)
            residuals.append(x)
            x = self._encode_path[f"pool{d}"](x)
        return x, residuals


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hook_channels,
        encoder_channels,
        n_filters,
        depth,
        n_convs,
        batch_norm,
        hook_from_index,
        hook_to_index,
    ):
        super().__init__()

        self._depth = depth
        self._hook_channels = hook_channels
        self._hook_from_index = hook_from_index
        self._hook_to_index = hook_to_index
        self._decode_path = nn.ModuleDict()
        self._out_channels = {}
        n_filters = n_filters * 2 * depth
        for d in reversed(range(self._depth)):
            self._out_channels[d] = n_filters
            if d == self._hook_to_index:
                in_channels += self._hook_channels

            self._decode_path[f"upsample{d}"] = UpSample(in_channels, n_filters)
            self._decode_path[f"convblock{d}"] = ConvBlock(
                n_filters + encoder_channels[d], n_filters, n_convs, batch_norm
            )

            in_channels = n_filters
            n_filters = n_filters // 2

    def forward(self, x, residuals, hook_in=None):
        out = x
        for d in reversed(range(self._depth)):
            if hook_in is not None and d == self._hook_to_index:
                out = concatenator(out, hook_in)

            out = self._decode_path[f"upsample{d}"](out)
            out = concatenator(out, residuals[d])
            out = self._decode_path[f"convblock{d}"](out)

            if self._hook_from_index is not None and d == self._hook_from_index:
                return out

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, batch_norm, residual=False):
        super().__init__()
        self._residual = residual
        block = nn.ModuleList()
        for _ in range(n_convs):
            block.append(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3
                )
            )
            block.append(nn.LeakyReLU(inplace=True))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self._block = nn.Sequential(*block)

    def forward(self, x):
        out = self._block(x)
        if self._residual:
            return concatenator(out, x)
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sampler = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.up_sampler(x)
        out = self.conv(out)
        return self.activation(out)


def concatenator(x, x2):
    x2_cropped = center_crop(x2, x.shape[-1])
    conc = torch.cat([x, x2_cropped], dim=1)
    return conc


# Pytorch Lightning Wrapper
class HookNetPL(BaseSegmentationModel):
    """
    PyTorch implementation of HookNet

    Args:
        config: DotMap config
        n_classes: number of classes
        input_shape: list of shapes for two branches (should be equal)
        output_shape: list of shapes for output mask (should be equal)
        depth: depth of network: default 4
        n_convs: number of conv2d for each conv block: default 2
        n_filters: number of initial filters: default 16
        batch_norm: apply batch norm layer: default False
        hooks: list of tuples (hook_from_index, hook_to_index)
            for each branch (order: low-res, mid-res and high-res):
            default ((0, None), (1, 2), (None, 3))
    """

    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        in_channels: int = 3,
        depth: int = 4,
        n_convs: int = 2,
        n_filters: int = 2,
        batch_norm: bool = True,
        hooks=({"from": 0, "to": None}, {"from": 2, "to": 1}, {"from": None, "to": 3}),
        res_names: Dict[str, str] = None,
        # loss_weights=None,
    ):
        if res_names is None:
            res_names = {"high": "high", "mid": "mid", "low": "low"}
        # if loss_weights is None:
        #     loss_weights = {"target": 0.33, "mid": 0.33, "context": 0.0}

        super(HookNetPL, self).__init__(
            config,
            n_classes=n_classes,
            in_channels=in_channels,
        )
        if self.n_classes == 2:
            self.n_classes = 1
        self.depth = depth
        self.n_convs = n_convs
        self.n_filters = n_filters
        # self.loss_weights = loss_weights
        self.hooks = hooks
        self.res_names = res_names

        self.model = MyHookNet(
            n_classes=self.n_classes,
            depth=depth,
            n_convs=n_convs,
            n_filters=n_filters,
            batch_norm=batch_norm,
            hooks=hooks,
        )

    # @property
    # def multi_loss(self) -> bool:
    #     return self.loss_weights["context"] > 0.0 or self.loss_weights["mid"] > 0.0

    def _forward(self, x):
        return self.model(
            x[self.res_names["high"]],
            x[self.res_names["mid"]],
            x[self.res_names["low"]],
        )["out"]


if __name__ == "__main__":
    config = DotMap(
        {
            "num_classes": 3,
            "trainer": {
                "optimizer_params": {"lr": 1e-3},
                "batch_size": 1,
                "loss": ["ce"],
                "classes_loss_weights": None,
                "multi_loss_weights": None,
                "samples_per_class": None,
                "sync_dist": False,
            },
            "devices": {
                "nodes": 1,
                "gpus": 1,
            },
            "metrics": {"threshold": 0.5},
        }
    )

    _in_channels = 3
    _input_shape = (284, 284)
    _output_shape = (70, 70)
    images = {
        "high": torch.rand(4, _in_channels, *_input_shape),
        "mid": torch.rand(4, _in_channels, *_input_shape),
        "low": torch.rand(4, _in_channels, *_input_shape),
    }
    target = torch.randint(0, config.num_classes, (4, 1, *_output_shape))
    # target = (
    #     torch.nn.functional.one_hot(target, config.num_classes)
    #     .squeeze(dim=1)
    #     .permute(0, 3, 1, 2)
    # )

    model = HookNetPL(
        config=config,
        n_classes=config.num_classes,
        in_channels=_in_channels,
    )

    out = model((images, target))
    print(out["loss"])
