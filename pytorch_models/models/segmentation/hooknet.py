from typing import List, Dict
import torch
import torchvision.transforms.functional as Fv
from torch import nn, Tensor
from dotmap import DotMap

from ...models.base import BaseModel
from ...utils.weight_init import weights_init_kaiming


class HookNet(BaseModel):
    """
    PyTorch implementation of HookNet

    Args:
        config: DotMap config
        input_shape: list of shapes for two branches (should be equal)
        hook_indices: list or tuple of two integers for hooking indices
        depth: depth of network: default 4
        n_convs: number of conv2d for each conv block: default 2
        n_filters: number of initial filters: default 16
        kernel_size: kernel size for convolutions: default 3
        padding: padding value or mode: default 'valid'
        batch_norm: apply batch norm layer: default False
        activation: activation function to use: default 'relu'
        loss_weights: each model's loss weights for final loss computation: default [1.0, 0.0]
        merge_type: merge type for concatenations ('concat', 'add', 'subtract', 'multiply'): default 'concat'
        predict_target_only: predict target only: default True
    """

    def __init__(
        self,
        config: DotMap,
        input_shape: List[int],
        hook_indices: [tuple[int], List[int]],
        depth: int = 4,
        n_convs: int = 2,
        n_filters: int = 16,
        kernel_size: int = 3,
        padding: str = "valid",
        batch_norm: bool = False,
        activation: str = "relu",
        l2_lambda: float = 0.001,
        loss_weights: List[float] = {"target": 1.0, "context": 0.0},
        merge_type: str = "concat",
        predict_target_only: bool = True,
    ):
        super(HookNet, self).__init__(config)

        if input_shape[0] != input_shape[1]:
            raise ValueError("input shapes of both branches should be the same")

        if hook_indices[1] < 1:
            raise ValueError(
                "hook index for target should be > 0, due to the fact that context resolution (um/px) at 0 should be "
                "larger than that of target at 0. (e.g. target -> 0.5 um/px, context -> 1 um/px, both at 0 depth, "
                "for this example hook index can be [0,1] so that target res is 1 um/px and context is 1 um/px at "
                "depths 1 and 0 respectively) "
            )

        if not check_input(
            depth=depth,
            input_size=input_shape[0][1],
            filter_size=kernel_size,
            n_convs=n_convs,
        ):
            raise ValueError("input_shapes are not valid model parameters")

        activations = {"relu": nn.ReLU}

        # hyperparams
        self._input_shape_target = input_shape[0]
        self._input_shape_context = input_shape[1]

        self._hook_indices = {(depth - 1) - hook_indices[0]: hook_indices[1] - 1}
        self._loss_weights = loss_weights
        # self._merge_type = merge_type
        self._predict_target_only = predict_target_only

        self._activation = activations[activation]

        # determine multi-loss model from loss weights
        self._multi_loss = loss_weights["context"] > 0.0
        # TODO: self.loss = get_loss()

        # ******************** #
        # build CONTEXT branch #
        # ******************** #
        # encoder
        self.context_encoder = Encoder(
            self.in_channels,
            depth,
            n_convs,
            n_filters,
            kernel_size,
            padding,
            activation=self._activation,
            batch_norm=batch_norm,
            name="context",
        )
        # mid conv
        self.context_mid_conv = ConvBlock(
            in_channels=n_filters * 2 ** (depth - 1),
            n_convs=n_convs,
            n_filters=n_filters * 2 * (depth + 1),
            kernel_size=kernel_size,
            padding=padding,
            activation=self._activation,
            batch_norm=batch_norm,
        )
        # decoder
        self.context_decoder = Decoder(
            in_channels=n_filters * 2 * (depth + 1),
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            kernel_size=kernel_size,
            hook_indices=self._hook_indices,
            padding=padding,
            activation=self._activation,
            batch_norm=batch_norm,
            merge_type=merge_type,
            name="context",
        )

        # ******************* #
        # build TARGET branch #
        # ******************* #
        # encoder
        self.target_encoder = Encoder(
            self.in_channels,
            depth,
            n_convs,
            n_filters,
            kernel_size,
            padding,
            activation=self._activation,
            batch_norm=batch_norm,
            name="target",
        )
        # mid conv
        self.target_mid_conv = ConvBlock(
            in_channels=n_filters * 2 ** (depth - 1),
            n_convs=n_convs,
            n_filters=n_filters * 2 * (depth + 1),
            kernel_size=kernel_size,
            padding=padding,
            activation=self._activation,
            batch_norm=batch_norm,
        )
        # decoder
        self.target_decoder = Decoder(
            in_channels=n_filters * 2 * (depth + 1),
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            kernel_size=kernel_size,
            hook_indices=self._hook_indices,
            padding=padding,
            activation=self._activation,
            batch_norm=batch_norm,
            merge_type=merge_type,
            name="target",
        )

    def _forward(self, context, target):
        context, con_residuals = self.context_encoder.forward(context)
        context = self.context_mid_conv.forward(context)

        target, tar_residuals = self.target_encoder.forward(target)
        target = self.target_mid_conv.forward(target)

        context, hooks = self.context_decoder.forward(
            context, con_residuals, return_hooks=True
        )
        target = self.target_decoder.forward(
            target, tar_residuals, hooks, return_hooks=False
        )

        # # softmax output
        # net = Conv2D(self._n_classes, 1, activation="softmax")(net)
        #
        # # set output shape
        # self._out_shape = int_shape(net)[1:]
        #
        # # Reshape net
        # flatten = Reshape(
        #     (self._out_shape[0] * self._out_shape[1], self._out_shape[2]),
        #     name=reshape_name,
        # )(net)

        return target

    @property
    def input_shape(self) -> List[int]:
        """Return the input shape of the model"""
        return self._input_shape_target

    @property
    def output_shape(self) -> List[int]:
        """Return the output shape of the model before flattening"""
        return self._out_shape

    @property
    def multi_loss(self) -> bool:
        return self._multi_loss


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        n_convs: int,
        n_filters: int,
        kernel_size: int,
        padding: str = "valid",
        activation=nn.ReLU,
        batch_norm: bool = True,
        name: str = "target",
    ):
        super(Encoder, self).__init__()
        self.name = name
        self.layers = dict()
        self.n_filters = n_filters
        _in_channels = in_channels
        _n_filters = self.n_filters

        for d in range(depth):
            self.layers[d] = dict()

            # conv block
            self.layers[d]["conv"] = ConvBlock(
                in_channels=_in_channels,
                n_convs=n_convs,
                n_filters=_n_filters,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                batch_norm=batch_norm,
            )

            # downsample
            self.layers[d]["downsample"] = Downsample()

            # input channels for next block in the number of filters (output) of this block
            _in_channels = _n_filters

            # increase number of filters with factor 2
            _n_filters *= 2

    def forward(self, x):
        residuals = []
        for d in range(len(self.layers)):
            # apply conv block
            x = self.layers[d]["conv"](x)

            # keep tensor for skip connection
            residuals.append(x)

            # apply downsampling
            x = self.layers[d]["downsample"](x)

        # return output and skip connections
        return x, residuals


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        depth: int,
        n_convs: int,
        kernel_size: int,
        hook_indices: Dict,
        padding="valid",
        activation=nn.ReLU,
        batch_norm: bool = True,
        merge_type="concat",
        name="target",
    ):
        super(Decoder, self).__init__()
        self.n_filters = n_filters
        self.depth = depth
        self.hook_indices = hook_indices
        self.merge_type = merge_type
        # set start number of filters of decoder
        _n_filters = self.n_filters * 2 * depth
        _in_channels = in_channels

        self.layers = dict()

        # loop through depth in reverse
        for d in reversed(range(self.depth)):
            self.layers[d] = dict()

            # # hook if hook is available
            # if b in inhooks:
            # # combine feature maps via merge type
            # if self.merge_type != "concat":
            # self.layers[d]['merger'] = Merger()

            if name == "target" and d in hook_indices.values():
                # hook_indices dict has unique keys AND values
                hook_c_d = list(hook_indices.keys())[
                    list(hook_indices.values()).index(d)
                ]
                _in_channels = n_filters * 2 ** (depth - 1 - hook_c_d) + _in_channels

            # upsample
            self.layers[d]["upsample"] = Upsample(
                in_channels=_in_channels,
                n_filters=_n_filters,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
            )

            _in_channels = _n_filters

            # apply conv block
            self.layers[d]["conv"] = ConvBlock(
                in_channels=_in_channels * 2,
                n_convs=n_convs,
                n_filters=_n_filters,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                batch_norm=batch_norm,
            )

            _n_filters = _n_filters // 2

    def forward(
        self, x: Tensor, residuals: List[Tensor], inhooks: Dict = {}, return_hooks=True
    ) -> Tensor:
        # list for keeping potential hook Tensors
        outhooks = []

        # loop through depth in reverse
        for d in reversed(range(self.depth)):
            # hook if hook is available
            if d in inhooks:
                # combine feature maps via merge type
                if self.merge_type == "concat":
                    x = concatenate(x, inhooks[d])
                # # combine via merger
                # else:
                #     net = self._merger(net, inhooks[d])

            # upsample
            x = self.layers[d]["upsample"](x)

            # concatenate residuals/skip connections
            x = concatenate(x, residuals[d])

            # apply conv block
            x = self.layers[d]["conv"](x)

            if return_hooks:
                # set potential hook
                outhooks.append(x)

        if return_hooks:
            # get hooks from potential hooks
            hooks = {}
            for shook, ehook in self.hook_indices.items():
                hooks[ehook] = outhooks[shook]

            return x, hooks
        else:
            return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_convs: int,
        n_filters: int,
        kernel_size: int,
        padding="valid",
        activation=nn.ReLU,
        batch_norm=True,
    ):
        super(ConvBlock, self).__init__()
        _in_channels = in_channels
        self.layers = dict()
        self.batch_norm = batch_norm
        # loop through number of convolutions in convolution block
        for n in range(n_convs):
            self.layers[n] = dict()
            # apply 2D convolution
            conv = nn.Conv2d(
                in_channels=_in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=padding,
            )
            _in_channels = n_filters
            # nn.init.kaiming_normal(conv.weight)
            weights_init_kaiming(conv)
            self.layers[n]["conv"] = conv
            # activation function
            self.layers[n]["act"] = activation()
            # apply batch normalization
            if self.batch_norm:
                self.layers[n]["bnorm"] = nn.BatchNorm2d(num_features=n_filters)

    def forward(self, x):
        for n in range(len(self.layers)):
            x = self.layers[n]["conv"](x)
            x = self.layers[n]["act"](x)
            if self.batch_norm:
                x = self.layers[n]["bnorm"](x)
        return x


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int,
        padding="valid",
        activation=nn.ReLU,
    ):
        super(Upsample, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.act = activation()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.act(x)
        return x


def concatenate(net: Tensor, item: Tensor) -> Tensor:
    """ "Concatenate feature maps"""
    # crop feature maps
    crop_size = int(int(item.shape[2] - net.shape[2]) / 2)
    item_cropped = Fv.pad(item, -crop_size)
    return torch.cat([item_cropped, net], axis=1)


class Merger(nn.Module):
    def __init__(self, current_filters: int, activation=nn.ReLU, padding="valid"):
        super(Merger, self).__init__()
        # adapt number of filters via 1x1 convolutional to allow merging
        self.conv1 = nn.LazyConv2d(
            current_filters, 1, activation=activation, padding=padding
        )

        # Combine feature maps by adding
        if self._merge_type == "add":
            self.op = torch.add
        # Combine feature maps by subtracting
        if self._merge_type == "subtract":
            self.op = torch.subtract
        # Combine feature maps by multiplication
        if self._merge_type == "multiply":
            self.op = torch.multiply
        # Raise ValueError if merge type is unsupported
        else:
            raise ValueError(f"unsupported merge type: {self._merge_type}")

    def forward(self, net: Tensor, item: Tensor) -> Tensor:
        """ "Combine feature maps"""

        # crop feature maps
        crop_size = int(int(item.shape[2] - net.shape[2]) / 2)
        item_cropped = Fv.pad(item, -crop_size)

        # adapt number of filters via 1x1 convolutional to allow merge
        item_cropped = self.conv1(item_cropped)
        return self.op(item_cropped, net)


# This constraint assumes valid convolutions with stride of 1, 2x2 pooling and 2x2 upsampling
def check_input(depth, input_size, filter_size, n_convs):
    """checks if input is valid for model configuration

    Args:
        depth (int): depth of the model
        input_size (int): shape of model (width or height)
        filter_size (int): filter size  convolutions
        n_convs (int): number of convolutions per depth
    """

    def is_even(size):
        return size % 2 == 0

    i1 = input_size
    # encoding
    for _ in range(depth):
        # input_size reduced through valid convs
        i1 -= (filter_size - 1) * n_convs
        # check if inputsize before pooling is even
        if not is_even(i1):
            return False

        # max pooling
        i1 /= 2
        if i1 <= 0:
            return False

    # decoding
    for _ in range(depth):
        # input_size reduced through valid convs
        i1 -= (filter_size - 1) * n_convs

        # check if inputsize before upsampling is even
        if not is_even(i1):
            return False

        # upsampling
        i1 *= 2
        i1 -= filter_size - 1

        if i1 <= 0:
            return False

    # check if inputsize is even
    if not is_even(i1):
        False

    i1_end = i1 - (filter_size - 1) * n_convs

    if i1_end <= 0:
        return False

    return True
