import torch
import torch.nn as nn

from ....utils.weight_init import init_weights


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.ReLU(),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2, mode="bilinear"):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)

        if is_deconv:
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=4, stride=2, padding=1
            )
        elif mode == "bilinear":
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        elif mode == "nearest":
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        else:
            self.up = nn.Upsample(mode=mode, scale_factor=2)

        if not is_deconv:
            self.up = nn.Sequential(
                self.up,
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find("unetConv2") != -1:
                continue
            init_weights(m, init_type="kaiming")

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, *bridge):
        out = self.up(x)
        for i in range(len(bridge)):
            crop = self.center_crop(bridge[i], out.shape[2:])
            out = torch.cat([out, crop], 1)
        return self.conv(out)


class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2, mode="bilinear"):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            if mode == "bilinear":
                self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            elif mode == "nearest":
                self.up = nn.UpsamplingNearest2d(scale_factor=2)
            else:
                self.up = nn.Upsample(mode=mode, scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find("unetConv2") != -1:
                continue
            init_weights(m, init_type="kaiming")

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
