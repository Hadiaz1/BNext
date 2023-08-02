
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x


class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.residual(x)


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn((self.shape)) * 0.001, requires_grad=True)

        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        if self.training:
            self.weight.data.clamp_(-1.5, 1.5)

        real_weights = self.weight

        if self.temperature < 1e-7:
            binary_weights_no_grad = real_weights.sign()
        else:
            binary_weights_no_grad = (real_weights / self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        cliped_weights = real_weights

        if self.training:
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            binary_weights = binary_weights_no_grad

        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y


class HardSign(nn.Module):
    def __init__(self, range=[-1, 1], progressive=False):
        super(HardSign, self).__init__()
        self.range = range
        self.progressive = progressive
        self.register_buffer("temperature", torch.ones(1))

    def adjust(self, x, scale=0.1):
        self.temperature.mul_(scale)

    def forward(self, x):
        replace = x.clamp(self.range[0], self.range[1])
        x = x.div(self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if not self.progressive:
            sign = x.sign()
        else:
            sign = x
        return (sign - replace).detach() + replace


class ConvNeXtBlock(Residual):
    def __init__(self, channels, kernel_size, mult=4, p_drop=0.):
        padding = (kernel_size - 1) // 2
        hidden_channels = channels * mult
        super().__init__(
            HardSign(range=[-1.5, 1.5]),
            HardBinaryConv(channels, channels, kernel_size=3,  padding=padding),
            LayerNormChannels(channels),
            HardSign(range=[-1.5, 1.5]),
            HardBinaryConv(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            HardSign(range=[-1.5, 1.5]),
            HardBinaryConv(hidden_channels, channels, kernel_size=1),
            nn.Dropout(p_drop)
        )

class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__(
            LayerNormChannels(in_channels),
            HardSign(range=[-1.5, 1.5]),
            HardBinaryConv(in_channels, out_channels, kernel_size=3, stride=stride),
        )

class Stage(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_blocks, kernel_size, p_drop=0.):
        layers = [] if in_channels == out_channels else [DownsampleBlock(in_channels, out_channels)]
        layers += [ConvNeXtBlock(out_channels, kernel_size, p_drop=p_drop) for _ in range(num_blocks)]
        super().__init__(*layers)

class ConvNeXtBody(nn.Sequential):
    def __init__(self, in_channels, channel_list, num_blocks_list, kernel_size, p_drop=0.):
        layers = []
        for out_channels, num_blocks in zip(channel_list, num_blocks_list):
            layers.append(Stage(in_channels, out_channels, num_blocks, kernel_size, p_drop))
            in_channels = out_channels
        super().__init__(*layers)


class Stem(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size),
            LayerNormChannels(out_channels)
        )

class Head(nn.Sequential):
    def __init__(self, in_channels, classes):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, classes)
        )


class ConvNeXt(nn.Sequential):
    def __init__(self, classes, channel_list, num_blocks_list, kernel_size, patch_size,
                 in_channels=3, res_p_drop=0.):
        super().__init__(
            Stem(in_channels, channel_list[0], patch_size),
            ConvNeXtBody(channel_list[0], channel_list, num_blocks_list, kernel_size, res_p_drop),
            Head(channel_list[-1], classes)
        )

