import torch
import math
import torch.nn as nn
import torch.nn.functional as f
from basicsr.utils.registry import ARCH_REGISTRY





class NTIREMeanShift(nn.Module):
    r"""MeanShift for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, rgb_range: int, sign: int = -1, data_type: str = 'DIV2K') -> None:
        super(NTIREMeanShift, self).__init__()

        self.sign = sign

        self.rgb_range = rgb_range
        self.rgb_std = (1.0, 1.0, 1.0)
        if data_type == 'DIV2K':
            # RGB mean for DIV2K 1-800
            self.rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == 'DF2K':
            # RGB mean for DF2K 1-3450
            self.rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f'Unknown data type for MeanShift: {data_type}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.Tensor(self.rgb_std)
        weight = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        bias = self.sign * self.rgb_range * torch.Tensor(self.rgb_mean) / std
        return f.conv2d(input=x, weight=weight.type_as(x), bias=bias.type_as(x))

class NTIREShiftConv2d1x1(nn.Conv2d):
    r"""ShiftConv2d1x1 for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), bias: bool = True, shift_mode: str = '+', val: float = 1.,
                 **kwargs) -> None:
        super(NTIREShiftConv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                                  dilation=dilation, groups=1, bias=bias, **kwargs)

        assert in_channels % 5 == 0, f'{in_channels} % 5 != 0.'
        self.in_channels = in_channels
        self.channel_per_group = in_channels // 5
        self.shift_mode = shift_mode
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cgp = self.channel_per_group
        mask = torch.zeros(self.in_channels, 1, 3, 3)
        if self.shift_mode == '+':
            mask[0 * cgp:1 * cgp, 0, 1, 2] = self.val
            mask[1 * cgp:2 * cgp, 0, 1, 0] = self.val
            mask[2 * cgp:3 * cgp, 0, 2, 1] = self.val
            mask[3 * cgp:4 * cgp, 0, 0, 1] = self.val
            mask[4 * cgp:, 0, 1, 1] = self.val
        elif self.shift_mode == 'x':
            mask[0 * cgp:1 * cgp, 0, 0, 0] = self.val
            mask[1 * cgp:2 * cgp, 0, 0, 2] = self.val
            mask[2 * cgp:3 * cgp, 0, 2, 0] = self.val
            mask[3 * cgp:4 * cgp, 0, 2, 2] = self.val
            mask[4 * cgp:, 0, 1, 1] = self.val
        else:
            raise NotImplementedError(f'Unknown shift mode for ShiftConv2d1x1: {self.shift_mode}.')

        x = f.conv2d(input=x, weight=mask.type_as(x), bias=None,
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=self.in_channels)
        x = f.conv2d(input=x, weight=self.weight, bias=self.bias,
                     stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        return x

class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)

class LayerNorm4D(nn.Module):
    r"""LayerNorm for 4D input.

    Modified from https://github.com/sail-sg/poolformer.

    Args:
        num_channels (int): Number of channels expected in input
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5

    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class _Swish(torch.autograd.Function):  # noqa
    @staticmethod
    def forward(ctx, i):  # noqa
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish(nn.Module):
    r"""A memory-efficient implementation of Swish. The original code is from
        https://github.com/zudi-lin/rcan-it/blob/main/ptsr/model/_utils.py.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        return _Swish.apply(x)

class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(self, upscale: int, in_channels: int,
                 out_channels: int, upsample_mode: str = 'csr') -> None:

        layer_list = list()
        if upsample_mode == 'csr':  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'Upscale {upscale} is not supported.')
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == 'lsr':  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale ** 2)))
            layer_list.append(nn.PixelShuffle(upscale))
        elif upsample_mode == 'denoising' or upsample_mode == 'deblurring' or upsample_mode == 'deraining':
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        else:
            raise ValueError(f'Upscale mode {upscale} is not supported.')

        super(Upsampler, self).__init__(*layer_list)

class MLP4D(nn.Module):
    r"""Multi-layer perceptron for 4D input.

    Args:
        in_features: Number of input channels
        hidden_features:
        out_features: Number of output channels
        act_layer:

    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer: nn.Module = nn.GELU) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = NTIREShiftConv2d1x1(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = NTIREShiftConv2d1x1(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SelfGate(nn.Module):
    def __init__(self, planes: int, kernel_size: int) -> None:
        super().__init__()

        self.h_mixer = nn.Conv2d(planes, planes, kernel_size=(kernel_size, 1),
                                 padding=(kernel_size // 2, 0), groups=planes)
        self.w_mixer = nn.Conv2d(planes, planes, kernel_size=(1, kernel_size),
                                 padding=(0, kernel_size // 2), groups=planes)
        self.c_mixer = Conv2d1x1(planes, planes)

        self.proj = Conv2d1x1(planes, planes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """

        # 1
        # _h = x * self.h_mixer(x)
        # _hw = _h * self.w_mixer(_h)
        # _chw = _hw * self.c_mixer(_hw)

        # # 2
        _hw = self.h_mixer(x) * self.w_mixer(x)
        _hw=x+_hw
        _chw = _hw * self.c_mixer(_hw)
        _chw=_hw+_chw

        # # 3
        # _h = self.h_mixer(x)
        # _w = self.w_mixer(x)
        # _c = self.c_mixer(x)
        # _chw = _c * _h * _w

        return self.proj(_chw)


class TransLayer(nn.Module):
    def __init__(self, planes: int, kernel_size: int, act_layer: nn.Module) -> None:
        super().__init__()

        self.sg = SelfGate(planes, kernel_size)
        self.norm1 = LayerNorm4D(planes)

        self.mlp = MLP4D(planes, planes * 2, act_layer=act_layer)
        self.norm2 = LayerNorm4D(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sg(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



@ARCH_REGISTRY.register()
class EXnet(nn.Module):
    r"""
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 n_l: int, dim: int, kernel_size: int) -> None:
        super(EXnet, self).__init__()

        self.sub_mean = NTIREMeanShift(255, sign=-1, data_type='DF2K')
        self.add_mean = NTIREMeanShift(255, sign=1, data_type='DF2K')

        self.head = nn.Sequential(Conv2d3x3(num_in_ch, dim))

        self.body = nn.Sequential(*[TransLayer(planes=dim,
                                               kernel_size=kernel_size,
                                               act_layer=Swish)  # noqa
                                    for _ in range(n_l)])

        self.tail = Upsampler(upscale=upscale, in_channels=dim,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x


