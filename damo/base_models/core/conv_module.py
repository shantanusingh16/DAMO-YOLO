import warnings
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def kaiming_init(module, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """Initialize the weights of the module using Kaiming initialization."""
    if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
        init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    else:
        raise TypeError(f"Unsupported module type: {type(module)}")


def constant_init(module, val, bias=0):
    """Initialize the weights and bias of the module to a constant value."""
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
        init.constant_(module.weight, val)
        if module.bias is not None:
            init.constant_(module.bias, bias)
    else:
        raise TypeError(f"Unsupported module type: {type(module)}")


def build_norm_layer(norm_cfg, num_features):
    """Build normalization layer based on configuration."""
    layer_type = norm_cfg.get('type', 'BN') if norm_cfg else 'BN'
    if layer_type == 'BN':
        return 'bn', nn.BatchNorm2d(num_features)
    elif layer_type == 'SyncBN':
        return 'sync_bn', nn.SyncBatchNorm(num_features)
    elif layer_type == 'GN':
        num_groups = norm_cfg.get('num_groups', 32)
        return 'gn', nn.GroupNorm(num_groups, num_features)
    elif layer_type == 'LN':
        return 'ln', nn.LayerNorm(num_features)
    elif layer_type == 'IN':
        return 'in', nn.InstanceNorm2d(num_features)
    else:
        raise ValueError(f"Unsupported norm layer type: {layer_type}")


def build_activation_layer(act_cfg):
    """Build activation layer based on configuration."""
    if act_cfg is None:
        return nn.ReLU(inplace=True)
    
    layer_type = act_cfg.get('type', 'ReLU')
    if layer_type == 'ReLU':
        inplace = act_cfg.get('inplace', True)
        return nn.ReLU(inplace=inplace)
    elif layer_type == 'LeakyReLU':
        negative_slope = act_cfg.get('negative_slope', 0.01)
        inplace = act_cfg.get('inplace', True)
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif layer_type == 'PReLU':
        return nn.PReLU()
    elif layer_type == 'ELU':
        alpha = act_cfg.get('alpha', 1.0)
        inplace = act_cfg.get('inplace', True)
        return nn.ELU(alpha=alpha, inplace=inplace)
    elif layer_type == 'Sigmoid':
        return nn.Sigmoid()
    elif layer_type == 'Tanh':
        return nn.Tanh()
    elif layer_type == 'GELU':
        return nn.GELU()
    elif layer_type == 'Swish':
        return nn.SiLU()  # Swish is also known as SiLU in PyTorch
    else:
        raise ValueError(f"Unsupported activation layer type: {layer_type}")


def build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    """Build convolution layer based on configuration."""
    layer_type = conv_cfg.get('type', 'Conv2d') if conv_cfg else 'Conv2d'
    if layer_type == 'Conv2d':
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    elif layer_type == 'Conv1d':
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    elif layer_type == 'Conv3d':
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    else:
        raise ValueError(f"Unsupported conv layer type: {layer_type}")


def build_padding_layer(pad_cfg, padding):
    """Build padding layer based on configuration."""
    layer_type = pad_cfg.get('type', 'zeros') if pad_cfg else 'zeros'
    if layer_type == 'zeros':
        return nn.ZeroPad2d(padding)
    elif layer_type == 'reflection':
        return nn.ReflectionPad2d(padding)
    elif layer_type == 'replication':
        return nn.ReplicationPad2d(padding)
    elif layer_type == 'constant':
        value = pad_cfg.get('value', 0)
        return nn.ConstantPad2d(padding, value)
    else:
        raise ValueError(f"Unsupported padding layer type: {layer_type}")


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act')):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(
                norm_cfg, norm_channels)  # type: ignore
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None  # type: ignore

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,
                x: torch.Tensor,
                activate: bool = True,
                norm: bool = True) -> torch.Tensor:
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x
