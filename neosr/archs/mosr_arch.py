import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_
from neosr.archs.arch_util import DropPath, DySample, net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.reshape(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(self, dim,
                 expansion_ratio=8 / 3,
                 kernel_size=7,
                 conv_ratio=1.0,
                 drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.Mish()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                              groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x  # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut


class GatedBlocks(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_blocks,
                 drop_path,
                 expansion_ratio
                 ):
        super().__init__()
        self.in_to_out = Dynamic_conv2d(in_dim, out_dim, 3, padding=1)

        self.gcnn = nn.Sequential(
            *[GatedCNNBlock(out_dim, expansion_ratio=expansion_ratio, drop_path=drop_path)
              for _ in range(n_blocks)
              ])

    def forward(self, x):
        x = self.in_to_out(x)
        x = x.permute(0, 2, 3, 1)
        x = self.gcnn(x)
        return x.permute(0, 3, 1, 2)


@ARCH_REGISTRY.register()
class mosr(nn.Module):
    def __init__(self,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 upscale: int = 4,
                 blocks: tuple[int] = (3, 3, 9, 3),
                 dims: tuple[int] = (48, 96, 192, 288),
                 upsampler: str = "ps",
                 drop_path: float = 0.,
                 expansion_ratio: float = 1.0
                 ):
        super(mosr, self).__init__()
        len_blocks = len(blocks)
        dims = [in_ch] + list(dims)
        self.gblocks = nn.Sequential(
            *[GatedBlocks(dims[i], dims[i + 1], blocks[i], drop_path=drop_path, expansion_ratio=expansion_ratio)
              for i in range(len_blocks)]
        )

        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(dims[-1],
                          out_ch * (upscale ** 2),
                          3, padding=1),
                nn.PixelShuffle(upscale)
            )
        elif upsampler == "dys":
            self.upsampler = DySample(dims[-1], out_ch, upscale)
        elif upsampler == "conv":
            if upsampler != 1:
                msg = "conv supports only 1x"
                raise ValueError(msg)

            self.upsampler = nn.Conv2d(dims[-1],
                                       out_ch,
                                       3, padding=1)
        else:
            raise NotImplementedError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                ["ps", "dys", "conv"] conv supports only 1x')
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.gblocks(x)
        return self.upsampler(x)
