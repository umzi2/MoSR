import torch
from torch import nn
from torch.nn.init import trunc_normal_
from neosr.archs.arch_util import DropPath, DySample
from neosr.utils.registry import ARCH_REGISTRY


upscale, __ = net_opt()


class DCCM(nn.Sequential):
    "Doubled Convolutional Channel Mixer"

    def __init__(self, in_ch: int, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(out_ch * 2, out_ch, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        self.in_to_out = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.dcm_mix = DCCM(out_dim, out_dim)
        self.gcnn = nn.Sequential(
            *[GatedCNNBlock(out_dim,
                            expansion_ratio=expansion_ratio,
                            drop_path=drop_path[i])
              for i in range(n_blocks)
              ])

    def forward(self, x):
        x = self.in_to_out(x)
        short_cut = x
        x = self.gcnn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dcm_mix(x)
        return x + short_cut


@ARCH_REGISTRY.register()
class mosr(nn.Module):
    def __init__(self,
                 in_ch: int = 3,
                 out_ch: int = 3,
                 upscale: int = upscale,
                 blocks: list[int] = [3, 3, 15, 3],
                 dims: list[int] = [64, 96, 192, 288],
                 upsampler: str = "ps",
                 drop_path: float = 0.1,
                 expansion_ratio: float = 1.0
                 ):
        super(mosr, self).__init__()
        len_blocks = len(blocks)
        dims = [in_ch] + list(dims)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(blocks))]
        self.gblocks = nn.Sequential(
            *[GatedBlocks(dims[i], dims[i + 1], blocks[i], drop_path=dp_rates[sum(blocks[:i]):sum(blocks[:i+1])],
                          expansion_ratio=expansion_ratio
                          )
              for i in range(len_blocks)]
        )

        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(dims[-1],
                          out_ch * (upscale ** 2),
                          3, padding=1),
                nn.PixelShuffle(upscale)
            )
            # trunc_normal_(self.upsampler[0].weight, std=0.02)
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

    def forward(self, x):
        x = self.gblocks(x)
        return self.upsampler(x)

