# Mamba Out Super Resolution
This architecture was inspired by MambaOut

```py
def pixelshuffle_scale(ps_size: int, channels: int):
    return math.isqrt(ps_size // channels)


def dysample_scale(ds_size: int):
    return math.isqrt(ds_size // 8)


def get_seq_len(state_dict: Mapping[str, object], seq_key: str) -> int:
    """
    Returns the length of a sequence in the state dict.

    The length is detected by finding the maximum index `i` such that
    `{seq_key}.{i}.{suffix}` is in `state` for some suffix.

    Example:
        get_seq_len(state, "body") -> 5
    """
    prefix = seq_key + '.'

    keys: set[int] = set()
    for k in state_dict.keys():
        if k.startswith(prefix):
            index = k[len(prefix) :].split('.', maxsplit=1)[0]
            keys.add(int(index))

    if len(keys) == 0:
        return 0
    return max(keys) + 1


def detect(state):
        len_blocks = get_seq_len(state, 'gblocks')
        blocks = [get_seq_len(state, f'gblocks.{index}.gcnn') for index in range(len_blocks)]
        dims = [state[f'gblocks.{index}.gcnn.0.norm.weight'].shape[0] for index in range(len_blocks)]
        in_ch = state['gblocks.0.in_to_out.weight'].shape[1]
        expansion = state['gblocks.0.gcnn.0.fc2.weight'].shape
        expansion_ratio = expansion[1] / expansion[0]
        if 'upsampler.weight' in state:
            upsampler = 'conv'
            upscale = 1
            out_ch = state['upsampler.weight'].shape[0]
        elif 'upsampler.0.weight' in state:
            upsampler = 'ps'
            out_ch = in_ch
            upscale = pixelshuffle_scale(state['upsampler.0.weight'].shape[0], out_ch)
        else:
            upsampler = 'dys'
            out_ch = state['upsampler.end_conv.weight'].shape[0]
            upscale = dysample_scale(state['upsampler.offset.weight'].shape[0])
""")
```
### References:
Training code from [NeoSR](https://github.com/muslll/neosr)

[MambaOut](https://github.com/yuweihao/MambaOut)

[DCCM](https://github.com/dslisleedh/PLKSR/blob/main/plksr/archs/plksr_arch.py#L44)

[get_seq_len](https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/util/__init__.py#L60)
### TODO:
- release metrics and pretrain
