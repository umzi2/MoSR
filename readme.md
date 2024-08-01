# Mamba Out Super Resolution
This architecture was inspired by MambaOut

```py 
def detect(state):
    # Get values from state
    n_block = get_seq_len(state, "gblocks") - 6
    in_ch = state["gblocks.0.weight"].shape[1]
    dim = state["gblocks.0.weight"].shape[0]

    # Calculate expansion ratio and convolution ratio
    expansion_ratio = (state["gblocks.1.fc1.weight"].shape[0] / 
                       state["gblocks.1.fc1.weight"].shape[1]) / 2
    conv_ratio = state["gblocks.1.conv.weight"].shape[0] / dim

    # Determine upsampler type and calculate upscale
    if "upsampler.init_pos" in state:
        upsampler = "dys"
        out_ch = state["upsampler.end_conv.weight"].shape[0]
        upscale = math.isqrt(state["upsampler.offset.weight"].shape[0] // 8)
    elif "upsampler.ps_act.0.weight" in state:
        upsampler = "psa"
        out_ch = in_ch
        upscale = math.isqrt(state["upsampler.ps_act.0.weight"].shape[0] // out_ch)
    else:
        upsampler = "ps"
        out_ch = in_ch
        upscale = math.isqrt(state["upsampler.0.weight"].shape[0] // out_ch)

    # Print results
    print(f"""    in_ch: {in_ch}
    out_ch: {out_ch}
    dim: {dim}
    n_block: {n_block}
    upsampler: {upsampler}
    upscale: {upscale}
    expansion_ratio: {expansion_ratio}
    conv_ratio: {conv_ratio}""")

signature = [
    'gblocks.0.weight',
    'gblocks.0.bias',
    'gblocks.1.norm.weight',
    'gblocks.1.norm.bias',
    'gblocks.1.fc1.weight',
    'gblocks.1.fc1.bias',
    'gblocks.1.conv.weight',
    'gblocks.1.conv.bias',
    'gblocks.1.fc2.weight',
    'gblocks.1.fc2.bias',
]
```
### References:
Training code from [NeoSR](https://github.com/muslll/neosr)

[MambaOut](https://github.com/yuweihao/MambaOut)

### TODO:
- release metrics and pretrain
