# Mamba Out Super Resolution
This architecture was inspired by MambaOut

```py 
def detect(state):
    len_blocks = get_seq_len(state, "gblocks")
    blocks = [get_seq_len(state, f"gblocks.{index}.gcnn")
              for index in range(len_blocks)
              ]
    dims = [state[f"gblocks.{index}.gcnn.0.norm.weight"].shape[0]
            for index in range(len_blocks)
            ]
    state_keys = state.keys()
    in_ch = state["gblocks.0.in_to_out.weight"].shape[2]
    expansion = state["gblocks.0.gcnn.0.fc2.weight"].shape
    expansion_ratio = expansion[1] / expansion[0]
    if "upsampler.weight" in state_keys:
        upsampler = "conv"
        upscale = 1
        out_ch = state["upsampler.weight"].shape[0]
    elif "upsampler.0.weight" in state_keys:
        upsampler = "ps"
        out_ch = in_ch
        upscale = int((state["upsampler.0.weight"].shape[0] / out_ch) ** 0.5)
    else:
        upsampler = "dys"
        out_ch = state["upsampler.end_conv.weight"].shape[0]
        upscale = int((state["upsampler.offset.weight"].shape[0] / 2 / 4) ** 0.5)

    print(f"""
in_ch = {in_ch}
out_ch = {out_ch}
upscale = {upscale}
blocks = {blocks}
dims = {dims}
upscaler = {upsampler}
expansion_ratio = {expansion_ratio}
""")
```
### References:
Training code from [NeoSR](https://github.com/muslll/neosr)

[MambaOut](https://github.com/yuweihao/MambaOut)

[DyConv](https://github.com/kaijieshi7/Dynamic-convolution-Pytorch)
### TODO:
- release metrics and pretrain
