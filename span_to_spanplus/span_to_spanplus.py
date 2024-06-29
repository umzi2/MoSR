import torch
from convert_keys import key_to_key

input_folder = "no_norm_span.pth"
out_folder = "span_plus.pth"


def load_model(state_dict):
    unwrap_keys = ["state_dict", "params_ema", "params-ema", "params", "model", "net"]
    for key in unwrap_keys:
        if key in state_dict and isinstance(state_dict[key], dict):
            return state_dict[key]


model = torch.load(input_folder)
span_model = load_model(model)
span_model.pop("no_norm")
for i in list(span_model.keys()):
    span_model[key_to_key[i]] = span_model.pop(i)
torch.save(span_model, out_folder)
