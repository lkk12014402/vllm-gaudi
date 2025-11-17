from safetensors.torch import save_file
import torch

"""
RowParallelLinear(in_features=2048, output_features=1024, bias=False, tp_size=1, reduce_results=True)
"""

inp = torch.load("vllm_ios_per_tensor/0000_model_model_layers_0_self_attn_o_proj_15054e2e_in.pt")
out = torch.load("vllm_ios_per_tensor/0000_model_model_layers_0_self_attn_o_proj_15054e2e_out.pt")

print(inp.shape)
print(out.shape)


# read weight, scales


from safetensors.torch import load_file

loaded = load_file("../Qwen3-0.6B-FP8/model.safetensors")
print(loaded.keys())          # ['input', 'output']
# print(loaded["input"].shape)  # torch.Size([1, 64, 3072])
# 'model.layers.0.self_attn.o_proj.input_scale', 'model.layers.0.self_attn.o_proj.weight_scale'
input_scale = loaded["model.layers.0.self_attn.o_proj.input_scale"]

weight_scale = loaded["model.layers.0.self_attn.o_proj.weight_scale"]

weight = loaded["model.layers.0.self_attn.o_proj.weight"]
print("input_scale: ", input_scale.shape)
print("weight_scale: ", weight_scale.shape)
print("weight: ", weight.shape)

print(weight)

data = {
    "input": inp,
    "ref_output": out[:, :, :8].contiguous(),
    "input_scale": input_scale * 2.0,
    "weight_scale": weight_scale * 2.0,
    "weight": (weight[:8, :].contiguous().float() * 0.5).to(torch.float8_e4m3fn)
}

"""
data = {
    "input": inp,
    "ref_output": out,
    "input_scale": input_scale * 2.0,
    "weight_scale": weight_scale * 2.0,
    "weight": (weight.float() * 0.5).to(torch.float8_e4m3fn) 
}
"""

save_file(data, "linear_w8a8fp8_static_per_tensor.safetensors")

