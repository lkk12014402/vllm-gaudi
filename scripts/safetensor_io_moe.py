from safetensors.torch import save_file
import torch

"""
(EngineCore_DP0 pid=377312) ====================================module: FusedMoE(
(EngineCore_DP0 pid=377312)   global_num_experts=128, local_num_experts=128, top_k=8, intermediate_size_per_partition=768, tp_size=1,
(EngineCore_DP0 pid=377312)   ep_size=1, reduce_results=True, renormalize=True, use_grouped_topk=False, scoring_func='softmax', activation='silu'
(EngineCore_DP0 pid=377312)   (moe_op): VllmMixtureOfExpertsOpFP8PerChannel(
(EngineCore_DP0 pid=377312)     (w13_list): ModuleList(
(EngineCore_DP0 pid=377312)       (0-127): 128 x MoeFP8Matmul()
(EngineCore_DP0 pid=377312)     )
(EngineCore_DP0 pid=377312)     (w2_list): ModuleList(
(EngineCore_DP0 pid=377312)       (0-127): 128 x MoeFP8Matmul()
(EngineCore_DP0 pid=377312)     )
(EngineCore_DP0 pid=377312)   )
(EngineCore_DP0 pid=377312) )

"""

router_logits = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_router_logits.pt", weights_only=False)
hidden_states = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_hidden_states.pt", weights_only=False)
out = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_out0.pt", weights_only=False)


w13_weight = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_w13_weight.pt", weights_only=False)
w13_weight_scale = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_w13_weight_scale.pt", weights_only=False)
w13_input_scale = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_w13_input_scale.pt", weights_only=False)


w2_weight = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_w2_weight.pt", weights_only=False)
w2_weight_scale = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_w2_weight_scale.pt", weights_only=False)
w2_input_scale = torch.load("vllm_ios/0000_model_model_layers_0_mlp_experts_c5bf8f88_w2_input_scale.pt", weights_only=False)

print("router_logits: ", router_logits.shape)
print("hidden_states: ", hidden_states.shape)
print("out: ", out.shape)

print(router_logits)
print(hidden_states)
print(out)

print("w13_weight: ", w13_weight.shape)
print("w13_weight_scale: ", w13_weight_scale.shape)
print("w13_input_scale: ", w13_input_scale.shape)


print("w2_weight: ", w2_weight.shape)
print("w2_weight_scale", w2_weight_scale.shape)
print("w2_input_scale: ", w2_input_scale.shape)


data = {
    "router_logits": router_logits,
    "hidden_states": hidden_states,
    "ref_output": out,
    # "w2_weight": (w2_weight.float() * 0.5).to(torch.float8_e4m3fn),
    "w2_weight": w2_weight,
    "w2_weight_scale": w2_weight_scale,
    "w2_input_scale": w2_input_scale,
    "w13_weight": w13_weight,
    "w13_weight_scale": w13_weight_scale,
    "w13_input_scale": w13_input_scale.expand(128).clone().contiguous()
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

save_file(data, "moe_w8a8fp8_static_per_channel.safetensors")

