import torch
import torch.nn.functional as F

# -----------------------------
# 维度与路由（示例数值）
# -----------------------------
H = 4096             # hidden_dim
FF = 14336           # ffn_dim
N_EXPERTS = 8
N_TOKENS  = 128
TOP_K = 2
ACTIVATION = "gelu"  # 可选: "gelu" | "relu" | "silu"

# 输入 & 路由权重（Top-K）
hidden_states = torch.randn(N_TOKENS, H, dtype=torch.bfloat16)
logits = torch.randn(N_TOKENS, N_EXPERTS, dtype=torch.float32)
routing_weights_full = F.softmax(logits, dim=-1, dtype=torch.float32)
router_weights, expert_routing_table = torch.topk(routing_weights_full, TOP_K, dim=-1)  # shapes: [N, K], [N, K]
router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
router_weights = router_weights.to(dtype=torch.bfloat16)

# -----------------------------
# 专家权重（fused: 只用 w12 与 w3）
# w12 形状需与你的 fused 格式一致（常见为把前两次 GEMM 的权重按列拼接，例如 (H, 2*FF)）
# w3 形状通常为 (FF, H)
# -----------------------------
# 示例中演示从“未融合”的 w1/w2 构造 w12；若你已有框架导出的 fused 权重，直接装载即可。
w1 = [torch.randn(H, FF, dtype=torch.bfloat16) for _ in range(N_EXPERTS)]
w2 = [torch.randn(H, FF, dtype=torch.bfloat16) for _ in range(N_EXPERTS)]
w12 = [torch.cat([w1[e], w2[e]], dim=1).to("hpu") for e in range(N_EXPERTS)]  # (H, 2*FF) on HPU
w3  = [torch.randn(FF, H, dtype=torch.bfloat16).to("hpu") for _ in range(N_EXPERTS)]  # (FF, H) on HPU

# -----------------------------
# 静态量化的 scales（Tensor 版本）
# 这些通常来自“measurement 模式”多 batch 统计 + 你的离线策略（P99/max/EMA 等）
# 类型要求：fp8_fused_weights 重载需要 Tensor / Tensor[]（在 HPU 或 CPU 上，官方建议 FP32/BF16）
# -----------------------------
# 全局的 hidden_states 激活 scale（标量 Tensor）
d_scale_hidden_states = torch.tensor(1.0, dtype=torch.float32).to("hpu")  # ← 示例值，占位

# 第三次 GEMM 第一输入的激活 scales：逐 expert，Tensor 列表
d_scale_intermediate_hidden_states = [
    torch.tensor(1.0, dtype=torch.float32).to("hpu")  # ← 用你的统计结果替换
    for _ in range(N_EXPERTS)
]

# 权重 scales：逐 expert，Tensor 列表（若你有更细粒度，如 per-channel/per-block，请改用对应重载）
d_scale_w12 = [
    torch.tensor(1.0, dtype=torch.float32).to("hpu")  # ← 用你的离线量化结果替换
    for _ in range(N_EXPERTS)
]
d_scale_w3 = [
    torch.tensor(1.0, dtype=torch.float32).to("hpu")  # ← 用你的离线量化结果替换
    for _ in range(N_EXPERTS)
]

# -----------------------------
# 推理前向（FP8/BF8 fused - Tensor scales 重载）
# 签名（节选）：fp8_fused_weights(
#   hidden_states, expert_routing_table, router_weights,
#   w12, w3,
#   d_scale_hidden_states, d_scale_intermediate_hidden_states, d_scale_w12, d_scale_w3,
#   permuted_weights, activation, experts_min, experts_max, *, chunk_size=0, total_experts=0
# ) -> Tensor
# -----------------------------
with torch.inference_mode():
    out = torch.ops.hpu.mixture_of_experts(
        hidden_states.to("hpu"),
        expert_routing_table.to(torch.int64).to("hpu"),
        router_weights.to("hpu"),

        # fused 权重（只传 w12, w3）
        w12,
        w3,

        # ==== FP8/BF8（Tensor scales）====
        d_scale_hidden_states,
        d_scale_intermediate_hidden_states,
        d_scale_w12,
        d_scale_w3,

        # 其后是：permuted_weights, activation, experts_min, experts_max
        False,                 # 若你已预先按内核需求排列权重，可设 True
        ACTIVATION,
        0, N_EXPERTS - 1,

        # （可选）专家并行/切分支持：
        # chunk_size=0,
        # total_experts=N_EXPERTS,
    )

print("MoE output shape:", out.shape)  # 期望: [N_TOKENS, H]
