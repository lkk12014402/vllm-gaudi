import re
import json
from collections import defaultdict
from typing import Dict, List
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

# 你现有的映射：键 -> 文件名
# 建议从一个 json 文件读入；这里直接示例放字典
weights_map: Dict[str, str] = {
    # 示例（请用你的完整map替换）
    "model.layers.0.mlp.experts.0.down_proj.input_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.down_proj.weight": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.down_proj.weight_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.gate_proj.input_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.gate_proj.weight_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.up_proj.input_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.up_proj.weight": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.0.up_proj.weight_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.down_proj.input_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.down_proj.weight": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.down_proj.weight_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.gate_proj.input_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.gate_proj.weight": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.gate_proj.weight_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.up_proj.input_scale": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.up_proj.weight": "model-00001-of-00007.safetensors",
    "model.layers.0.mlp.experts.1.up_proj.weight_scale": "model-00001-of-00007.safetensors",
    # ... 继续添加其它专家
}

# 正则抽取: layer id, expert id, proj type (gate_proj/up_proj/down_proj), field (weight/weight_scale/input_scale)
PATTERN = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight(?:_scale)?|input_scale)"
)

def group_keys_by_file(weights_map: Dict[str, str]) -> Dict[str, List[str]]:
    file_to_keys = defaultdict(list)
    for k, fname in weights_map.items():
        file_to_keys[fname].append(k)
    return file_to_keys

def load_all(file_to_keys: Dict[str, List[str]], root_dir: str = ".") -> Dict[str, torch.Tensor]:
    """
    读取所有 safetensors，返回 key->tensor
    root_dir: 如果文件路径需要前缀目录
    """
    loaded = {}
    for fname, keys in file_to_keys.items():
        path = f"{root_dir}/{fname}" if not fname.startswith("/") else fname
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in keys:
                # safetensors 内部的实际键需与外部 k 一致（如果不一致，需要映射）
                if k not in f.keys():
                    raise KeyError(f"Key {k} not found in file {path}. Available: {list(f.keys())[:5]} ...")
                loaded[k] = f.get_tensor(k)
    return loaded

def infer_sizes(example_tensors: Dict[str, torch.Tensor]):
    # 选一个 gate_proj.weight
    gate_key = next(k for k in example_tensors if "gate_proj.weight" in k)
    up_key   = next(k for k in example_tensors if "up_proj.weight" in k)
    down_key = next(k for k in example_tensors if "down_proj.weight" in k)
    gate_w = example_tensors[gate_key]
    up_w   = example_tensors[up_key]
    down_w = example_tensors[down_key]
    # 期望尺寸: (intermediate_size, hidden_size) 与 (hidden_size, intermediate_size)
    if gate_w.shape != up_w.shape:
        raise ValueError(f"gate_proj.weight shape {gate_w.shape} != up_proj.weight shape {up_w.shape}")
    intermediate_size, hidden_size = gate_w.shape
    if down_w.shape != (hidden_size, intermediate_size):
        raise ValueError(f"down_proj.weight shape {down_w.shape}, expected ({hidden_size}, {intermediate_size})")
    return hidden_size, intermediate_size

def collect_expert_ids(loaded: Dict[str, torch.Tensor]) -> List[int]:
    expert_ids = set()
    for k in loaded:
        m = PATTERN.match(k)
        if not m:
            continue
        expert_ids.add(int(m.group(2)))
    return sorted(expert_ids)

def build_fused(
    loaded: Dict[str, torch.Tensor],
    hidden_size: int,
    intermediate_size: int,
    expert_ids: List[int],
    layer_id: int,
):
    # 构建容器
    w13_weights      = []
    w13_weight_scales = []
    w13_input_scales  = []
    w2_weights       = []
    w2_weight_scales  = []
    w2_input_scales   = []

    for e in expert_ids:
        base = f"model.layers.{layer_id}.mlp.experts.{e}"

        gate_w = loaded[f"{base}.gate_proj.weight"]
        up_w   = loaded[f"{base}.up_proj.weight"]
        down_w = loaded[f"{base}.down_proj.weight"]

        # 拼 w13
        w13_w = torch.cat([gate_w, up_w], dim=0)  # (2*intermediate_size, hidden_size)
        w13_weights.append(w13_w)

        # weight_scale 可能是标量(per-tensor) 或 向量(per-channel)
        gate_w_scale = loaded.get(f"{base}.gate_proj.weight_scale")
        up_w_scale   = loaded.get(f"{base}.up_proj.weight_scale")
        down_w_scale = loaded.get(f"{base}.down_proj.weight_scale")

        if gate_w_scale is not None and up_w_scale is not None:
            # 维度判断：如果是标量 => cat 后得长度2；如果是 (intermediate_size,) => cat 后(2*intermediate_size,)
            w13_weight_scales.append(torch.cat([gate_w_scale.view(-1), up_w_scale.view(-1)], dim=0))
        else:
            w13_weight_scales.append(None)

        if down_w_scale is not None:
            w2_weight_scales.append(down_w_scale.view(-1))
        else:
            w2_weight_scales.append(None)

        # input_scale（若存在且 gate/up 相等可直接保留两个；否则拼接）
        gate_in_scale = loaded.get(f"{base}.gate_proj.input_scale")
        up_in_scale   = loaded.get(f"{base}.up_proj.input_scale")
        down_in_scale = loaded.get(f"{base}.down_proj.input_scale")

        if gate_in_scale is not None and up_in_scale is not None:
            w13_input_scales.append(torch.cat([gate_in_scale.view(-1), up_in_scale.view(-1)], dim=0))
        else:
            w13_input_scales.append(None)

        if down_in_scale is not None:
            w2_input_scales.append(down_in_scale.view(-1))
        else:
            w2_input_scales.append(None)

        # w2
        w2_weights.append(down_w)

    # 叠成最终张量（None 的跳过）
    w13_weights_tensor = torch.stack(w13_weights, dim=0)  # (num_experts, 2*intermediate_size, hidden_size)
    w2_weights_tensor  = torch.stack(w2_weights, dim=0)   # (num_experts, hidden_size, intermediate_size)

    def stack_or_none(lst: List[torch.Tensor | None]):
        vals = [x for x in lst if x is not None]
        if len(vals) == 0:
            return None
        # 确保所有 shape 一致
        shape0 = vals[0].shape
        if any(x.shape != shape0 for x in vals[1:]):
            raise ValueError("Inconsistent scale shapes among experts.")
        return torch.stack(vals, dim=0)  # (num_experts, scale_len)

    w13_weight_scales_tensor = stack_or_none(w13_weight_scales)
    w13_input_scales_tensor  = stack_or_none(w13_input_scales)
    w2_weight_scales_tensor  = stack_or_none(w2_weight_scales)
    w2_input_scales_tensor   = stack_or_none(w2_input_scales)

    return {
        "w13_weight": w13_weights_tensor,
        "w2_weight":  w2_weights_tensor,
        "w13_weight_scale": w13_weight_scales_tensor,
        "w2_weight_scale":  w2_weight_scales_tensor,
        "w13_input_scale":  w13_input_scales_tensor,
        "w2_input_scale":   w2_input_scales_tensor,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": len(expert_ids),
        "expert_ids": expert_ids,
        "layer_id": layer_id,
    }

def main(
    weights_map: Dict[str, str],
    root_dir: str = ".",
    layer_id: int = 0,
    out_prefix: str = "fused_layer0",
    save_pt: bool = True,
    save_sft: bool = True,
):
    file_to_keys = group_keys_by_file(weights_map)
    loaded = load_all(file_to_keys, root_dir=root_dir)

    hidden_size, intermediate_size = infer_sizes(loaded)
    expert_ids = collect_expert_ids(loaded)

    fused = build_fused(
        loaded=loaded,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        expert_ids=expert_ids,
        layer_id=layer_id,
    )

    print("Fused summary:",
          {k: v.shape if torch.is_tensor(v) else v for k, v in fused.items()
           if k.startswith("w")})

    # 保存为 .pt
    if save_pt:
        torch.save(fused, f"{out_prefix}_fused.pt")
        print(f"Saved PyTorch fused file: {out_prefix}_fused.pt")

    # 保存为 safetensors
    if save_sft:
        sft_dict = {k: v for k, v in fused.items() if torch.is_tensor(v)}
        save_safetensors(sft_dict, f"{out_prefix}_fused.safetensors")
        print(f"Saved safetensors fused file: {out_prefix}_fused.safetensors")

if __name__ == "__main__":
    # 根据实际情况修改 root_dir（存放 model-00001-of-00007.safetensors 的目录）
    main(weights_map, root_dir=".", layer_id=0, out_prefix="layer0")
