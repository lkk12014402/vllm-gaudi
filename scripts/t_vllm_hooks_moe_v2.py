import torch
import uuid
import os
from typing import Any, Dict, List

def _to_cpu_detach(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_detach(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_cpu_detach(v) for k, v in obj.items()}
    return obj

def install_fused_moe_io_hooks(self,
                               layer_filter_prefix="model.model.layers",
                               only_first_layer=True,
                               capture_original=True):
    """
    在所有 FusedMoE 层（或仅第一层）安装 pre + forward hook。
    记录:
      original_input_hidden / original_input_router (pre_hook clone)
      forward_input_hidden / forward_input_router (forward阶段看到的)
      outputs: list[Tensor] 统一化
      layer_name
    """
    model = self.model_runner.model
    self.__io_cache = []
    self.__io_handles = []

    def make_pre_hook(name):
        def _pre(mod, inputs):
            if not capture_original:
                return
            # inputs: (hidden_states, router_logits)
            rec: Dict[str, Any] = {
                "layer": name,
                "original_input_hidden": _to_cpu_detach(inputs[0]),
                "original_input_router": _to_cpu_detach(inputs[1]) if len(inputs) > 1 else None,
            }
            # 先放一个占位，后面 forward_hook 会补充
            self.__io_cache.append(rec)
        return _pre

    def make_forward_hook(name):
        def _forward(mod, inputs, output):
            # 匹配已经添加的最后一个记录（假设顺序一致）
            # 如果 capture_original=False 或 pre_hook 没有运行，需要新建记录
            if not self.__io_cache or self.__io_cache[-1].get("layer") != name or "forward_input_hidden" in self.__io_cache[-1]:
                self.__io_cache.append({"layer": name})

            rec = self.__io_cache[-1]
            rec["forward_input_hidden"] = _to_cpu_detach(inputs[0])
            rec["forward_input_router"] = _to_cpu_detach(inputs[1]) if len(inputs) > 1 else None

            # 规范化输出
            outputs: List[torch.Tensor] = []
            if torch.is_tensor(output):
                outputs = [output]
            elif isinstance(output, (list, tuple)):
                for o in output:
                    if torch.is_tensor(o):
                        outputs.append(o)
            else:
                # 其他类型直接跳过
                pass
            rec["outputs"] = [_to_cpu_detach(o) for o in outputs]
            rec["output_shapes"] = [tuple(o.shape) for o in outputs]

        return _forward

    # 遍历模块找 FusedMoE
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    only_first_layer = True

    for name, module in model.named_modules():
        if not name.startswith(layer_filter_prefix):
            continue
        if isinstance(module, FusedMoE):
            if only_first_layer and ".layers.0." not in name:
                continue
            # 安装 pre + forward hook
            pre_h = module.register_forward_pre_hook(make_pre_hook(name), with_kwargs=False)
            fwd_h = module.register_forward_hook(make_forward_hook(name), with_kwargs=False)
            self.__io_handles.append(pre_h)
            self.__io_handles.append(fwd_h)

    return {
        "hooked_layers": len(self.__io_handles)//2,
        "total_handles": len(self.__io_handles),
        "example_layer_names": list({rec["layer"] for rec in getattr(self, "__io_cache", [])})[:3],
    }

def fetch_and_clear_fused_moe_ios(self):
    data = getattr(self, "__io_cache", [])
    self.__io_cache = []
    return {
        "num_records": len(data),
        "records": data,
    }

def dump_fused_moe_ios(self, out_dir="./vllm_moe_ios", fmt="pt"):
    os.makedirs(out_dir, exist_ok=True)
    dumped = []
    for i, rec in enumerate(getattr(self, "__io_cache", [])):
        uid = uuid.uuid4().hex[:8]
        base = f"{i:04d}_{rec['layer'].replace('.', '_')}_{uid}"
        # 输入可能是 original_input_hidden 或 forward_input_hidden，也都存
        for key in ["original_input_hidden", "original_input_router",
                    "forward_input_hidden", "forward_input_router"]:
            val = rec.get(key)
            if val is None:
                continue
            path = os.path.join(out_dir, f"{base}_{key}.{fmt}")
            if fmt == "pt":
                torch.save(val, path)
            else:
                import numpy as np
                np.save(path, val.numpy())
        # 输出
        for j, out_t in enumerate(rec.get("outputs", [])):
            path = os.path.join(out_dir, f"{base}_out{j}.{fmt}")
            if fmt == "pt":
                torch.save(out_t, path)
            else:
                import numpy as np
                np.save(path, out_t.numpy())
        dumped.append({
            "layer": rec["layer"],
            "output_shapes": rec.get("output_shapes"),
            "base": base
        })
    return {"dumped": len(dumped), "detail": dumped}

def remove_fused_moe_io_hooks(self):
    for h in getattr(self, "__io_handles", []):
        h.remove()
    removed = len(getattr(self, "__io_handles", []))
    self.__io_handles = []
    return {"removed_handles": removed}


from vllm import LLM, SamplingParams
os.environ["VLLM_SKIP_WARMUP"] = "true"


model = "../Qwen3-30B-A3B-FP8-Static"
#kwargs = {"tensor_parallel_size": 4, "enforce_eager": True}
kwargs = {"tensor_parallel_size": 2, "enforce_eager": False}
kwargs = {"tensor_parallel_size": 2, "enforce_eager": True}
kwargs = {"tensor_parallel_size": 1, "enforce_eager": False}
# kwargs = {"tensor_parallel_size": 1, "enforce_eager": True}
if os.path.basename(model) in ["Qwen3-30B-A3B", "DeepSeek-V2-Lite-Chat"]:
    kwargs["enable_expert_parallel"] = True
elif "Qwen3-30B-A3B" in model:
    kwargs["enable_expert_parallel"] = True
print(kwargs)


def main():
    llm = LLM(model=model, max_model_len=4096, trust_remote_code=True, **kwargs)

    print(llm.collective_rpc(install_fused_moe_io_hooks))

    outs = llm.generate(
        prompts=["Hooks with vLLM"],
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),)


    data = llm.collective_rpc(fetch_and_clear_fused_moe_ios)
    print(data)
    exit()

    # io_per_worker = llm.collective_rpc(fetch_and_clear_block_ios)
    # print(io_per_worker)

    print(llm.collective_rpc(dump_block_ios))

    print(llm.collective_rpc(remove_block_io_hooks))



if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


    main()
