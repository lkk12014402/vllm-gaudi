import os
# from multiprocessing import Process, freeze_support
import inspect
# freeze_support()

from vllm import LLM, SamplingParams

os.environ["VLLM_SKIP_WARMUP"] = "true"
prompts = [
    "Hello, my name is",
    "0.999 compares to 0.9 is ",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0, max_tokens=50)
# model = "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/"
model = "../Llama-3.1-8B-Instruct-FP8-Static"
# model = "../Llama-3.1-8B-Instruct-FP8"
# model = "../Qwen3-8B"
# model = "../Llama-3.1-8B-Instruct-FP8-DYNAMIC"

# model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model = "Qwen/Qwen3-30B-A3B"
model = "../Qwen3-30B-A3B-FP8_DYNAMIC"
# model = "../Qwen3-30B-A3B-FP8"
#model = "../Qwen3-30B-A3B-FP8-Static"
model = "../Llama-3.1-8B-Instruct-FP8-Static"
model = "../Qwen3-0.6B-FP8-Static"
# model = "../Qwen3-0.6B-FP8"
model = "../Qwen3-30B-A3B-FP8-Static"
#kwargs = {"tensor_parallel_size": 4, "enforce_eager": True}
kwargs = {"tensor_parallel_size": 2, "enforce_eager": False}
kwargs = {"tensor_parallel_size": 2, "enforce_eager": True}
kwargs = {"tensor_parallel_size": 1, "enforce_eager": False}
kwargs = {"tensor_parallel_size": 1, "enforce_eager": True}
if os.path.basename(model) in ["Qwen3-30B-A3B", "DeepSeek-V2-Lite-Chat"]:
    kwargs["enable_expert_parallel"] = True
elif "Qwen3-30B-A3B" in model:
    kwargs["enable_expert_parallel"] = True
print(kwargs)


# t_vllm.py
import multiprocessing as mp
from vllm import LLM, SamplingParams

def dump_block_ios_old(self, out_dir="./vllm_ios", fmt="pt"):
    import os, torch, numpy as np, uuid
    os.makedirs(out_dir, exist_ok=True)
    ret = []
    for i, rec in enumerate(getattr(self, "__io_cache", [])):
        uid = uuid.uuid4().hex[:8]
        base = f"{i:04d}_{rec['layer'].replace('.', '_')}_{uid}"
        in_p, out_p = os.path.join(out_dir, base+"_in."+fmt), os.path.join(out_dir, base+"_out."+fmt)
        if fmt == "pt":
            torch.save(rec["input"], in_p); torch.save(rec["output"], out_p)
        elif fmt == "npy":
            np.save(in_p, rec["input"].numpy()); np.save(out_p, rec["output"].numpy())
        ret.append({"layer": rec["layer"], "input_path": in_p, "output_path": out_p})
    return ret

def dump_block_ios(self, out_dir="./vllm_ios", fmt="pt", save_original=True):
    """
    保存已收集的前向 IO 数据。
    期望 __io_cache 中的每条记录 rec 至少包含:
        rec["layer"] : str
        rec["hidden_states"] 或 rec["input"]
        rec["router_logits"] (可能不存在)
        rec["output"] 或 rec["outputs"] (单 tensor 或列表/tuple)

    如果之前使用了 pre_hook 还可能包含:
        rec["original_hidden_states"]
        rec["original_router_logits"]

    参数:
        out_dir: 输出目录
        fmt: "pt" 或 "npy"
        save_original: 如果存在 original_* 字段则一并保存
    返回:
        list[dict]: 每条记录的文件路径元数据
    """
    import os, uuid
    import torch
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    records = getattr(self, "__io_cache", [])
    ret = []

    def _save_tensor(t, path):
        if t is None:
            return None
        if fmt == "pt":
            torch.save(t, path)
        elif fmt == "npy":
            np.save(path, t.numpy())
        else:
            raise ValueError(f"Unsupported fmt={fmt}")
        return path

    for i, rec in enumerate(records):
        uid = uuid.uuid4().hex[:8]
        layer_name = rec.get("layer", f"layer_{i}")
        safe_layer = layer_name.replace(".", "_")
        base = f"{i:04d}_{safe_layer}_{uid}"

        # 兼容旧结构：input -> hidden_states
        hidden_states = rec.get("hidden_states", rec.get("input"))
        router_logits = rec.get("router_logits")
        outputs = rec.get("outputs")

        weights = rec.get("weights")

        # 兼容旧结构：如果只有单输出 rec["output"]
        if outputs is None and rec.get("output") is not None:
            outputs = [rec["output"]]
        if outputs is None:
            outputs = []

        meta = {
            "layer": layer_name,
            "hidden_states_shape": tuple(hidden_states.shape) if torch.is_tensor(hidden_states) else None,
            "router_logits_shape": tuple(router_logits.shape) if torch.is_tensor(router_logits) else None,
            "num_outputs": len(outputs),
            "output_shapes": [tuple(o.shape) for o in outputs if torch.is_tensor(o)],
            "files": {}
        }

        # 保存输入 hidden_states
        hs_path = os.path.join(out_dir, f"{base}_hidden_states.{fmt}")
        meta["files"]["hidden_states"] = _save_tensor(hidden_states, hs_path)

        # 保存 router_logits（如果存在）
        if torch.is_tensor(router_logits):
            rl_path = os.path.join(out_dir, f"{base}_router_logits.{fmt}")
            meta["files"]["router_logits"] = _save_tensor(router_logits, rl_path)

        # 保存原始（pre_hook clone）版本
        if save_original:
            orig_hs = rec.get("original_hidden_states")
            orig_rl = rec.get("original_router_logits")
            if torch.is_tensor(orig_hs):
                orig_hs_path = os.path.join(out_dir, f"{base}_original_hidden_states.{fmt}")
                meta["files"]["original_hidden_states"] = _save_tensor(orig_hs, orig_hs_path)
            if torch.is_tensor(orig_rl):
                orig_rl_path = os.path.join(out_dir, f"{base}_original_router_logits.{fmt}")
                meta["files"]["original_router_logits"] = _save_tensor(orig_rl, orig_rl_path)

        # 保存输出
        for j, o in enumerate(outputs):
            if not torch.is_tensor(o):
                continue
            out_path = os.path.join(out_dir, f"{base}_out{j}.{fmt}")
            meta["files"][f"out{j}"] = _save_tensor(o, out_path)

        print("===========================================================================================================")
        print(weights)
        for k in weights:
            if not torch.is_tensor(weights[k]):
                continue
            out_path = os.path.join(out_dir, f"{base}_{k}.{fmt}")
            meta["files"][f"{k}"] = _save_tensor(weights[k], out_path)

        ret.append(meta)

    return ret

def install_block_io_hooks(self, layer_filter_prefix="model.model.layers"):
    import torch, re
    model = self.model_runner.model
    self.__io_cache = []
    self.__io_handles = []

    def make_hook(name):
        def _hook(mod, args, kwargs, output):
            print(f"=====================================================name: {name}")
            print(f"=====================================================inputs: {args}")
            print(f"=====================================================output: {output}")
            print(f"=====================================================kwargs: {kwargs}")

            if len(args) >= 2:
                hidden_states = args[0]
                router_logits = args[1]
            elif len(args) == 1:
                hidden_states = args[0]
                router_logits = kwargs.get("router_logits")
            else:
                hidden_states = kwargs.get("hidden_states")
                router_logits = kwargs.get("router_logits")

            print(f"================================hidden_states: {hidden_states.shape}")
            print(f"================================router_logits: {router_logits}")

            if isinstance(output, (tuple, list)):
                out_tensors = [o for o in output if torch.is_tensor(o)]
                main_out = out_tensors[0] if out_tensors else None
            else:
                main_out = output

            print(f"=============================main_out: {main_out.shape}")
            print(f"=============================main_out: {main_out}")


            weights = {}
            for attr in [
                    "w13_weight", "w2_weight",
                    "w13_scale", "w2_scale",
                    "w13_weight_scale", "w2_weight_scale",
                    "w13_input_scale", "w2_input_scale",
                    "a1_scale", "a2_scale",
                    "weight_scale", "weight_scale_2",
                    "input_scale",
                    ]:
                val = getattr(mod, attr, None)
                print(f"====================attr: {attr}, val: {val}")

                if torch.is_tensor(val):
                    weights[attr] = val.detach().cpu()

            def _to_cpu_detach(t):
                if torch.is_tensor(t):
                    return t.detach().to("cpu")
                if isinstance(t, (list, tuple)):
                    return type(t)(_to_cpu_detach(tt) for tt in t)
                if isinstance(t, dict):
                    return {k: _to_cpu_detach(v) for k, v in t.items()}
                return t

            self.__io_cache.append({
                "layer": name,
                "hidden_states": _to_cpu_detach(hidden_states) if torch.is_tensor(hidden_states) else None,
                "router_logits": _to_cpu_detach(router_logits) if torch.is_tensor(router_logits) else None,
                "output": _to_cpu_detach(main_out) if torch.is_tensor(main_out) else None,
                "weights": weights,
                "raw_output_type": type(output).__name__,
            })
            print("_to_cpu_detach(main_out): ", _to_cpu_detach(main_out))
            # exit()
        return _hook

    for name, module in model.named_modules():
        print(f"====================================name: {name}")
        print(f"====================================module: {module}")
        if name.startswith(layer_filter_prefix):
            # if "down_proj" not in name:
            # if name != "model.model.layers.0._orig_mod.mlp.experts":
            if name != "model.model.layers.0.mlp.experts":
                continue

            sig = inspect.signature(module.forward)
            params = list(sig.parameters.values())
            print(f"params: {params}")
            print(f"sig: {sig}")
            # h = module.register_forward_hook(make_hook(name), with_kwargs=False)
            h = module.register_forward_hook(make_hook(name), with_kwargs=True)
            self.__io_handles.append(h)

    return dict(
        hooked_layers=len(self.__io_handles),
        example_layers=[h.id for h in self.__io_handles[:3]],
    )

def fetch_and_clear_block_ios(self):
    data = getattr(self, "__io_cache", [])
    self.__io_cache = []
    return data

def remove_block_io_hooks(self):
    handles = getattr(self, "__io_handles", [])
    for h in handles:
        h.remove()
    self.__io_handles = []
    return {"removed": len(handles)}

def main():
    llm = LLM(model=model, max_model_len=4096, trust_remote_code=True, **kwargs)


    print(llm.collective_rpc(install_block_io_hooks))

    outs = llm.generate(
        prompts=["Hooks with vLLM"],
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),)

    # io_per_worker = llm.collective_rpc(fetch_and_clear_block_ios)
    # print(io_per_worker)

    print(llm.collective_rpc(dump_block_ios))

    print(llm.collective_rpc(remove_block_io_hooks))
    exit()


    layer_io = io_per_worker[0]  # list of (name, input_tensor_cpu, output_tensor_cpu)

    for name, x, y in layer_io[:3]:
        """
        print(name, tuple(x.shape) if hasattr(x, "shape") else type(x),
              tuple(y.shape) if hasattr(y, "shape") else type(y))
        """
        print(name)
        print(x)
        print(y)

    exit()




    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


    main()
