import os
# from multiprocessing import Process, freeze_support

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

def dump_block_ios(self, out_dir="./vllm_ios", fmt="pt"):
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

def install_block_io_hooks(self, layer_filter_prefix="model.model.layers"):
    import torch, re
    model = self.model_runner.model
    self.__io_cache = []
    self.__io_handles = []

    def make_hook(name):
        def _hook(mod, inputs, output):
            # print(f"=====================================================name: {name}")
            # print(f"=====================================================inputs: {inputs}")
            # print(f"=====================================================inputs: {len(inputs)}")
            x = inputs[0]
            y = output[0] if isinstance(output, (tuple, list)) else output
            # print(f"================x: {x}, {x.shape}")
            # print(f"================y: {y}, {y.shape}")
            def _to_cpu_detach(t):
                if torch.is_tensor(t):
                    return t.detach().to("cpu")
                if isinstance(t, (list, tuple)):
                    return type(t)(_to_cpu_detach(tt) for tt in t)
                if isinstance(t, dict):
                    return {k: _to_cpu_detach(v) for k, v in t.items()}
                return t
            # self.__io_cache.append((name, _to_cpu_detach(x), _to_cpu_detach(y)))
            self.__io_cache.append({"layer": name, "input": _to_cpu_detach(x), "output": _to_cpu_detach(y)})
            # print(self.__io_cache)
            # exit()
        return _hook

    for name, module in model.named_modules():
        print(name)
        print(module)
        if name.startswith(layer_filter_prefix):
            # if "down_proj" not in name:
            if "o_proj" not in name:
                continue
            h = module.register_forward_hook(make_hook(name), with_kwargs=False)
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
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),)

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
