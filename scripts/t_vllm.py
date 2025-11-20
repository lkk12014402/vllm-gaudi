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
model = "../Qwen3-30B-A3B-FP8"
model = "../Qwen3-30B-A3B-FP8-Static"
# model = "../Llama-3.1-8B-Instruct-FP8-Static"
# model = "../Qwen3-0.6B-FP8-Static"
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


# t_vllm.py
import multiprocessing as mp
from vllm import LLM, SamplingParams

def main():
    llm = LLM(model=model, max_model_len=4096, trust_remote_code=True, **kwargs)

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
