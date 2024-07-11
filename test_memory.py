import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_memory_usage():
    allocated_memory = torch.cuda.max_memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Allocated memory: {allocated_memory / 1024**3:.2f} GB")
    print(f"Reserved memory: {reserved_memory / 1024**3:.2f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "/data/aigc/llama2"

# 加载前显存使用情况
print("Before loading the model:")
print_memory_usage()

# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(device)
for name, param in model.named_parameters():
    print(f"Parameter {name} dtype: {param.dtype}")
    # break  # 只打印第一个参数的数据类型

print_memory_usage()
