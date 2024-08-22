import torch

# 创建一个大的张量并放到 GPU
x = torch.randn(1024, 1024, 1024, device='cuda')
print(f"Memory allocated before offload: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(f"Memory reserved before offload (including cache): {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

# 将张量卸载到 CPU
x = x.to('cpu')
torch.cuda.synchronize()  # 确保卸载完成
print(f"Memory allocated after offload: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
print(f"Memory reserved after offload (including cache): {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

# 清空缓存
torch.cuda.empty_cache()
print(f"Memory reserved after emptying cache: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
