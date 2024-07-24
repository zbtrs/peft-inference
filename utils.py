import torch

def print_memory_usage(print_image):
    print(print_image)
    allocated_memory = torch.cuda.max_memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Allocated memory: {allocated_memory / 1024**3:.2f} GB")
    print(f"Reserved memory: {reserved_memory / 1024**3:.2f} GB")
    torch.cuda.reset_max_memory_allocated()