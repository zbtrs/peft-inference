import torch
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


class saved_tensors_hooks:
    def __init__(
            self,
            pack_hook: Callable[[torch.Tensor], Any],
            unpack_hook: Callable[[Any], torch.Tensor],
    ):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )
        return self

    def __exit__(self, *args: object):
        torch._C._autograd._pop_saved_tensors_default_hooks()


class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory=True, device_type="cuda"):
        device_module = getattr(torch, device_type, torch.cuda)
        self.tensor_dict = {}

        def create_pack_to_cpu():
            def pack_to_cpu(tensor):

                if not pin_memory:
                    cpu_tensor = tensor.cpu()
                else:
                    cpu_tensor = torch.empty(
                        tensor.size(),
                        dtype=tensor.dtype,
                        layout=tensor.layout,
                        pin_memory=(device_module.is_available() and not tensor.is_sparse),
                    )
                    cpu_tensor.copy_(tensor, non_blocking=True)

                tensor_id = id(cpu_tensor)
                print(f"pack tensor id:{tensor_id}")

                self.tensor_dict[tensor_id] = cpu_tensor
                return (tensor.device, cpu_tensor)

            return pack_to_cpu

        def create_unpack_from_cpu():
            def unpack_from_cpu(packed):
                device, cpu_tensor = packed
                tensor_id = id(cpu_tensor)
                print(f"unpack tensor id:{tensor_id}")

                if tensor_id in self.tensor_dict:
                    return self.tensor_dict[tensor_id]
                else:
                    gpu_tensor = cpu_tensor.to(device, non_blocking=pin_memory)
                    self.tensor_dict[tensor_id] = gpu_tensor
                    return gpu_tensor

            return unpack_from_cpu

        super().__init__(create_pack_to_cpu(), create_unpack_from_cpu())

    def prefetch_to_gpu(self):
        for tensor_id in self.tensor_dict:
            cpu_tensor = self.tensor_dict[tensor_id]
            gpu_tensor = cpu_tensor.to('cuda')
            self.tensor_dict[tensor_id] = gpu_tensor


def print_memory_usage(step):
    print(f"{step}:")
    print(f"  Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print()

# 创建较大的张量
a = torch.randn(10000, 10000, requires_grad=True, device="cuda")
b = torch.randn(10000, 10000, requires_grad=True, device="cuda")
c = torch.randn(10000, 10000, requires_grad=True, device="cuda")

# 不使用 save_on_cpu 的情况
print_memory_usage("Before without save_on_cpu")

def f_no_save_on_cpu(a, b, c):
    prod_1 = a * b
    prod_2 = prod_1 * c
    y = prod_2 * a
    return y

y_no_save = f_no_save_on_cpu(a, b, c)

print_memory_usage("After without save_on_cpu")
y_no_save.sum().backward()

print_memory_usage("After backward without save_on_cpu")

# 保存梯度
grad_a_no_save = a.grad.clone()
grad_b_no_save = b.grad.clone()
grad_c_no_save = c.grad.clone()

# 重置梯度
a.grad.zero_()
b.grad.zero_()
c.grad.zero_()

# 使用 save_on_cpu 的情况
print_memory_usage("Before with save_on_cpu")

def f_with_save_on_cpu(a, b, c):
    with save_on_cpu() as sa:
        prod_1 = a * b
        prod_2 = prod_1 * c
    y = prod_2 * a
    return y, sa

y_save, sa = f_with_save_on_cpu(a, b, c)

print_memory_usage("After with save_on_cpu")

sa.prefetch_to_gpu()

print_memory_usage("After prefetch_to_gpu")

y_save.sum().backward()

print_memory_usage("After backward with save_on_cpu")

# 保存梯度
grad_a_save = a.grad.clone()
grad_b_save = b.grad.clone()
grad_c_save = c.grad.clone()

# 对比前向结果和梯度
print("Forward outputs match:", torch.allclose(y_no_save, y_save))
print("Gradient a match:", torch.allclose(grad_a_no_save, grad_a_save))
print("Gradient b match:", torch.allclose(grad_b_no_save, grad_b_save))
print("Gradient c match:", torch.allclose(grad_c_no_save, grad_c_save))