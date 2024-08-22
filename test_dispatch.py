import torch

class PyTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(),kwargs=None):
        print(f"PyTensor into {func}")
        

x = torch.randn(8)
# y1 = PyTensor(x)
y1 = x
y2 = torch.add(y1,x)
print(y2)
y3 = torch.add(y2,y1)