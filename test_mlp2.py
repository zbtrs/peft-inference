import gc
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from pydantic.v1 import NoneStr
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.image_transforms import normalize
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(weight)
        return input.mm(weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        return grad_input, None

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return CustomLinearFunction.apply(input, self.weight)

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(LlamaMLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        torch.manual_seed(0)
        self.up_proj = CustomLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = CustomLinear(self.intermediate_size, self.hidden_size, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        init.kaiming_uniform_(self.up_proj.weight,a=math.sqrt(5))
        init.kaiming_uniform_(self.down_proj.weight,a=math.sqrt(5))


    def forward(self, x):
        # with torch.no_grad():
        size1 = torch.cuda.memory_allocated()
        print(f"Initial memory: {size1 / (1024**3):.4f} GB")

        self.up_proj1 = self.up_proj(x)
        print(f"up_proj1 size: {self.up_proj1.nelement() * self.up_proj1.element_size() / (1024**3):.4f} GB")
        size2 = torch.cuda.memory_allocated()
        print(f"After up_proj1: {(size2 - size1) / (1024**3):.4f} GB")

        x.detach_()
        del x
        x = None
        torch.cuda.empty_cache()
        size3 = torch.cuda.memory_allocated()
        print(f"After clearing x: {(size3 - size2) / (1024**3):.4f} GB")

        self.down_proj1 = self.down_proj(self.up_proj1)
        size4 = torch.cuda.memory_allocated()
        print(f"After down_proj1: {(size4 - size3) / (1024**3):.4f} GB")

        self.up_proj1.detach_()
        del self.up_proj1
        self.up_proj1 = None
        torch.cuda.empty_cache()
        size5 = torch.cuda.memory_allocated()
        print(f"After clearing up_proj1: {(size5 - size4) / (1024**3):.4f} GB")


        self.up_proj2 = self.up_proj(self.down_proj1)
        size6 = torch.cuda.memory_allocated()
        print(f"After act_fn: {(size6 - size5) / (1024**3):.4f} GB")

        self.down_proj1.detach_()
        del self.down_proj1
        self.down_proj1 = None
        torch.cuda.empty_cache()
        size7 = torch.cuda.memory_allocated()
        print(f"After clearing down_proj1: {(size7 - size6) / (1024**3):.4f} GB")

        self.down_proj2 = self.down_proj(self.up_proj2)

        return self.down_proj2

hidden_size = 2000
intermediate_size = 5000
model = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
for param in model.parameters():
    param.requires_grad = False
model.to("cuda")
criterion = nn.MSELoss()
input_tensor = torch.randn(hidden_size, hidden_size).to("cuda")
input_tensor.requires_grad = True
residual = input_tensor
target_tensor = torch.randn(hidden_size, hidden_size).to("cuda")
output = model(input_tensor)
output = output + residual
loss = criterion(output, target_tensor)
loss.backward()

print(loss)