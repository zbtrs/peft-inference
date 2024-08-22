import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
from torch.cuda import nvtx

class DeepModel(nn.Module):
    def __init__(self, layer_num=10, layer_size=4096):
        super(DeepModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_size, layer_size) for _ in range(layer_num)])
        self.relu = nn.ReLU()

    def forward(self, x, offload=False):
        if offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                prev_tensor = x
                for idx, layer in enumerate(self.layers):
                    nvtx.range_push(f"Layer {idx} - GPU Computation")
                    x = self.relu(layer(prev_tensor))  # GPU 上的计算
                    nvtx.range_pop()  # 结束 GPU 计算的标记
                    prev_tensor = x
            return x
        else:
            prev_tensor = x
            for idx, layer in enumerate(self.layers):
                nvtx.range_push(f"Layer {idx} - GPU Computation")
                x = self.relu(layer(prev_tensor))  # GPU 上的计算
                nvtx.range_pop()  # 结束 GPU 计算的标记
                
                prev_tensor = x

            return x


def test_model(offload=False):
    # 清空缓存以确保测试的准确性
    torch.cuda.empty_cache()

    # 定义模型和输入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepModel().to(device)
    model.train()

    # 创建较大的输入数据
    x = torch.randn(1024, 4096).to(device)
    target = torch.randn(1024, 4096).to(device)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 查看开始时的显存使用
    torch.cuda.reset_peak_memory_stats()
    print(f"Memory allocated before forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

    # 前向传播
    output = model(x, offload=offload)

    # 查看前向传播后的显存使用
    print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Max memory allocated during forward pass: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")

    # 清除 GPU 上的显存
    torch.cuda.empty_cache()
    
    loss = criterion(output, target)
    # 反向传播
    nvtx.range_push("Backward Pass")
    loss.backward()
    nvtx.range_pop()

    # 清除 GPU 上的显存
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\nTesting with offloading:")
    test_model(offload=True)
