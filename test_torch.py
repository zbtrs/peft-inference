import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 第一个线性层，将输入的10个特征映射到5个特征
        self.relu = nn.ReLU()  # ReLU激活层
        self.fc2 = nn.Linear(5, 1)  # 第二个线性层，将5个特征映射到1个特征

    def forward(self, x):
        x1 = self.fc1(x)
        x.detach()
        x = None
        x2 = self.relu(x1)
        x1.detach()
        x1 = None
        x3 = self.fc2(x2)
        x2.detach()
        x2 = None
        return x3


# 实例化模型
model = SimpleModel()
model.train()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成一些随机数据
inputs = torch.randn(1, 10)  # 1个样本，10个特征
targets = torch.randn(1, 1)  # 1个样本，1个目标

# 前向传播
outputs = model(inputs)
loss = criterion(outputs, targets)

# 反向传播和优化
optimizer.zero_grad()  # 清空梯度
loss.backward()  # 反向传播计算梯度
optimizer.step()  # 更新参数

# 输出损失值
print(f'Loss: {loss.item()}')
