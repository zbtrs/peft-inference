import torch
import torch.nn as nn
import torch.optim as optim

class SimpleDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(SimpleDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(0, 1)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        x2 = self.linear2(self.linear1(x))
        x = x + x2
        x = self.norm2(x)
        return x.transpose(0, 1)

# 定义模型，包含多个相同的解码器层
class SimpleModel(nn.Module):
    def __init__(self, num_layers, d_model, nhead):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([SimpleDecoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x[:, -1, :])
        return x

num_layers = 4
d_model = 128
nhead = 8
model = SimpleModel(num_layers, d_model, nhead)

inputs = torch.randint(0, 1000, (32, 10))  # (batch_size, seq_len)
labels = torch.randint(0, 10, (32,))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


model.train()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()

overall_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

model.zero_grad()
optimizer.zero_grad()

hidden_states = model.embedding(inputs)
end_states = [hidden_states]
forward_states = [hidden_states.detach().clone().requires_grad_(True)]

for layer in model.layers:
    hidden_states = layer(hidden_states)
    end_states.append(hidden_states)
    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    forward_states.append(hidden_states)

outputs = model.fc(hidden_states[:, -1, :])
loss = criterion(outputs, labels)

loss.backward(retain_graph=True)

for i in range(len(forward_states) - 1, 0, -1):
    grad_outputs = forward_states[i].grad
    end_states[i].backward(grad_outputs,retain_graph=True)

submodule_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

for name in overall_grads:
    if name in submodule_grads and not torch.equal(overall_grads[name], submodule_grads[name]):
        print(f'Gradient mismatch in {name}')
    elif name in submodule_grads:
        print(f'Gradient match in {name}')
    else:
        print(f'Gradient missing in submodules for {name}')
