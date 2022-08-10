import torch
import numpy as np
from torch import nn, optim
from visdom import Visdom
import matplotlib.pyplot as plt


# 一些参数
num_time_steps = 50
input_size = 1
hidden_size = 16
output_size = 1
batch_size = 1
epochs = 6000
lr = 0.01


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size = input_size,  # 输入维度就是 1
            hidden_size = hidden_size,  # 记忆单元用一个 16 维的向量表示
            num_layers = 1,  # 只用一层的 RNN
            batch_first = True  # 输入的 x 的形状为 (batch, seq=50, feature=1)
        )
        
        for p in self.rnn.parameters():  # 对参数初始化
            nn.init.normal_(p, mean=0.0, std=0.001)
        
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)  # out.shape (batch, seq, hidder_size)

        # 变换形状，为了输入线性层，相当于同一个batch里，全部拉平，保留每个词对应的输出构成一个二维的向量，形状为(batch*seq, hidder_size)
        out = out.reshape(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)  # 扩展维度，在指定维度上插入一个维度，变为 (1, batch*seq, out_size)
        return out, hidden_prev


# 对以上网络做一个测试
# net = Net()
# x = torch.randn(2, 50, 1)  # 模拟一个输入(batch=2, seq=50, feature=1)
# hidden_prev = torch.zeros(1, 2, 16)
# out, hidder_prev = net(x, hidden_prev)

net = Net()
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
hidden_prev = torch.zeros(1, batch_size, hidden_size)
viz = Visdom()  # 可视化监测
viz.line([0.], [0.], win="train_loss", opts=dict(title="train loss"))  # 监听训练损失


# 开始训练
print("start training...")
for epoch in range(epochs):
    # 生成数据
    start = np.random.randint(3, size=1)[0]  # 随机生成一个 [0, 3) 范围内的整数
    time_steps = np.linspace(start, start + 10, num_time_steps)  # 在 [start, start+10] 的范围内均匀生成 num_time_steps 个数
    data = np.sin(time_steps)  # 对生成的 num_time_steps 个数求解 sin 值
    data = data.reshape(num_time_steps, 1)
    # 相当于给了 1~49 的值，然后希望出预测 2~50 的结果
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)  # 取 第一个数值到倒数第二个数值的 sin 运算结果
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)  # 取 第二个数值到最后一个数值的 sin 运算结果

    # 开始计算
    out, hidden_prev = net(x, hidden_prev)  # out: [1, batch*seq=1*(50-1)=49, out_size=1]  hidden_prev: [num_layers=1, batch=1, hidden_size=16]
    hidden_prev = hidden_prev.detach()  # 将 requires_grad 属性设置为 False，不对它进行梯度更新

    # 计算损失并更新梯度
    l =  loss(out, y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    # 打印每个样本训练完成后的 loss
    # print("epoch {} loss: {}".format(epoch, l.item()))
    viz.line([l.item()], [epoch], win="train_loss", update="append")


# ------------------- 预测部分 -----------------------

# 生成数据
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(-1, 1)
x = torch.tensor(data[:-1]).float().reshape(1, -1, 1)
y = torch.tensor(data[1:]).float().reshape(1, -1, 1)

# 对新生成的数据进行预测并可视化
net.eval()
with torch.no_grad():
    # 用前一个真实值去预测后一个值，结果很好了
    out, hidden_prev = net(x, hidden_prev)
    out = out.reshape(out.shape[1]).numpy()

    # 只有第一个值为真实值，之后根据自己的预测值进行连续预测
    # out = []
    # x = torch.tensor([data[0][0]]).float().reshape(1, 1, 1)
    # for i in range(49):
    #     x, hidden_prev = net(x, hidden_prev)
    #     out.append(x.item())

    # 可视化结果
    plt.figure(figsize=(20, 8), dpi=80)
    x = range(num_time_steps)
    plt.plot(x, data.reshape(num_time_steps), color="m", linestyle="-.", label="True curve")
    plt.plot(x[1:], out, label="Prediction curve")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
