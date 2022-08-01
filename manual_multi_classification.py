# 导入需要的包
import torch
import torch.nn.functional as F
from torch import nn, optim
from mnist_train import load_data, cal_acc
from visdom import Visdom  # 使用这个包来可视化一些需要监听的曲线，监听前要启动服务 python -m visdom.server

# 一些参数
batch_size = 256
lr = 0.001
epochs = 10

# 下载数据集
train_loader, test_loader = load_data(batch_size)

# 手动实现需要更新的参数
# 输入的一个 barch 形状为 [256, 1, 28, 28]
w1, b1 = torch.randn(784, 200, requires_grad=True), torch.zeros(200, 1, requires_grad=True)
w2, b2 = torch.randn(200, 64, requires_grad=True), torch.zeros(64, 1, requires_grad=True)
w3, b3 = torch.randn(64, 10, requires_grad=True), torch.zeros(10, 1, requires_grad=True)

# 当初始化没有做好时，可能损失一直不会变化，可以使用凯明的初始化
# torch.nn.init.kaiming_normal_(w1)
# torch.nn.init.kaiming_normal_(w2)
# torch.nn.init.kaiming_normal_(w3)


# 定义前向传播的过程
def forward(x):
    x = w1.T @ x.T + b1
    x = F.relu(x)
    x = torch.mm(w2.T, x) + b2
    x = F.relu(x)
    x = w3.T @ x + b3
    return x.T


# 定义优化器与损失函数
optimizer = optim.Adam([w1, b1, w2, b2, w3, b3], lr)
loss = nn.CrossEntropyLoss()

# 使用 visdom 监视训练损失的变化
viz = Visdom()  # 先创建一个实例
# viz.line([0.], [0.], win="train loss", opts=dict(title="train loss"))  # 一个窗口只监听一条曲线
# viz.line([[0.0, 0.0]], [0.], win="train-loss&&acc-rate", opts=dict(title="train-loss&&acc-rate", legend=['loss', 'acc']))  # 一个窗口两条曲线

# 开始训练
print("start training...")
flag = True
for epoch in range(epochs):
    ls = 0.0
    for x, y in train_loader:
        x = x.reshape(x.shape[0], -1)
        y_hat = forward(x)

        # 通过 visdom 查看最后一个 epoch 时 第一个 batch 的原始图像与预测结果
        if flag and epoch == epochs - 1:
            viz.images(x.reshape(-1, 1, 28, 28), win='x')
            viz.text(str(y_hat.argmax(dim=1).detach().cpu().numpy()), win='y_hat', opts=dict(title='y_hat'))
            flag = False

        l = loss(y_hat, y)
        ls += l.item()

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    acc = cal_acc(forward, test_loader)
    # viz.line([ls], [epoch], win="train loss", update="append")  # 一个窗口只监听一条曲线
    # viz.line([[ls, acc]], [epoch], win="train-loss&&acc-rate", update="append")  # 一个窗口两条曲线
    print("epoch {} loss: {} Acc: {}".format(epoch, ls, acc))
