# 导入需要的包
import torch
import torch.utils.data as data
from torch import nn, optim
import torch.nn.functional as F
import torchvision


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载数据集
def load_data(batch_size):
    train_loader = data.DataLoader(
        torchvision.datasets.MNIST(
            "msist_data", train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ),
        batch_size=batch_size, shuffle=True
    )
    test_loader = data.DataLoader(
        torchvision.datasets.MNIST(
            "msist_data", train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# 将 label 转变为 one_hot 向量
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


# 计算每个epoch测试集的准确度
def cal_acc(net, test_loader):
    total_item = 0.0
    total_correct = 0.0
    for x, y in test_loader:
        total_item += x.size(0)
        x = x.view(x.size(0), -1)
        out = net(x)
        pred = out.argmax(dim=1)
        total_correct += pred.eq(y).sum().item()
    return total_correct / total_item


# 训练
def train(net, train_loader, test_loader, epochs, optimizer):
    print("train start...")
    for epoch in range(epochs):
        ls = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            # x.shape [512, 1, 28, 28]    y.shape [512]
            x = x.view(x.size(0), -1)  # x.shape [512, 784] 相当于 reshape 的用法
            y_hat = net(x)
            y = one_hot(y)
            loss = F.mse_loss(y, y_hat)
            ls += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = cal_acc(net, test_loader)
        print("epoch {} --> loss: {} ---> Acc: {}".format(epoch, ls, acc))


if __name__ == '__main__':
    # 设置一些参数
    batch_size = 512
    epochs = 5
    lr = 0.01

    # 载入数据
    train_loader, test_loader = load_data(batch_size)

    # 开始训练
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    train(net, train_loader, test_loader, epochs, optimizer)
