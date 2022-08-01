# 导入需要的包
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


# 定义模型
class Net(nn.Module):
    def __int__(self):
        super(Net, self).__int__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载数据集
def load_data(batch_size):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "msist_data", train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "msist_data", train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        ),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

# 训练
def train(train_loader, test_loader, epochs):
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            print(x.shape, y.shape)
            break


if __name__ == '__main__':
    # 设置一些参数
    batch_size = 512
    epochs = 3

    # 载入数据
    train_loader, test_loader = load_data(batch_size)

    # 开始训练
    train(train_loader, test_loader, epochs)

