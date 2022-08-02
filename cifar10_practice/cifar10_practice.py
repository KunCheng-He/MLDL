import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from lenet5 import Lenet5


# 加载数据集
def load_dataset(batch_size):
    train_loader = DataLoader(
        datasets.CIFAR10(
            "../CIFAR10", train=True, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ]), download=True
        ), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.CIFAR10(
            "../CIFAR10", train=False, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ]), download=True
        ), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# 训练函数
def train(net, train_loader, test_loader, epochs, lr, device):
    loss = nn.CrossEntropyLoss().to(device)
    optimize = optim.Adam(net.parameters(), lr=lr)

    net = net.to(device)
    for epoch in range(epochs):
        ls = 0.0
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            ls += l.item()

            optimize.zero_grad()
            l.backward()
            optimize.step()
        
        # 计算该 epoch 训练完后目前模型的正确率
        net.eval()
        with torch.no_grad():
            all_item = 0.0
            acc_item = 0.0
            for x, y in test_loader:
                all_item += y.shape[0]
                x, y = x.to(device), y.to(device)
                y_hat = net(x)
                y_hat = y_hat.argmax(dim=1)
                acc_item += torch.eq(y, y_hat).sum().item()
        
        print("epoch {} loss: {:.5f} acc: {:.5f}".format(epoch+1, ls, acc_item / all_item))


if __name__ == "__main__":
    # 设置一些参数
    batch_size = 32
    epochs = 30
    lr = 1e-3
    device = torch.device("cuda")

    # 加载数据集
    train_loader, test_loader = load_dataset(batch_size)

    # 开始训练
    train(Lenet5(), train_loader, test_loader, epochs, lr, device)
