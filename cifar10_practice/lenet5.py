import torch
from torch import nn


class Lenet5(nn.Module):
    def __init__(self) -> None:
        super(Lenet5, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # 通道：3 --> 6  形状：32-5+1=28
            nn.AvgPool2d(kernel_size=2, stride=2),  # 通道：6  形状：(28-2+2)/2=14
            nn.Conv2d(6, 16, kernel_size=5),  # 通道：6 ---> 16  形状：14-5+1=10
            nn.AvgPool2d(kernel_size=2, stride=2),  # 通道：16  形状：(10-2+2)/2=5
            nn.Flatten(),  # 拉平每个样本变为 16*5*5=400
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    # 测试网络
    net = Lenet5()
    x = torch.rand(2, 3, 32, 32)
    y = net(x)
    print(y.shape)
