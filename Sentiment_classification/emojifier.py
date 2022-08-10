"""
该练习来吴恩达老师《序列模型》课程，数据集通过网络在 Github 获得
该练习是为一句话匹配最合适的 emoji 表情
"""


# 导入所需要的包
from emo_utils import read_csv, label_to_emoji, label_to_one_hot, read_glove_vecs
import torch
from torch import nn, optim
import torch.functional as F
import visdom


# 将一个句子所有的词转化为 一个50维的向量，然后求其平均值
def sentence_to_avg(sentence, word_to_vec_map):
    sentence = sentence.lower().split()
    avg = torch.zeros(50)
    for i in sentence:
        avg += word_to_vec_map[i]
    
    return avg / len(sentence)

# 将所有文本转换为一个整体的 tensor 进行输入
def all_sentence_to_tensor(text, word_to_vec_map):
    vec = sentence_to_avg(text[0], word_to_vec_map).reshape(1, -1)
    for x in text[1:]:
        vec = torch.cat((vec, sentence_to_avg(x, word_to_vec_map).reshape(1, -1)), dim=0)  # 第0维的数值会不断增加
    return vec


# 定义模型，实现前向传播
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.line = nn.Linear(50, 5)
        self.softmax = nn.Softmax(dim=1)  # 每行做 softmax
    
    def forward(self, x):
        x = self.line(x)
        return self.softmax(x)


# 计算测试正确率
def cal_acc(net, x, y):
    y_hat = net(x)
    y_hat = y_hat.argmax(dim=1)
    return torch.eq(y_hat, y).sum().item() / y.shape[0]


if __name__ == "__main__":
    # 加载数据集
    x_train, y_train = read_csv("./emo_data/train_emoji.csv")
    x_test, y_test = read_csv("./emo_data/tesss.csv")

    # 加载预训练的 50维GloVe 模型
    word_to_vec_map = read_glove_vecs("emo_data/glove.6B.50d.txt")

    # 将训练集文本转为一个 张量，标签也转化为对应的 张量
    x_train = all_sentence_to_tensor(x_train, word_to_vec_map)  # x_train.shape (132, 50)
    y_train = torch.from_numpy(label_to_one_hot(y_train, 5))
    x_test = all_sentence_to_tensor(x_test, word_to_vec_map)
    y_test = torch.from_numpy(y_test)
    
    # 可视化一些参数
    vis = visdom.Visdom()
    vis.line([0.], [0.], win="train_loss", opts=dict(title="train loss"))  # 监听训练损失
    vis.line([0.], [0.], win="test_acc", opts=dict(title="test acc"))

    # 开始训练
    epochs = 400
    lr = 0.01
    net = Net()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    print("train start...")
    for epoch in range(epochs):
        y_hat = net(x_train)
        ls = loss(y_hat, y_train)
        l = ls.item()

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        acc = cal_acc(net, x_test, y_test)
        vis.line([l], [epoch], win="train_loss", update="append")
        vis.line([acc], [epoch], win="test_acc", update="append")
        # print("epoch {} ---> loss: {:5f} ---> acc: {:5f}".format(epoch+1, l, acc))
    
    print("train over...")

    tips = """
    -----------------------------------------------------------------
    以下是一个测试程序，你可以自己输入一句英文句子，模型可以给出相应的 emoji 表情
    例句：I want a big meal ---> 🍴
         You love me ---> ❤️ 
         You very happy ---> 😀
    输入 0 退出
    -----------------------------------------------------------------
    """
    print(tips)
    sentence = input("请输入一个英文句子：")
    while True:
        if sentence == '0':
            break
        x = sentence_to_avg(sentence, word_to_vec_map).reshape(1, -1)
        y = net(x).argmax(dim=1).item()
        print("你可以搭配：", label_to_emoji(y))
        sentence = input("请输入一个英文句子：")
