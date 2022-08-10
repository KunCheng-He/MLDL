from torch import nn, optim
from emo_utils import read_csv, read_glove_vecs, label_to_one_hot, label_to_emoji
import torch
import visdom


# 一个句子转为 tensor
def an_sentence_to_tensor(sentence, word_to_vec_map):
    sentence = sentence.lower().split()
    vec = word_to_vec_map[sentence[0]].reshape(1, -1)
    for i in sentence[1:]:
        vec = torch.cat((vec, word_to_vec_map[i].reshape(1, -1)), dim=0)
    return vec.unsqueeze(dim=1)


# 所有的文本转为 tensor，因为每个句子不一样长，所以这里我用列表进行存储
def all_text_to_tensor(text, word_to_vec_map):
    tensor_list = []
    for i in text:
        tensor_list.append(an_sentence_to_tensor(i, word_to_vec_map))
    return tensor_list


# 计算模型正确率
def cal_acc(net, x, y):
    acc_num = 0
    for i, j in zip(x, y):
        y_hat = net(i).argmax(dim=1).item()
        if y_hat == j.item():
            acc_num += 1
    return acc_num / len(x)


# 定义网络
class Lstm(nn.Module):
    def __init__(self) -> None:
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(50, 30)
        self.line = nn.Linear(30, 5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out, (ht, ct) = self.lstm(x)  # 默认的两个参数为 0
        out = out.reshape(out.shape[0], 30)
        out = out.sum(dim=0, keepdim=True)
        out = self.softmax(self.line(out))
        return out


if __name__ == "__main__":
    # 加载数据集
    x_train, y_train = read_csv("./emo_data/train_emoji.csv")
    x_test, y_test = read_csv("./emo_data/tesss.csv")

    # 加载预训练的 50维GloVe 模型
    word_to_vec_map = read_glove_vecs("emo_data/glove.6B.50d.txt")

    # 将输入和标签转换为所需要形状的 张量
    x_train = all_text_to_tensor(x_train, word_to_vec_map)
    y_train = torch.from_numpy(label_to_one_hot(y_train, 5))
    x_test = all_text_to_tensor(x_test, word_to_vec_map)
    y_test = torch.from_numpy(y_test)

    # 可视化一些参数
    vis = visdom.Visdom()
    vis.line([0.], [0.], win="train_loss", opts=dict(title="train loss"))  # 监听训练损失
    vis.line([0.], [0.], win="test_acc", opts=dict(title="test acc"))

    # 开始训练
    epochs = 60
    lr = 0.01
    net = Lstm()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    print("train start...")
    for epoch in range(epochs):
        l = 0.0
        net.train()
        for x, y in zip(x_train, y_train):
            y_hat = net(x)  # x.shape: (seq, batch=1, feature=50)
            ls = loss(y_hat, y.reshape(1, -1))
            l += ls.item()

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        
        net.eval()
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
        x = an_sentence_to_tensor(sentence, word_to_vec_map)
        y = net(x).argmax(dim=1).item()
        print("你可以搭配：", label_to_emoji(y))
        sentence = input("请输入一个英文句子：")
