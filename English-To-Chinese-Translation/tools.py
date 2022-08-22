import torch
import math
import copy
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 构建词嵌入
class Embedding(nn.Module):
    def __init__(self, dim_word, vocab) -> None:
        """参数：
        dim_word(int): 词嵌入的维度
        vocab(int): 词表的大小
        """
        super(Embedding, self).__init__()
        self.word_to_vec = nn.Embedding(vocab, dim_word)
        self.dim_word = dim_word
    

    def forward(self, x):
        return self.word_to_vec(x) * math.sqrt(self.dim_word)  # 乘后面这一项进行缩放


# 构建位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, dim_word, dropout, max_len=5000) -> None:
        """参数：
        dim_word(int): 词嵌入的维度（位置编码要和词嵌入的维度相同）
        droport(float): Dropout层置零的概率
        max_len(int): 句子的最大长度
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # 初始化一个位置编码矩阵
        pe = torch.zeros(max_len, dim_word)

        # 初始化一个绝对位置矩阵
        position = torch.arange(0, max_len,).unsqueeze(1)  # 生成矩阵后在第一维添加一个维度（shape: max_len*1）

        # 初始化一个变化矩阵（要乘上后面那一项位置编码才平滑）
        div_term = torch.exp(torch.arange(0, dim_word, 2) * -(math.log(10000.0) / dim_word))  # 为了控制位置编码的每一维三角函数的周期不同

        # 位置编码矩阵下标偶数项用sin赋值，奇数项用cos赋值
        pe[:, 0::2] = torch.sin(position * div_term)  # 里面经过相乘得到 max_len*(dimword/2) 形状的矩阵
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码扩充一个维度，因为词嵌入是一个三维的张量
        pe = pe.unsqueeze(0)

        # 将位置矩阵注册为模型的 buffer
        # buffer 参数不用优化
        # 参数可随模型保存，之后可直接导入
        self.register_buffer("pe", pe)

    
    def forward(self, x):
        """参数：
        x: 词嵌入之后的表示

        注：之前生成了一个很长的位置编码矩阵（max_len），实际的句子可能没有那么长，所以矩阵需要截取为当前句子的长度
        """
        x = x + self.pe[:, :x.shape[1]]
        x = x.detach()  # 从计算图中脱离出来，不计算梯度
        return self.dropout(x)


# 绘制位置编码的代码
def matplot_post_encoding():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe(torch.zeros(1, 100, 20))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d"%p for p in [4, 5, 6, 7]])
    plt.show()


# 构建一个掩码张量
def subsequent_mask(size):
    """参数：
    size(int): 掩码张量的后两个维度
    """
    attn_shape = (1, size, size)

    # 全一矩阵通过 np.triu() 形成上三角矩阵
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 翻转为下三角矩阵
    return torch.from_numpy(1 - mask)


# 注意力分数
def attention(query, key, value, mask=None, dropout=None):
    """参数：
    query, key, value: 注意力的三个输入张量，形状为 （batch_size, seq_long, dim_word）
    mask: 掩码张量
    dropout: 传入的 Dropout 实例化对象
    """
    # 拿到 dim_word 词嵌入的维度
    dim_word = query.shape[-1]

    # 将 query 和 key 进行矩阵乘法，然后除以缩放系数（最后两个维度做矩阵乘法，key后两个维度需要转置）
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_word)

    # 如果使用掩码，将为 0 的部分的值设置为一个很小的数
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    
    # 最后将这个分数做一个 Softmax
    p_attn = F.softmax(score, dim=-1)

    # 是否使用 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # 最后返回注意力分数与 value 的乘积， 并返回我们的注意力分数
    return torch.matmul(p_attn, value), p_attn


# 模型层的深度拷贝（多头注意力使用多个线性层）
def deep_clone(model, n):
    """参数：
    model: 需要拷贝的模型层
    n(int): 需要拷贝几份
    """
    # 将拷贝好的层放到 Pytorch 专门的 List 中
    return nn.ModuleList([copy.deepcopy(model) for _ in range(n)])


# 多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, head_n, dim_word, dropout=0.1) -> None:
        """参数：
        head_n(int): 多头的头数
        dim_word(int): 词嵌入的维度
        droport(float): Dropout层置零的概率
        """
        super(MultiHeadedAttention, self).__init__()

        # 检查：多头的数量一定要可以整除词嵌入的维度
        assert dim_word % head_n == 0

        # 获取每个头 词向量的维度
        self.head_dim_word = dim_word // head_n
        
        self.head_n = head_n
        self.dim_word = dim_word

        # 获取 4 个线性层
        self.linears = deep_clone(nn.Linear(dim_word, dim_word), 4)

        # 初始化注意力张量
        self.attn = None

        # 初始化一个 dropout 对象
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, query, key, value, mask=None):
        # 如果使用 mask 需要对维度进行扩充
        if mask is not None:
            mask = mask.unsqueeze(1)  # 代表多头中的第 n 个头
        
        # 获取 batch_size 大小
        batch_size = query.shape[0]

        # 通过 zip 将模型与输入连接起来
        # 再利用 view 与 transpose 进行维度的变换，最后输出
        # 理解：model 每次取出一个线性层，x 每次取出一个元素，因为后面只有三个元素，所以第四个线性层取不到
        # 相当于 query 经过第一个 线性层，key 经过第二个线性层，value 经过第三个线性层
        # 经过线性层变化后，张量形状并没有变，还是 (batch_size, seq_len, dim_word)
        # 线性层后经过 reshape ，理解为句子长度 seq_len 没变，但是把 dim_word 等分成了 多头 n 份
        # 之后对 （1， 2） 维进行转置，理解为第一个头拿到了句子中每个句子的前 n 个特征
        # 最后的形状 (batch_size, 多头数, seq_len, 每个头拿到的特征数)
        query, key, value = [model(x).reshape(batch_size, -1, self.head_n, self.head_dim_word).transpose(1, 2) for model, x in zip(self.linears, (query, key, value))]

        # 将每个头的输出传入注意力层（最后两个维度进行计算）
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 计算完注意力后开始恢复维度（每一节特征经过不同的头输出出来，再拼接起来）
        x = x.transpose(1, 2).reshape(batch_size, -1, self.dim_word)

        # 拼接完成后是(batch_size, seq_len, dim_word)，再经过最后一个线性层
        return self.linears[-1](x)


# 前馈全连接网络
class FeedForward(nn.Module):
    def __init__(self, dim_word, d_ff, dropout=0.1) -> None:
        """参数：
        dim_word(int): 词嵌入的维度
        d_ff(int): 全连接网络中间层的维度
        droport(float): Dropout层置零的概率
        """
        super(FeedForward, self).__init__()
        self.line1 = nn.Linear(dim_word, d_ff)
        self.line2 = nn.Linear(d_ff, dim_word)
        self.dropout = nn.Dropout(p=dropout)
    

    def forward(self, x):
        # 线性层1 ---> ReLu ---> Dropout ---> 线性层2
        return self.line2(self.dropout(F.relu(self.line1(x))))


# LayerNorm 规范化层
class LayerNorm(nn.Module):
    def __init__(self, dim_word, eps=1e-6) -> None:
        """参数：
        dim_word(int): 词嵌入的维度
        eps(float): 一个很小的数，防止除0
        """
        super(LayerNorm, self).__init__()
        # 申明两个参数，并加入模型参数中进行训练
        self.a = nn.Parameter(torch.ones(dim_word))
        self.b = nn.Parameter(torch.zeros(dim_word))
        self.eps = eps
    

    def forward(self, x):
        # 最后一个维度是词嵌入的维度，相当于求每个词词嵌入的均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 减均值，除方差，参数 a 全一，参数 b 全零，像线性层，使这个过程可以学习
        return self.a * (x - mean) / (std + self.eps) + self.b


if __name__ == "__main__":
    # 定义一些参数
    dim_word = 30  # 词嵌入维度
    vocab = 1000  # 最多有1000个词
    head_n = 6  # 6 个头，这样每个头分到 5 个维度的特征
    
    # 假设我现在有一个输入，两个句子，每个句子4个词，每个词用一个数字表示
    x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # x.shape (2, 4)
    # 做词嵌入
    emb = Embedding(dim_word, vocab)
    x = emb(x)  # 做完词嵌入后 x.shape (2, 4, dim_word=30)
    # 加入位置信息
    pos = PositionalEncoding(dim_word, 0.1, max_len=100)
    x = pos(x)  # 加入位置信息后 x.shape (2, 4, 30)
    # 生成一个掩码向量
    mask = subsequent_mask(4)  # 这里传入的参数 size 为序列长度

    # 计算自注意力
    # query = key = value = x
    # atte, p_atte = attention(query, key, value, mask=mask)
    # print(atte, p_atte)
    # print(atte.shape, p_atte.shape)

    # 计算多头注意力
    query = key = value = x
    multi_head = MultiHeadedAttention(head_n, dim_word)
    y = multi_head(query, key, value, mask)
    print("多头注意力的输出： ", y.shape)

    # 注意力层输出后经过前馈全连接层
    feed_layer = FeedForward(dim_word, 90)
    y = feed_layer(y)
    print("前馈全连接层： ", y.shape)

    # 前馈全连接出来经过规范化层
    layer_norm = LayerNorm(dim_word)
    y = layer_norm(y)
    print("layer norm: ", y.shape)

