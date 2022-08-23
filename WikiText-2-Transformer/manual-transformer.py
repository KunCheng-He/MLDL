import torch
import math
import copy
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, greedy_decode, SimpleLossCompute  # run_epoch


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
def subsequent_mask(a, b):
    """参数：
    a(int): 当前 query 的序列长度
    b(int): 当前 key 的序列长度
    """
    attn_shape = (1, a, b)

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


# 一个编码器块
class EncodeBlock(nn.Module):
    def __init__(self, dim_word, head_n, d_ff, dropout=0.1, eps=1e-6) -> None:
        """参数：
        dim_word(int): 词嵌入的维度
        head_n(int): 多头的头数
        d_ff(int): 前馈网络中间层的层数
        dropout(float): Dropout层置零的概率
        eps(float): 一个很小的数，防止分母为零
        """
        super().__init__()

        # 多头注意力机制
        self.multihead = MultiHeadedAttention(head_n, dim_word, dropout=dropout)

        # 第一个规范化层
        self.layernorm1 = LayerNorm(dim_word, eps=eps)

        # 前馈网络层
        self.feedforward = FeedForward(dim_word, d_ff, dropout=dropout)

        # 第二个规范化层
        self.layernorm2 = LayerNorm(dim_word, eps=eps)


    def forward(self, x):
        """参数：
        x: (batch_size, seq_len, dim_word)
        """
        query = key = value = x
        x = x + self.layernorm1(self.multihead(query, key, value))
        return x + self.layernorm2(self.feedforward(x))


# 编码器
class Encoder(nn.Module):
    def __init__(self, N, dim_word, head_n, d_ff, dropout=0.1, eps=1e-6) -> None:
        """参数：
        N(int): 有多少个编码器块
        dim_word(int): 词嵌入的维度
        head_n(int): 多头的头数
        d_ff(int): 前馈网络中间层的层数
        dropout(float): Dropout层置零的概率
        eps(float): 一个很小的数，防止分母为零
        """
        super().__init__()

        self.encoder = [EncodeBlock(dim_word, head_n, d_ff, dropout=dropout, eps=eps) for _ in range(N)]


    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

# 一个解码器块
class DecodeBlock(nn.Module):
    def __init__(self, dim_word, head_n, d_ff, dropout=0.1, eps=1e-6) -> None:
        """参数：
        dim_word(int): 词嵌入的维度
        head_n(int): 多头的头数
        d_ff(int): 前馈网络中间层的层数
        dropout(float): Dropout层置零的概率
        eps(float): 一个很小的数，防止分母为零
        """
        super().__init__()
        
        # 第一个多头注意力机制
        self.multihead1 = MultiHeadedAttention(head_n, dim_word, dropout=dropout)

        # 第一个规范化层
        self.layernorm1 = LayerNorm(dim_word, eps=eps)

        # 第二个多头注意力机制
        self.multihead2 = MultiHeadedAttention(head_n, dim_word, dropout=dropout)

        # 第二个规范化层
        self.layernorm2 = LayerNorm(dim_word, eps=eps)

        # 前馈网络层
        self.feedforward = FeedForward(dim_word, d_ff, dropout=dropout)

        # 第三个规范化层
        self.layernorm3 = LayerNorm(dim_word, eps=eps)


    def forward(self, x, memory):
        """参数：
        x: (batch_size, seq_len, dim_word)
        code: 编码器的输出 (batch_size, seq_len, dim_word)
        """
        # 一开始是自注意力机制，还有 mask
        mask = subsequent_mask(x.shape[1], x.shape[1])
        query = key = value = x
        x = x + self.layernorm1(self.multihead1(query, key, value, mask=mask))
        # 第二多头注意力的 key 和 value 来自编码器，query 来自解码器
        query = x
        key = value = memory
        x = x + self.layernorm2(self.multihead2(query, key, value))
        return x + self.layernorm3(self.feedforward(x))


# 解码器
class Decoder(nn.Module):
    def __init__(self, N, dim_word, head_n, d_ff, dropout=0.1, eps=1e-6) -> None:
        """参数：
        N(int): 解码器的个数
        dim_word(int): 词嵌入的维度
        head_n(int): 多头的头数
        d_ff(int): 前馈网络中间层的层数
        dropout(float): Dropout层置零的概率
        eps(float): 一个很小的数，防止分母为零
        """
        super().__init__()

        self.decoder = [DecodeBlock(dim_word, head_n, d_ff, dropout=dropout, eps=eps) for _ in range(N)]


    def forward(self, x, memory):
        """参数：
        x: (batch_size, seq_len, dim_word)
        code: 编码器的输出 (batch_size, seq_len, dim_word)
        """
        for layer in self.decoder:
            x = layer(x, memory)
        return x


# Transformer 最后一个输出部分
class Output(nn.Module):
    def __init__(self, decode_dim, target_dim) -> None:
        """参数：
        decode_dim: 解码器的输出的维度，一般就是 dim_word
        target_dim: 我们分类的一个维度
        """
        super().__init__()
        self.linear = nn.Linear(decode_dim, target_dim)
        self.softmax = nn.Softmax(dim=-1)
    

    def forward(self, x):
        return self.softmax(self.linear(x))


# 组装出最后的 Transformer
class ManualTransformer(nn.Module):
    def __init__(self, dim_word, source_vocab, target_vocab, head_n, d_ff, N_encoder, N_decoder, out_dim, dropout=0.1, max_len=1000, eps=1e-6) -> None:
        """参数：
        dim_word(int): 词嵌入的维度
        source_vocab(int): 输入词的词量大小
        target_vocab(int): 输出词的词量大小
        head_n: 多头的个数
        d_ff: 前馈网络中间的维度
        N_encoder(int): 编码器的个数
        N_decoder(int): 解码器的个数
        out_dim(int): 最后线性分类器输出的维度
        dropout(float): Dropout层置零的概率
        max_len: 位置编码时，指定序列的一个最大长度
        eps(float): 一个很小的数，防止规范化使分母为0
        """
        super().__init__()

        # 输入的词嵌入层
        self.input_embedding = Embedding(dim_word, source_vocab)

        # 位置编码器
        self.pos = PositionalEncoding(dim_word, dropout, max_len=max_len)

        # 编码器
        self.encoder = Encoder(N_encoder, dim_word, head_n, d_ff, dropout=dropout, eps=eps)

        # 解码器词嵌入层
        self.out_embedding = Embedding(dim_word, target_vocab)

        # 解码器
        self.decoder = Decoder(N_decoder, dim_word, head_n, d_ff, dropout=dropout, eps=eps)

        # 输出层
        self.output = Output(dim_word, out_dim)


    def forward(self, source, taget):
        source = self.pos(self.input_embedding(source))  # 将输入进行词嵌入和加入位置编码
        memery = self.encoder(source)  # 编码器的全部输出
        taget = self.pos(self.out_embedding(taget))  # 输出进行词嵌入并加入位置编码
        return self.output(self.decoder(taget, memery))


# 构建数据生成器
def data_generator(max_v, batch, num_batch):
    """参数;
    max_v(int): 随机生成数据的最大值
    batch(int): batch_size
    num_batch(int): batch 的个数
    """
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, max_v, size=(batch, 10)))  # 这个10相当于序列的长度
        data[:, 0] = 1  # 所有序列的第一个位置改为 1，代表起始标志位
        # 因为是 Copy 任务，所以输入和输出相同，使用 yield 返回（迭代器）
        # 使用 Batch 对输出和输出进行对应批次的掩码张量生成
        yield Batch(data, copy.deepcopy(data))


# 重写适用于我自己封装的 transformer 模型的训练方法
def run_epoch(data_iter, model, loss_compute, optimizer=None, out_flag=False):
    "Standard Training and Logging Function"
    start = time.time()
    total_loss = 0
    total_num = 0
    for batch in data_iter:
        out = model.forward(batch.src, batch.trg)
        loss = loss_compute(out, torch.from_numpy(np.eye(out.shape[-1])[batch.trg_y - 1]).float())
        total_loss += loss.item()
        total_num += batch.src.shape[0]
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start
    if out_flag:
        print("Loss sum: {:.6f} times: {:.6f}".format(total_loss / total_num, elapsed))


# 训练函数
def run(model, loss, max_v, batch, train_num_batch, eval_num_batch, optimizer, epochs=10):
    """参数：
    model: 训练的模型
    loss: 定义的损失函数
    max_v(int): 随机生成数据的最大值
    batch(int): batch_size
    train_num_batch(int): 训练时 batch 的个数
    eval_num_batch(int): 验证时 batch 的个数
    epochs: 训练轮次
    """
    for epoch in range(epochs):
        model.train()  # 使用训练模式
        # 使用 run_epoch 包，工具将对模型使用给定的损失函数进行单轮的参数更新，并打印每轮参数更新的结果
        # 我自己封装的 transformer 模型的接口和 run_epoch 工具中 forward 参数不同，所以上方我重写了 run_epoch
        run_epoch(data_generator(max_v, batch, train_num_batch), model, loss, optimizer)

        model.eval()
        run_epoch(data_generator(max_v, batch, eval_num_batch), model, loss, out_flag=True)


if __name__ == "__main__":
    """------------------------------测试整体的 ManualTransformer--------------------------------
    一个 Copy 任务，输入和输出完全相同
    """
    # 定义参数
    dim_word = 300
    source_vocab = 1000
    target_vocab = 1000
    head_n = 6
    d_ff = 600
    N_encoder = 3
    N_decoder = 3
    out_dim = 100
    # 生成的数据范围为 1~100 ， 训练数据有 8*20=160 条，验证数据有 8*5=40 条
    # 每个 batch 里的每个序列长度为 10，在生成数据时指定了
    # 训练 10 个 epoch ，在 run 函数指定了默认值
    max_v = 101
    batch_size = 8
    train_num_batch = 20
    eval_num_batch = 5

    # 定义模型
    transformer = ManualTransformer(dim_word, source_vocab, target_vocab, head_n, d_ff, N_encoder, N_decoder, out_dim)
    # print(transformer)

    # 初始化模型参数
    for _ in transformer.parameters():
        if _.dim() > 1:
            nn.init.xavier_uniform_(_)
    
    # 获取模型优化器
    # optimizer = get_std_opt(transformer)
    optimizer = torch.optim.Adam(transformer.parameters())

    # 获取标签平滑对象
    criterion = LabelSmoothing(size=target_vocab, padding_idx=0, smoothing=0.0)

    # 利用平滑标签计算损失
    # loss = SimpleLossCompute(transformer.parameters(), criterion, optimizer)
    loss = nn.CrossEntropyLoss()

    # 开始训练
    run(transformer, loss, max_v, batch_size, train_num_batch, eval_num_batch, optimizer)
