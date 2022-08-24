"""
该 demo 中我们使用的 WikiText-2 数据集来训练模型
这个数据集 600 篇训练文章，60篇验证文章，60篇测试文章
train tokens: 2086708  valid tokens: 218177  test tokens: 246212
Vocab: 33278  OoV(正常英文词汇不在该数据集中的比例): 2.6%
"""


# 导入时间工具包
import time

# 导入torch相关的包
import torch
from torch import nn, optim

# 导入文本处理的工具包
import torchtext

# 导入英文分词的工具
from torchtext.data.utils import get_tokenizer

# 导入已经构建好的 Transformer 模型
from pyitcast.transformer import TransformerModel


# 创建预料域，有四个参数
# tokenize: 使用 get_tokenizer("basic_english") 获得一个分割器对象
# init_token: 给文本一个开始标识符  eos_token: 终止标识符  lower: 存放文本全部转为小写
TEXT = torchtext.legacy.data.Field(
    tokenize=get_tokenizer("basic_english"),
    init_token='<sos>', eos_token='<eos>', lower=True
)


# 导入 WikiText-2 数据集，进行切分，并对这些文本施加刚刚创建的预料域
train_txt, val_txt, test_txt = torchtext.legacy.datasets.WikiText2.splits(TEXT, root="wikitext-2")

# 可通过如下代码查看
# train_txt.examples[0].text[:10]

# 将训练数据构建一个 vocab 对象，统计不重复词汇总数
TEXT.build_vocab(train_txt)

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 构建批次数据的函数
def batchify(data, batch_size):
    """  data: 之前得到的数据(train_txt, val_text, test_txt)
         batch_size: 批次大小
    """
    # 将句子中的单词映射为连续的数字，转换出来 data.shape (seq_len, 1)
    # data.examples[0].text 拿到传入 data 的所有 token
    data = TEXT.numericalize([data.examples[0].text])

    # 看需要用多少个 batch_size 才能遍历完所有的数据
    nbatch = data.shape[0] // batch_size

    # 利用 narrow 方法将不够一个 batch_size 的数据切割掉
    # 第一个参数代表切割维度，二、三个参数代表切割起始位置和终止位置
    data = data.narrow(0, 0, nbatch * batch_size)

    # 将形状变为 (nbatch, batch_size)  contiguous() 在内存上存放在一起，方便转移到运行的设备上
    data = data.reshape(batch_size, -1).T.contiguous()
    return data.to(device)


# 设置训练、验证和测试数据集的批次大小
train_batch_size = 20
eval_batch_size = 10

# 获取训练、验证和测试数据
train_data = batchify(train_txt, train_batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# 设置句子最大长度
bptt = 35


# 获取 batch
def get_batch(source, i):
    """  source: 代表数据
         i: 代表批次数
    """
    # 确定句子长度
    seq_len = min(bptt, len(source) - 1 - i)

    # 首先得到输入数据
    data = source[i : i + seq_len]
    # 目标数据
    target = source[i + 1: i + seq_len + 1].reshape(-1)
    # 相当于给定一个序列，预测后一个序列长啥样
    return data, target



ntokens = len(TEXT.vocab.stoi)  # 获取不重复词汇的总数
ninp = 200  # 词嵌入维度的大小
nhid = 200  # 前馈全连接层的节点数
nlayers = 2  # 编码器层的梳理
nhead = 2  # 多头注意力机制的头数
dropout = 0.2  # Drop层置零的比例

# 获取模型
model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers,dropout).to(device)

criterion = nn.CrossEntropyLoss()  # 损失函数
lr = 5.0  # 学习率
optimizer = optim.Adam(model.parameters(), lr=lr)  # 优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)  # 定义学习率的调整方法


# 训练代码
def train(epoch):
    model.train()
    total_loss = 0.0
    temp_loss = 0.0
    log_size = 200
    total_time = 0.0
    start_time = time.time()  # 获取当前时间
    # 开始遍历批次数据
    for batch, i in enumerate(range(0, train_data.shape[0], bptt)):
        data, target = get_batch(train_data, i)
        # 编码器的结构正常，但是解码器貌似只是一个线性层
        # 输出形状 (bptt, train_batch_size, ninp)
        output = model(data)
        loss = criterion(output.reshape(-1, ntokens), target)
        total_loss += loss.item()
        temp_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        # 梯度规范化，防止出现梯度消失或爆炸
        # 参数只能在 正负 0.5 之间，否者给规范到边界值
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)  
        optimizer.step()
        
        # 每 200 个 batch 打印一次信息
        if batch % log_size == 0 and batch > 0:
            elapsed = time.time() - start_time
            total_time += elapsed
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | loss {:5.4f} | ms/batch {:5.4f} |".format(
                    epoch, batch, train_data.shape[0], scheduler.get_last_lr()[0], temp_loss / log_size, elapsed
                )
            )
            temp_loss = 0.0  # 完成后零时损失清零
            start_time = time.time()  # 对之后 200 个 batch 重新计时
    
    return total_loss, total_time


# 模型评估代码
def evaluate(eval_model, data_source):
    """  eval_model: 当前评估的模型
         data_source: 所用的数据源
    """
    start_time = time.time()
    eval_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, data_source.shape[0] - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            total_loss += criterion(output.reshape(-1, ntokens), targets).item()
    return total_loss, time.time() - start_time



# 首先初始化一个最佳验证损失
best_val_loss = float("inf")  # 一开始设置为无穷大

# 定义训练 epochs
epochs = 10

# 定义最佳模型变量
best_model = None

# 开始训练和验证
for epoch in range(1, 11):
    # 训练
    train_loss, train_elapsed = train(epoch)

    # 验证
    val_loss, val_elapsed = evaluate(model, val_data)

    print("-"*50)
    print(
        "| epoch {:3d} | train loss: {:5.4f} | train time: {:5.4f} | val loss: {:5.4f} | val time: {:5.4f} |".format(
            epoch, train_loss, train_elapsed, val_loss, val_elapsed
        )
    )
    print("-"*50)


    # 取损失值最小的为最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
    
    # 每一轮都做学习率的调整
    scheduler.step()

# 模型训练完成直接测试
test_loss, test_elapsed = evaluate(best_model, test_data)
print("-"*50)
print("test loss: {:5.4f}  test time: {:5.4f}".format(test_loss, test_elapsed))
print("-"*50)
