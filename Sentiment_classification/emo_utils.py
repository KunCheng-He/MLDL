import csv
import emoji
import numpy as np


# emoji 索引所对应的 emoji 代码，可在 https://emojiterra.com/ 进行查询
emoji_dictionary = {
    "0": '\u2764\uFE0F',
    "1": '\u26BE',
    "2": '\U0001f600',
    "3": '\U0001f614',
    "4": '\U0001f374'
}


# 通过 label 转换为对应的 emoji 表情
def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)])

# 读取 csv 文件中的数据，转为 array 返回
def read_csv(filename):
    sentence = []  # 存储句子
    emoji_index = []  # 存储句子所对应的 emoji 表情的 index

    with open(filename) as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            sentence.append(line[0])
            emoji_index.append(line[1])
    
    return np.array(sentence), np.array(emoji_index, dtype=int)


# 将 label 标签从一个 (m,) 转变为一个指定维度的 one-hot 向量，变为 (m, c)
def label_to_one_hot(label, c):
    return np.eye(c)[label]


if __name__ == "__main__":
    x = label_to_emoji(4)
    print(x)
