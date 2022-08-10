import csv
import emoji
import torch
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


# 将词转换为词向量存储为 字典 方便后续使用
def read_glove_vecs(glove_file):
    word_to_vec_map = {}
    with open(glove_file, 'r',  encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip().split()
            word_to_vec_map[line[0]] = torch.tensor(list(map(float, line[1:])), dtype=torch.float32)
            line = f.readline()
    
    return word_to_vec_map

# 使用 numpy 简单实现 softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ == "__main__":
    w_t_i, i_t_w, w_t_v_m = read_glove_vecs("emo_data/glove.6B.50d.txt")
    print(w_t_i, "\n", i_t_w, "\n", w_t_v_m)
