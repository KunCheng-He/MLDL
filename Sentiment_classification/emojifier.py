"""
该练习来吴恩达老师《序列模型》课程，数据集通过网络在 Github 获得
该练习是为一句话匹配最合适的 emoji 表情
"""


# 导入所需要的包
import numpy as np
from emo_utils import read_csv, label_to_emoji, label_to_one_hot


# 加载数据集
x_train, y_train = read_csv("./emo_data/train_emoji.csv")
x_test, y_test = read_csv("./emo_data/tesss.csv")

print(y_train)
print(label_to_one_hot(y_train, 5))

# 参考：https://www.heywhale.com/mw/project/62f2073654f6ab9809740a5e
