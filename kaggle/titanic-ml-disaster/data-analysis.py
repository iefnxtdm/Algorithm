import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
data_train = pd.read_csv("D:/kaggle/Train.csv")
data_train.info() #查看各个列的信息
data_train.describe() #计算mean, std等


known_age = data_train[data_train.Age.notnull()].as_matrix()
unknown_age = data_train[data_train.Age.isnull()].as_matrix()
print(known_age)
    # y即目标年龄
y = known_age[:, 0]

print(y)
    # X即特征属性值
X = known_age[:, 1:]