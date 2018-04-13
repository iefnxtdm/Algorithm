import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

data_train = pd.read_csv('D:/kaggle/mnist/train.csv')
data_train.head(5)
np = data_train.as_matrix()
y = np[:,0] #结果
pixel = np[:,1:]
#pixel[0].size 784 nparray 