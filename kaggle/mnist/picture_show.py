import pandas as pd #数据分析
import numpy as np #科学计算
import tensorflow as tf
import matplotlib.pyplot as plt


train_data =  pd.read_csv('D:/mnist/test.csv')
X_train=train_data
del train_data
# 改变维度：第一个参数是图片数量，后三个参数是每个图片的维度
X_train = X_train.values.reshape(-1,28,28,1)
print(X_train.shape)
print("Train Sample:",X_train.shape[0])
# 归一化：将数据进行归一化到0-1 因为图像数据最大是255
X_train=X_train/255.0
# 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
plt.imshow(X_train[0][:,:,0], cmap="Greys")
plt.show()