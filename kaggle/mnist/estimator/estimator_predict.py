from __future__ import division, print_function, absolute_import  

import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
import cnn_estimator

data_test = pd.read_csv('D:/kaggle/mnist/test.csv')
datax = data_test.as_matrix()
datax = np.multiply(datax, 1.0/255)
datax = datax.astype(np.float32)

model = tf.estimator.Estimator(model_fn=cnn_estimator.model_fn, model_dir='D:/kaggle/mnist2/model1/')  
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': datax}, batch_size=128, shuffle=False)
y = model.predict(input_fn)
pd.DataFrame(y).to_csv('D:/kaggle/mnist2/predict.csv')