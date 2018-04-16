import pandas as pd #数据分析
import numpy as np #科学计算
import tensorflow as tf
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import cnn_train
MODEL_SAVE_PATH = "D:/kaggle/mnist2/"
MODEL_NAME = "model.ckpt"
def predict(data_test):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [100, \
            cnn_train.IMAGE_SIZE, cnn_train.IMAGE_SIZE, cnn_train.NUM_CHANNELS], name = 'x-input')
        #y_ = tf.placeholder(tf.float32, [None, cnn_train.OUTPUT_NODE], name= 'y-input')
        y = cnn_train.inference(x, False, None)

        data_np = data_test.as_matrix()
        xs = np.reshape(data_np, (-1, cnn_train.IMAGE_SIZE,cnn_train.IMAGE_SIZE, cnn_train.NUM_CHANNELS))
        print(xs.shape)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                res = []
                for i in range(0, xs.shape[0], 100):
                    x_tmp = xs[i:i+100,:]
                    #print(x_tmp.shape)
                    x_i = np.reshape(x_tmp, (100, cnn_train.IMAGE_SIZE,cnn_train.IMAGE_SIZE, cnn_train.NUM_CHANNELS))
                    y_ = sess.run(y, feed_dict={x:x_i})
                    res = np.append(res, np.argmax(y_, 1))
            return res

def main(argv=None):
    data_test =  pd.read_csv('D:/mnist/test.csv')
    res = predict(data_test)
    results = pd.Series(res,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv(MODEL_SAVE_PATH+"submit.csv",index=False)
if __name__ == '__main__':
    tf.app.run()