import pandas as pd #数据分析
import numpy as np #科学计算
import tensorflow as tf
from pandas import Series,DataFrame
import cnn_train
MODEL_SAVE_PATH = "D:/kaggle/mnist2/model1/"
MODEL_NAME = "model.ckpt"
def verify(data_test):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [1000, \
            cnn_train.IMAGE_SIZE, cnn_train.IMAGE_SIZE, cnn_train.NUM_CHANNELS], name = 'x-input')
        y_ = tf.placeholder(tf.float32, [None, cnn_train.OUTPUT_NODE], name= 'y-input')
        y = cnn_train.inference(x, False, None)

        data_np = data_test.as_matrix()
        data_x = data_np[:,1:]
        data_x = np.multiply(data_x, 1.0/255)
        data_y = cnn_train.dense_to_one_hot(data_test.iloc[:,0].values, 10)

        xs = np.reshape(data_x[0:1000,:],(-1, cnn_train.IMAGE_SIZE,cnn_train.IMAGE_SIZE, cnn_train.NUM_CHANNELS))
        validate_feed = {x: xs, y_: data_y[0:1,:]}

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        print(xs.shape)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                accu = sess.run(accuracy, feed_dict=validate_feed)
                print("accuracy=%f"%accu)

def main(argv=None):
    data_train =  pd.read_csv('D:/kaggle/mnist/train.csv')
    res = verify(data_train)

if __name__ == '__main__':
    tf.app.run()