# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt
import os


class LSTM():
    def __init__(self,
                 hidden_size=30,  # lstm隐藏节点
                 num_layers=2,  # lstm层数
                 timestamps=10,  # 训练序列长度
                 batch_size=32,
                 training_examples=10000,  # 训练集个数
                 test_examples=1000,
                 model_path="D:/kaggle/rnn/",
                 load_model=False,
                 training_steps=10000,
                 logdir="D:/kaggle/rnn/"
                 ):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.timestamps = timestamps
        self.train_examples = training_examples
        self.test_examples = test_examples
        self.log_dir = logdir
        self.training_steps = training_steps
        self.model_path = model_path
        self.build_graph()
        self.init_op()
        if load_model:
            self.load_model(model_path)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=[None, 1, self.timestamps])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])
            # 多层lstm
            cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
                for _ in range(self.num_layers)
            ])
            # 将多层lstm连接成rnn网络并计算前向传播结果
            outputs, _ = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
            # outputs维度【batch_size,time,hidden_size】
            output = outputs[:, -1, :]  # 最后一步的输出
            self.prediction = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.prediction)
            tf.summary.scalar('loss', self.loss)
            self.train_op = tf.contrib.layers.optimize_loss(
                self.loss, tf.train.get_global_step(),
                optimizer="Adagrad", learning_rate=0.1
            )
            self.init = tf.global_variables_initializer()
            self.merged_summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    # train_X: 10000, 1, 10
    def train(self, train_X, train_y):
        # get batch data
        with self.graph.as_default():
            ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
            ds = ds.repeat().shuffle(1000).batch(self.batch_size)
            X, y = ds.make_one_shot_iterator().get_next()

        for i in range(self.training_steps):
            batch_x, batch_y = self.sess.run([X, y])
            feed_dict = {self.X: batch_x, self.y: batch_y}
            _, loss, summary_str = self.sess.run(
                [self.train_op, self.loss, self.merged_summary_op], feed_dict=feed_dict)
            if i % 1000 == 0:
                print('train step:' + str(i) + ', loss=' + str(loss))
                self.saver.save(self.sess,
                                os.path.join(self.model_path, "lstm.ckpt"),
                                global_step=tf.train.get_global_step())

    def test(self, test_X, test_y):
        with self.graph.as_default():
            ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
            ds.batch(1)
            X, y = ds.make_one_shot_iterator().get_next()  # 这里的X,y都是tensor

        predictions = []
        labels = []
        for i in range(self.test_examples):
            batch_x, batch_y = self.sess.run([X, y])
            batch_x = np.reshape(batch_x, (-1, 1, 10))
            p = self.sess.run([self.prediction], feed_dict={self.X: batch_x})
            predictions.append(p)
            labels.append(batch_y)
        predictions = np.array(predictions).squeeze()  # 删除shape中为1的维度
        labels = np.array(labels).squeeze()
        rmse = np.sqrt((predictions - labels) ** 2).mean(axis=0)
        print("mean square error is: %f" % rmse)

        # 绘图
        plt.figure()
        plt.plot(predictions, label="predictions")
        # plt.plot(labels, label="real_sin")
        plt.legend()
        plt.show()

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            RuntimeError("model_path is not exist")
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)


def get_data(seq, timestamp):
    X = []
    y = []
    for i in range(len(seq) - timestamp):
        X.append([seq[i: i + timestamp]])
        y.append([seq[i + timestamp]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == '__main__':
    sample_gap = 0.01  # 采样间隔
    lstm = LSTM(load_model=False)
    test_start = (lstm.train_examples + lstm.timestamps) * sample_gap
    test_end = test_start + (lstm.test_examples + lstm.timestamps) * sample_gap
    train_x, train_y = get_data(np.sin(np.linspace(
        0, test_start, lstm.train_examples + lstm.timestamps, dtype=np.float32),
    ), lstm.timestamps)
    test_X, test_y = get_data(np.sin(np.linspace(test_start, test_end,
                                                 lstm.test_examples + lstm.timestamps,
                                                 dtype=np.float32)), lstm.timestamps)
    lstm.train(train_x, train_y)
    lstm.test(test_X, test_y)
